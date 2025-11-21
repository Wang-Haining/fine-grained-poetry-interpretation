"""
annotate interpretation corpus with emotion, sentiment, themes.

- reads csv or jsonl
- async batched annotation via guarded backend
- pydantic-validated json outputs
- resumable via provenance.jsonl + per-sample json files

note: no custom names start with "_" per user requirement.

"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple

import pandas as pd
from pydantic import BaseModel, Field, field_validator

from guarded_backend import GuardedBackend

logger = logging.getLogger(__name__)
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO, format="[%(asctime)s] %(levelname)s: %(message)s"
    )


# ----------------------------
# label space
# ----------------------------
Emotion = Literal[
    "fear", "anger", "trust", "sadness", "disgust", "anticipation", "joy", "surprise"
]
Sentiment = Literal["positive", "negative", "neutral"]
Theme = Literal[
    "nature",
    "body",
    "death",
    "love",
    "existential",
    "identity",
    "self",
    "beauty",
    "america",
    "loss",
    "animals",
    "history",
    "memories",
    "family",
    "writing",
    "ancestry",
    "thought",
    "landscapes",
    "war",
    "time",
    "religion",
    "grief",
    "violence",
    "aging",
    "childhood",
    "desire",
    "night",
    "mothers",
    "language",
    "birds",
    "social justice",
    "music",
    "flowers",
    "politics",
    "hope",
    "heartache",
    "fathers",
    "gender",
    "environment",
    "spirituality",
    "loneliness",
    "oceans",
    "dreams",
    "survival",
    "cities",
    "earth",
    "despair",
    "anxiety",
    "weather",
    "illness",
    "home",
]


class PoemLabels(BaseModel):
    emotion: List[Emotion] = Field(
        description="1-3 dominant emotions from the allowed set"
    )
    sentiment: Sentiment = Field(
        description="overall sentiment: positive, negative, or neutral"
    )
    themes: List[Theme] = Field(description="0-5 themes from the allowed set")

    @field_validator("emotion")
    @classmethod
    def nonempty_emotion(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("emotion must contain at least one label")
        if len(v) > 3:
            raise ValueError("emotion must contain at most 3 labels")
        return v

    @field_validator("themes")
    @classmethod
    def max_themes(cls, v: List[str]) -> List[str]:
        if len(v) > 5:
            raise ValueError("themes must contain at most 5 labels")
        return v


# ----------------------------
# provenance for resumability
# ----------------------------
def load_provenance(save_dir: Path) -> Dict[str, Dict[str, Any]]:
    prov_path = save_dir / "provenance.jsonl"
    if not prov_path.exists():
        return {}
    out: Dict[str, Dict[str, Any]] = {}
    with prov_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(row["sample_id"])] = row
    return out


def update_provenance(save_dir: Path, rec: Dict[str, Any]) -> None:
    prov_path = save_dir / "provenance.jsonl"
    tmp_path = save_dir / "provenance.jsonl.tmp"

    current = load_provenance(save_dir)
    current[str(rec["sample_id"])] = rec

    with tmp_path.open("w") as f:
        for r in current.values():
            f.write(json.dumps(r) + "\n")

    tmp_path.replace(prov_path)


def get_samples_to_process(sample_ids: List[str], save_dir: Path) -> List[str]:
    prov = load_provenance(save_dir)
    todo = []
    for sid in sample_ids:
        out_path = save_dir / "samples" / f"{sid}.json"
        if prov.get(sid, {}).get("status") == "success" and out_path.exists():
            continue
        todo.append(sid)
    return todo


# ----------------------------
# prompting
# ----------------------------
def build_prompts(
    poem: str,
    interpretation: Optional[str],
    *,
    use_interpretation: bool,
) -> Tuple[str, str]:
    allowed_emotions = [
        "fear",
        "anger",
        "trust",
        "sadness",
        "disgust",
        "anticipation",
        "joy",
        "surprise",
    ]
    allowed_sentiments = ["positive", "negative", "neutral"]
    allowed_themes = [
        "nature",
        "body",
        "death",
        "love",
        "existential",
        "identity",
        "self",
        "beauty",
        "america",
        "loss",
        "animals",
        "history",
        "memories",
        "family",
        "writing",
        "ancestry",
        "thought",
        "landscapes",
        "war",
        "time",
        "religion",
        "grief",
        "violence",
        "aging",
        "childhood",
        "desire",
        "night",
        "mothers",
        "language",
        "birds",
        "social justice",
        "music",
        "flowers",
        "politics",
        "hope",
        "heartache",
        "fathers",
        "gender",
        "environment",
        "spirituality",
        "loneliness",
        "oceans",
        "dreams",
        "survival",
        "cities",
        "earth",
        "despair",
        "anxiety",
        "weather",
        "illness",
        "home",
    ]

    sys_prompt = (
        "ROLE: you are an expert poetry annotator.\n"
        "GOAL: assign emotion, sentiment, and themes for the poem.\n\n"
        "STRICT OUTPUT CONTRACT:\n"
        "- output ONLY one JSON object and nothing else.\n"
        "- JSON must have exactly three keys: emotion, sentiment, themes.\n"
        "- do not add extra fields. do not wrap in markdown.\n"
        "- do not explain your reasoning.\n\n"
        "LABEL SELECTION RULES:\n"
        "1) EMOTION:\n"
        f"   - choose 1 to 3 dominant emotions from: {allowed_emotions}\n"
        "   - dominant = repeatedly evoked in imagery, tone, or speaker stance.\n"
        "   - if mixed, pick the strongest first.\n"
        "2) SENTIMENT:\n"
        f"   - choose exactly one from: {allowed_sentiments}\n"
        "   - sentiment reflects overall valence after reading the whole poem.\n"
        "3) THEMES:\n"
        f"   - choose 0 to 5 themes from: {allowed_themes}\n"
        "   - themes must be clearly text-supported, not speculative.\n"
        "   - if none fit, return an empty list.\n\n"
        "CONSERVATISM:\n"
        "- when uncertain, prefer fewer labels.\n"
        "- never invent labels outside the allowed sets.\n"
    )

    user_parts = [
        "TASK:\n"
        "Read the poem. Then return a JSON object that follows the contract.\n\n"
        "POEM:\n"
        f"{poem.strip()}",
    ]

    if use_interpretation and interpretation:
        user_parts += [
            "\nOPTIONAL CONTEXT (model interpretation). "
            "Use only if it helps disambiguate; the poem remains primary evidence:\n"
            f"{interpretation.strip()}",
        ]

    user_prompt = "\n".join(user_parts).strip()
    return sys_prompt, user_prompt


async def annotate_one(
    *,
    backend: GuardedBackend,
    sample_id: str,
    poem: str,
    interpretation: Optional[str],
    use_interpretation: bool,
    save_dir: Path,
    temperature: float,
) -> PoemLabels:
    sys_prompt, user_prompt = build_prompts(
        poem, interpretation, use_interpretation=use_interpretation
    )

    schema = {
        "name": "PoemLabels",
        "schema": PoemLabels.model_json_schema(),
        "strict": True,
    }

    labels: PoemLabels = await backend.guardrail(
        messages=[
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": user_prompt},
        ],
        response_model=PoemLabels,
        json_schema=schema,
        max_retries=3,
        temperature=temperature,
        top_p=1.0,
        max_tokens=256,
        reasoning_effort="high",
    )

    out_path = save_dir / "samples" / f"{sample_id}.json"
    out_path.write_text(labels.model_dump_json(indent=2))
    return labels


# ----------------------------
# io
# ----------------------------
def read_corpus(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".jsonl", ".json"}:
        rows = []
        with path.open("r") as f:
            for line in f:
                if not line.strip():
                    continue
                rows.append(json.loads(line))
        return pd.DataFrame(rows)
    return pd.read_csv(path)


def infer_sample_ids(df: pd.DataFrame, id_col: Optional[str]) -> List[str]:
    if id_col and id_col in df.columns:
        return df[id_col].astype(str).tolist()
    for cand in ["poem_id", "id", "sample_id"]:
        if cand in df.columns:
            return df[cand].astype(str).tolist()
    return [f"sample_{i}" for i in range(len(df))]


# ----------------------------
# main async
# ----------------------------
async def main_async(args: argparse.Namespace) -> None:
    input_path = Path(args.input_path)
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "samples").mkdir(parents=True, exist_ok=True)

    df = read_corpus(input_path)
    df.columns = [c.lower() for c in df.columns]

    sample_ids = infer_sample_ids(df, args.id_col)
    df["sample_id"] = sample_ids

    todo = get_samples_to_process(sample_ids, save_dir)
    logger.info(f"processing {len(todo)}/{len(sample_ids)} poems")

    backend = GuardedBackend(
        base_url=args.base_url,
        model=args.model,
        read_timeout=args.read_timeout,
    )

    sem = asyncio.Semaphore(args.max_concurrent)
    results: Dict[str, PoemLabels] = {}

    async def bounded_run(row: Dict[str, Any]) -> None:
        sid = str(row["sample_id"])
        poem = str(row.get("poem", "") or "")
        interp = (
            str(row.get("interpretation", "") or "")
            if args.use_interpretation
            else None
        )

        if sid not in todo:
            return

        update_provenance(save_dir, {"sample_id": sid, "status": "running"})
        try:
            async with sem:
                labels = await annotate_one(
                    backend=backend,
                    sample_id=sid,
                    poem=poem,
                    interpretation=interp,
                    use_interpretation=args.use_interpretation,
                    save_dir=save_dir,
                    temperature=args.temperature,
                )
            results[sid] = labels
            update_provenance(save_dir, {"sample_id": sid, "status": "success"})
        except Exception as e:
            update_provenance(
                save_dir,
                {"sample_id": sid, "status": "failed", "error": str(e)[:1000]},
            )
            logger.error(f"{sid} failed: {e}")

    tasks = [bounded_run(df.iloc[i].to_dict()) for i in range(len(df))]
    await asyncio.gather(*tasks)

    await backend.close()

    # merge any existing sample files (resumable final export)
    emotions, sentiments, themes = [], [], []
    for sid in sample_ids:
        sp = save_dir / "samples" / f"{sid}.json"
        if sp.exists():
            obj = PoemLabels.model_validate_json(sp.read_text())
            emotions.append(";".join(obj.emotion))
            sentiments.append(obj.sentiment)
            themes.append(";".join(obj.themes))
        else:
            emotions.append(None)
            sentiments.append(None)
            themes.append(None)

    df["emotion"] = emotions
    df["sentiment"] = sentiments
    df["themes"] = themes

    out_path = save_dir / args.output_name
    df.to_csv(out_path, index=False)
    logger.info(f"wrote annotated corpus to {out_path}")


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_path", required=True, help="csv or jsonl corpus")
    ap.add_argument("--save_dir", default="labels_out", help="output dir")
    ap.add_argument("--output_name", default="corpus_annotated.csv")
    ap.add_argument("--id_col", default=None, help="optional id column name")

    ap.add_argument("--base_url", default="http://127.0.0.1:8020")
    ap.add_argument("--model", default="openai/gpt-oss-120b")
    ap.add_argument("--read_timeout", type=float, default=1800.0)

    ap.add_argument("--max_concurrent", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--use_interpretation", action="store_true")

    return ap


if __name__ == "__main__":
    args = build_cli().parse_args()
    asyncio.run(main_async(args))
