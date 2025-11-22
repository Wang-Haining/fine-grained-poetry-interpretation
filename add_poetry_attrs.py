"""
add emotions, sentiment, and themes to poem_interpretation_corpus.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set

from datasets import load_dataset
from pydantic import BaseModel, Field, field_validator
from tqdm.asyncio import tqdm as atqdm

from guarded_backend import GuardedBackend

logger = logging.getLogger(__name__)


emotion_labels = [
    "fear",
    "anger",
    "trust",
    "sadness",
    "disgust",
    "anticipation",
    "joy",
    "surprise",
]

sentiment_labels = ["positive", "negative", "neutral"]

theme_labels = [
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
    "others",
]


class PoemAttrs(BaseModel):
    emotions: List[
        Literal[
            "fear",
            "anger",
            "trust",
            "sadness",
            "disgust",
            "anticipation",
            "joy",
            "surprise",
        ]
    ] = Field(description="1-3 dominant emotions")
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="overall sentiment"
    )
    themes: List[
        Literal[
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
            "others",
        ]
    ] = Field(default_factory=lambda: ["others"], description="themes or 'others'")

    @field_validator("emotions", mode="before")
    @classmethod
    def validate_emotions(cls, val) -> List[str]:
        if isinstance(val, str):
            val = [val]
        if not isinstance(val, list):
            raise ValueError("emotions must be a list")

        norm: List[str] = []
        seen: Set[str] = set()
        for item in val:
            low = str(item).strip().lower()
            if low in emotion_labels and low not in seen:
                norm.append(low)
                seen.add(low)

        if not norm:
            raise ValueError("emotions must not be empty")
        return norm[:3]

    @field_validator("sentiment", mode="before")
    @classmethod
    def validate_sentiment(cls, val: str) -> str:
        low = str(val).strip().lower()
        if low not in sentiment_labels:
            raise ValueError(f"invalid sentiment: {low}")
        return low

    @field_validator("themes", mode="before")
    @classmethod
    def validate_themes(cls, val) -> List[str]:
        if val is None:
            return ["others"]
        if isinstance(val, str):
            val = [val]
        if not isinstance(val, list):
            return ["others"]

        norm: List[str] = []
        seen: Set[str] = set()
        for item in val:
            low = str(item).strip().lower()
            if low in theme_labels and low not in seen:
                norm.append(low)
                seen.add(low)

        return norm if norm else ["others"]


@dataclass
class ProvenanceEntry:
    index: int
    status: Literal["pending", "running", "success", "failed"]
    timestamp: str
    error: Optional[str] = None
    output_path: Optional[str] = None


def load_provenance(out_dir: Path) -> Dict[int, ProvenanceEntry]:
    prov_path = out_dir / "provenance.jsonl"
    if not prov_path.exists():
        return {}
    entries: Dict[int, ProvenanceEntry] = {}
    with prov_path.open("r") as f:
        for line in f:
            if not line.strip():
                continue
            data = json.loads(line)
            entries[int(data["index"])] = ProvenanceEntry(**data)
    return entries


def write_provenance(out_dir: Path, entries: Dict[int, ProvenanceEntry]) -> None:
    prov_path = out_dir / "provenance.jsonl"
    tmp_path = out_dir / "provenance.jsonl.tmp"
    with tmp_path.open("w") as f:
        for entry in entries.values():
            f.write(json.dumps(entry.__dict__) + "\n")
    tmp_path.replace(prov_path)


def mark_status(
    out_dir: Path,
    entries: Dict[int, ProvenanceEntry],
    index: int,
    status: Literal["pending", "running", "success", "failed"],
    error: Optional[str] = None,
    output_path: Optional[str] = None,
) -> None:
    entries[index] = ProvenanceEntry(
        index=index,
        status=status,
        timestamp=datetime.now().isoformat(),
        error=error,
        output_path=output_path,
    )
    write_provenance(out_dir, entries)


def build_messages(row: Dict[str, str]) -> List[Dict[str, str]]:
    title = (row.get("title") or "").strip()
    author = (row.get("author") or "").strip()
    poem = (row.get("poem") or "").strip()
    interpretation = (row.get("interpretation") or "").strip()

    max_poem_chars = 8000
    max_interp_chars = 8000
    if len(poem) > max_poem_chars:
        poem = poem[:max_poem_chars]
    if len(interpretation) > max_interp_chars:
        interpretation = interpretation[:max_interp_chars]

    poem_missing = (not poem) or ("[mask" in poem.lower()) or ("<mask" in poem.lower())

    sys_prompt = f"""ROLE: literary annotator.

OUTPUT: JSON object with exactly three keys:
- "emotions": list of 1-3 from {emotion_labels}
- "sentiment": one of {sentiment_labels}
- "themes": list from {theme_labels} or ["others"]

RULES:
- emotions: pick 1-3 dominant emotions, strongest first
- sentiment: overall valence
- themes: text-supported themes; use ["others"] if none fit
- output ONLY the JSON object, no markdown, no explanation"""

    user_prompt = f"""title: {title or "unknown"}
author: {author or "unknown"}

POEM:
{poem if poem else "[missing]"}

{f"INTERPRETATION (fallback only):\n{interpretation}" if poem_missing and interpretation else ""}"""

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def annotate_one(
    *,
    backend: GuardedBackend,
    row: Dict[str, str],
    index: int,
    out_dir: Path,
    prov: Dict[int, ProvenanceEntry],
) -> None:
    rows_dir = out_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    out_path = rows_dir / f"{index}.json"

    mark_status(out_dir, prov, index, "running")

    messages = build_messages(row)

    try:
        doc: PoemAttrs = await backend.guardrail(
            messages=messages,
            response_model=PoemAttrs,
            max_retries=6,
            temperature=0.0,
            top_p=1.0,
            max_tokens=256,
        )

        payload = {
            "index": index,
            "emotions": doc.emotions,
            "primary_emotion": doc.emotions[0] if doc.emotions else None,
            "sentiment": doc.sentiment,
            "themes": doc.themes,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False))
        mark_status(out_dir, prov, index, "success", output_path=str(out_path))
    except Exception as exc:
        mark_status(out_dir, prov, index, "failed", error=str(exc))
        raise


async def run_all(args: argparse.Namespace) -> None:
    root_out = Path(args.out_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    ds_dict = load_dataset("haining/poem_interpretation_corpus")
    available = list(ds_dict.keys())
    if args.splits.lower() == "all":
        splits = available
    else:
        wanted = [s.strip() for s in args.splits.split(",") if s.strip()]
        splits = [s for s in wanted if s in available]
    logger.info(f"available splits: {available}")
    logger.info(f"selected splits: {splits}")

    backend = GuardedBackend(
        base_url=args.base_url,
        model=args.model,
        read_timeout=float(args.read_timeout),
    )

    for split in splits:
        out_dir = root_out / split
        rows_dir = out_dir / "rows"
        rows_dir.mkdir(parents=True, exist_ok=True)

        prov = load_provenance(out_dir)
        ds = ds_dict[split]
        total = len(ds)
        if args.limit and args.limit > 0:
            total = min(total, int(args.limit))

        indexes_to_process: List[int] = []
        for index in range(total):
            entry = prov.get(index)
            out_file = rows_dir / f"{index}.json"
            if entry and entry.status == "success" and out_file.exists():
                continue
            indexes_to_process.append(index)

        logger.info(f"[{split}] processing {len(indexes_to_process)}/{total} rows")
        sem = asyncio.Semaphore(int(args.max_concurrent))

        async def bounded(index: int) -> None:
            async with sem:
                row = dict(ds[index])
                await annotate_one(
                    backend=backend,
                    row=row,
                    index=index,
                    out_dir=out_dir,
                    prov=prov,
                )

        tasks = [asyncio.create_task(bounded(i)) for i in indexes_to_process]

        for fut in atqdm.as_completed(
            tasks, total=len(tasks), desc=f"annotating:{split}"
        ):
            try:
                await fut
            except Exception as exc:
                logger.error(f"[{split}] row failed: {exc}")

        logger.info(f"[{split}] complete")

    await backend.close()
    logger.info("all splits done")


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", type=str, required=True)
    ap.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    ap.add_argument("--read_timeout", type=float, default=1800.0)
    ap.add_argument("--out_dir", type=str, default="poem_attrs")
    ap.add_argument("--splits", type=str, default="all")
    ap.add_argument("--max_concurrent", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    return ap


def main() -> None:
    args = build_cli().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
