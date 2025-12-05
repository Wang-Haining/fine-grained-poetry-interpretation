"""
add emotions, sentiment, and themes to poem_interpretation_corpus.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Literal, Optional, Set, Tuple

from datasets import load_dataset
from pydantic import BaseModel, Field, field_validator
from tqdm.asyncio import tqdm as atqdm

from guarded_backend import GuardedBackend

logger = logging.getLogger(__name__)

# label spaces
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

# fixed 50 themes used to derive themes_50
themes_50 = [
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
themes_50_set = set(themes_50)

IMPORTANT_KEYS = ["emotions", "primary_emotion", "sentiment", "themes", "themes_50"]


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
    themes: List[str] = Field(
        default_factory=list,
        description="open-vocabulary themes (0-5 themes)",
    )

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
            return []
        if isinstance(val, str):
            val = [val]
        if not isinstance(val, list):
            return []
        norm: List[str] = []
        seen: Set[str] = set()
        for item in val:
            clean = str(item).strip().lower()
            if clean and clean not in seen:
                norm.append(clean)
                seen.add(clean)
        return norm[:5]


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


def is_nonempty_payload(d: dict) -> bool:
    def _empty(v):
        if v is None:
            return True
        if isinstance(v, float) and math.isnan(v):
            return True
        if isinstance(v, str) and v.strip() == "":
            return True
        if isinstance(v, (list, dict)) and len(v) == 0:
            return True
        return False

    return any(not _empty(d.get(k)) for k in IMPORTANT_KEYS)


def is_filled_json(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        d = json.loads(path.read_text())
    except Exception:
        return False
    return is_nonempty_payload(d)


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

    sys_prompt = (
        "ROLE: literary annotator.\n\n"
        "OUTPUT: JSON object with exactly three keys:\n"
        f'- "emotions": list of 1-3 from {emotion_labels}\n'
        f'- "sentiment": one of {sentiment_labels}\n'
        '- "themes": list of 0-5 theme strings (open vocabulary)\n\n'
        "RULES:\n"
        "- emotions: pick 1-3 dominant emotions, strongest first\n"
        "- sentiment: overall valence\n"
        "- themes: generate your own theme labels that capture the poem's content\n"
        "  - use concise, lowercase labels (1-3 words each)\n"
        "  - be specific and descriptive\n"
        "  - return empty list if no clear themes\n"
        "- output ONLY the JSON object, no markdown, no explanation"
    )

    user_prompt = (
        f"title: {title or 'unknown'}\n"
        f"author: {author or 'unknown'}\n\n"
        f"POEM:\n{poem if poem else '[missing]'}\n\n"
        f"{('INTERPRETATION (fallback only):\n' + interpretation) if poem_missing and interpretation else ''}"
    )

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
    split: str,
    log_failures: Optional[Path],
) -> None:
    rows_dir = out_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)
    out_path = rows_dir / f"{index}.json"

    # skip if already filled
    if is_filled_json(out_path):
        return

    mark_status(out_dir, prov, index, "running")

    messages = build_messages(row)

    try:
        doc: PoemAttrs = await backend.guardrail(
            messages=messages,
            response_model=PoemAttrs,
            max_retries=6,
            temperature=0.0,
            top_p=1.0,
            max_tokens=4096,
            reasoning_effort="medium",
        )

        open_themes = doc.themes
        only_50 = [t for t in open_themes if t in themes_50_set]

        payload = {
            "index": index,
            "row_index": index,  # convenience for downstream join
            "emotions": doc.emotions,
            "primary_emotion": doc.emotions[0] if doc.emotions else None,
            "sentiment": doc.sentiment,
            "themes": open_themes,
            "themes_50": only_50,
        }
        out_path.write_text(json.dumps(payload, ensure_ascii=False))
        mark_status(out_dir, prov, index, "success", output_path=str(out_path))
    except Exception as exc:
        mark_status(out_dir, prov, index, "failed", error=str(exc))
        if log_failures is not None:
            log_failures.parent.mkdir(parents=True, exist_ok=True)
            with log_failures.open("a", newline="") as f:
                w = csv.writer(f)
                w.writerow(
                    [
                        split,
                        index,
                        type(exc).__name__,
                        str(exc)[:500],
                        int(datetime.now().timestamp()),
                    ]
                )
        raise


def load_targets_csv(path: Path) -> Dict[str, Set[int]]:
    """
    return mapping split -> set(row_index) for targeted backfill.
    accepts columns: split,row_index or split,index
    """
    import pandas as pd

    df = pd.read_csv(path)
    cols = {c.lower(): c for c in df.columns}
    if "split" not in cols:
        raise ValueError("targets csv must have a 'split' column")
    if "row_index" in cols:
        idx_col = cols["row_index"]
    elif "index" in cols:
        idx_col = cols["index"]
    else:
        raise ValueError("targets csv must have 'row_index' or 'index' column")
    out: Dict[str, Set[int]] = {}
    for r in df.itertuples(index=False):
        s = getattr(r, cols["split"])
        i = int(getattr(r, idx_col))
        out.setdefault(str(s), set()).add(i)
    return out


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

    targets_map: Dict[str, Set[int]] = {}
    if args.targets_csv:
        targets_map = load_targets_csv(Path(args.targets_csv))
        logger.info(f"loaded targets for splits: {sorted(targets_map.keys())}")

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

        # build candidate indices
        if args.limit and args.limit > 0:
            total = min(total, int(args.limit))
        candidate_indices = list(range(total))

        # restrict to targets if provided
        if targets_map:
            only = targets_map.get(split, set())
            candidate_indices = [i for i in candidate_indices if i in only]

        # skip filled if requested
        if args.skip_filled:
            candidate_indices = [
                i
                for i in candidate_indices
                if not is_filled_json(rows_dir / f"{i}.json")
            ]

        logger.info(f"[{split}] processing {len(candidate_indices)}/{len(ds)} rows")
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
                    split=split,
                    log_failures=Path(args.log_failures) if args.log_failures else None,
                )

        tasks = [asyncio.create_task(bounded(i)) for i in candidate_indices]

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
    # new controls
    ap.add_argument(
        "--targets_csv",
        type=str,
        default=None,
        help="csv with split,row_index (or split,index)",
    )
    ap.add_argument(
        "--skip_filled",
        action="store_true",
        help="skip if an existing json already has non-empty fields",
    )
    ap.add_argument(
        "--log_failures", type=str, default=None, help="csv path to append failures"
    )
    return ap


def main() -> None:
    args = build_cli().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
