"""
add emotions, sentiment, themes, and themes_50 to poem_interpretation_corpus.
robust to missing row ids; resume-safe; supports random sampling.
"""

from __future__ import annotations

import argparse
import asyncio
import csv
import json
import logging
import math
import os
import random
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
emotion_labels: List[str] = [
    "fear",
    "anger",
    "trust",
    "sadness",
    "disgust",
    "anticipation",
    "joy",
    "surprise",
]
sentiment_labels: List[str] = ["positive", "negative", "neutral"]

# fixed 50 themes used for strict themes_50
themes_50: List[str] = [
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
    ] = Field(description="1-3 dominant emotions, strongest first")

    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="overall sentiment"
    )

    themes: List[str] = Field(
        default_factory=list,
        description="open-vocabulary themes (0-5 items)",
    )

    themes_50: List[
        Literal[  # strict whitelist to avoid noisy mapping later
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
    ] = Field(
        default_factory=list,
        description="subset of fixed 50 themes (0-5 items), only from the list provided",
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

    @field_validator("themes_50", mode="before")
    @classmethod
    def validate_themes_50(cls, val) -> List[str]:
        if val is None:
            return []
        if isinstance(val, str):
            val = [val]
        if not isinstance(val, list):
            return []
        norm: List[str] = []
        seen: Set[str] = set()
        for item in val:
            s = str(item).strip().lower()
            if s in themes_50_set and s not in seen:
                norm.append(s)
                seen.add(s)
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
        timestamp=datetime.now().isoformat(timespec="seconds"),
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


def detect_row_id(row: Dict[str, object], fallback_idx: int) -> int:
    """
    choose a stable id for this row for downstream join.
    prefers 'row_index', else '__index_level_0__', else 'id', else enumerate index.
    """
    for key in ("row_index", "__index_level_0__", "id"):
        if key in row:
            try:
                return int(row[key])  # may be numpy types
            except Exception:
                pass
    return int(fallback_idx)


def build_messages(row: Dict[str, str]) -> List[Dict[str, str]]:
    title = (row.get("title") or "").strip()
    author = (row.get("author") or "").strip()
    poem = (row.get("poem") or "").strip()
    interpretation = (row.get("interpretation") or "").strip()

    # keep payload small to avoid context bloat
    max_poem_chars = 6000
    max_interp_chars = 4000
    if len(poem) > max_poem_chars:
        poem = poem[:max_poem_chars]
    if len(interpretation) > max_interp_chars:
        interpretation = interpretation[:max_interp_chars]

    poem_missing = (not poem) or ("[mask" in poem.lower()) or ("<mask" in poem.lower())

    sys_prompt = (
        "role: literary annotator\n\n"
        "output: return a json object with exactly four keys:\n"
        f'- "emotions": list of 1-3 from {emotion_labels}\n'
        f'- "sentiment": one of {sentiment_labels}\n'
        '- "themes": list of 0-5 short open-vocabulary theme strings\n'
        f'- "themes_50": list of 0-5 from this fixed set only: {themes_50}\n\n'
        "rules:\n"
        "- pick 1-3 dominant emotions, strongest first\n"
        "- sentiment is overall valence\n"
        "- for themes: concise, lowercase, specific; return [] if none are clear\n"
        "- for themes_50: choose only from the fixed set; return [] if none fit\n"
        "- output only the json object with those four keys"
    )

    user_prompt = (
        f"title: {title or 'unknown'}\n"
        f"author: {author or 'unknown'}\n\n"
        f"poem:\n{poem if poem else '[missing]'}\n\n"
        f"{('interpretation (fallback only):\n' + interpretation) if poem_missing and interpretation else ''}"
    )

    return [
        {"role": "system", "content": sys_prompt},
        {"role": "user", "content": user_prompt},
    ]


async def annotate_one(
    *,
    backend: GuardedBackend,
    row: Dict[str, object],
    index: int,
    out_dir: Path,
    prov: Dict[int, ProvenanceEntry],
    split: str,
    log_failures: Optional[Path],
) -> None:
    rows_dir = out_dir / "rows"
    rows_dir.mkdir(parents=True, exist_ok=True)

    out_path = rows_dir / f"{index}.json"
    if is_filled_json(out_path):
        return

    mark_status(out_dir, prov, index, "running")
    messages = build_messages(row)

    try:
        # guarded backend should retry, but we add a light belt-and-suspenders loop
        attempts = 0
        last_err: Optional[Exception] = None
        while attempts < 3:
            try:
                doc: PoemAttrs = await backend.guardrail(
                    messages=messages,
                    response_model=PoemAttrs,
                    max_retries=3,
                    temperature=0.0,
                    top_p=1.0,
                    max_tokens=1024,
                    reasoning_effort="medium",
                )
                # success
                payload = {
                    "index": index,
                    "row_index": index,  # stable join key we control
                    "emotions": doc.emotions,
                    "primary_emotion": doc.emotions[0] if doc.emotions else None,
                    "sentiment": doc.sentiment,
                    "themes": doc.themes,
                    "themes_50": doc.themes_50,
                }
                out_path.write_text(json.dumps(payload, ensure_ascii=False))
                mark_status(out_dir, prov, index, "success", output_path=str(out_path))
                return
            except Exception as inner:
                last_err = inner
                await asyncio.sleep(1.5 * (attempts + 1))
                attempts += 1
        # if we get here, all tries failed
        raise last_err or RuntimeError("unknown annotation failure")
    except Exception as exc:
        mark_status(out_dir, prov, index, "failed", error=str(exc))
        if log_failures is not None:
            log_failures.parent.mkdir(parents=True, exist_ok=True)
            if not log_failures.exists():
                with log_failures.open("w", newline="") as f:
                    csv.writer(f).writerow(
                        ["split", "row_index", "error_type", "error", "ts"]
                    )
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


async def run_all(args: argparse.Namespace) -> None:
    root_out = Path(args.out_dir)
    root_out.mkdir(parents=True, exist_ok=True)

    ds_dict = load_dataset(args.dataset_id)
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

    failure_path = Path(args.log_failures) if args.log_failures else None
    if failure_path is not None and not failure_path.exists():
        failure_path.parent.mkdir(parents=True, exist_ok=True)
        with failure_path.open("w", newline="") as f:
            csv.writer(f).writerow(["split", "row_index", "error_type", "error", "ts"])

    for split in splits:
        out_dir = root_out / split
        rows_dir = out_dir / "rows"
        rows_dir.mkdir(parents=True, exist_ok=True)

        prov = load_provenance(out_dir)
        ds = ds_dict[split]
        n_total = len(ds)

        all_indices = list(range(n_total))
        if args.random_sample and args.limit and args.limit > 0:
            random.seed(args.seed)
            candidate_indices = random.sample(all_indices, k=min(args.limit, n_total))
        else:
            candidate_indices = all_indices[: (args.limit or n_total)]

        if args.skip_filled:
            before = len(candidate_indices)
            candidate_indices = [
                i
                for i in candidate_indices
                if not is_filled_json(rows_dir / f"{i}.json")
            ]
            logger.info(
                f"[{split}] skip_filled pruned {before - len(candidate_indices)} rows"
            )

        logger.info(f"[{split}] processing {len(candidate_indices)}/{n_total} rows")
        sem = asyncio.Semaphore(int(args.max_concurrent))

        async def bounded(row_idx: int) -> None:
            async with sem:
                row = dict(ds[row_idx])
                stable_id = detect_row_id(row, row_idx)
                await annotate_one(
                    backend=backend,
                    row=row,
                    index=stable_id,
                    out_dir=out_dir,
                    prov=prov,
                    split=split,
                    log_failures=failure_path,
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
    ap.add_argument(
        "--dataset_id", type=str, default="haining/poem_interpretation_corpus"
    )
    ap.add_argument("--base_url", type=str, required=True)
    ap.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    ap.add_argument("--read_timeout", type=float, default=1800.0)
    ap.add_argument("--out_dir", type=str, default="poem_attrs")
    ap.add_argument("--splits", type=str, default="all")
    ap.add_argument("--max_concurrent", type=int, default=32)
    ap.add_argument("--limit", type=int, default=1000)  # default to 1k as requested
    ap.add_argument("--random_sample", action="store_true", help="sample random rows")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--skip_filled",
        action="store_true",
        help="skip if an existing json already has non-empty fields",
    )
    ap.add_argument("--log_failures", type=str, default="poem_attrs/failures.csv")
    return ap


if __name__ == "__main__":
    args = build_cli().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    asyncio.run(run_all(args))
