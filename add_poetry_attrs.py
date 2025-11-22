"""
add emotions, sentiment, and themes to poem_interpretation_corpus.

resumability:
- provenance.jsonl tracks status per row index
- per-row outputs stored as json in out_dir/rows/{index}.json
- reruns skip successful rows automatically

labels:
- emotions: 1–3 dominant emotions from a fixed set
- sentiment: single overall sentiment
- themes: one or more canonical themes; if none fit, use the catch-all theme "others"
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
    "others",  # catch-all theme when no specific theme fits
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
    ] = Field(
        description="one or more dominant emotion labels from the fixed set, ordered by strength (most dominant first)"
    )
    sentiment: Literal["positive", "negative", "neutral"] = Field(
        description="overall sentiment label"
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
    ] = Field(
        default_factory=list,
        description='one or more theme labels from the fixed set; use "others" if no specific theme fits',
    )

    @field_validator("emotions", mode="before")
    @classmethod
    def validate_emotions(cls, val) -> List[str]:
        if isinstance(val, str):
            # model might still return a single string, normalize to list
            val = [val]

        if not isinstance(val, list):
            raise ValueError("emotions must be a list or a string")

        norm: List[str] = []
        seen: Set[str] = set()
        for item in val:
            if not isinstance(item, str):
                raise ValueError("emotions list must contain strings")
            low = item.strip().lower()
            if low not in emotion_labels:
                raise ValueError(f"invalid emotion: {low}")
            if low not in seen:
                norm.append(low)
                seen.add(low)

        if not norm:
            raise ValueError("emotions list must not be empty")

        # optional cap, keeps things compact
        return norm[:3]

    @field_validator("sentiment", mode="before")
    @classmethod
    def validate_sentiment(cls, val: str) -> str:
        if not isinstance(val, str):
            raise ValueError("sentiment must be a string")
        low = val.strip().lower()
        if low not in sentiment_labels:
            raise ValueError(f"sentiment must be one of {sentiment_labels}")
        return low

    @field_validator("themes", mode="before")
    @classmethod
    def validate_themes(cls, val) -> List[str]:
        # if model returns nothing or null, fall back to ["others"]
        if val is None:
            return ["others"]

        if isinstance(val, str):
            val = [val]

        if not isinstance(val, list):
            raise ValueError("themes must be a list or a string")

        norm: List[str] = []
        seen: Set[str] = set()
        for item in val:
            if not isinstance(item, str):
                raise ValueError("themes list must contain strings")
            low = item.strip().lower()
            if low not in theme_labels:
                raise ValueError(f"invalid theme: {low}")
            if low not in seen:
                norm.append(low)
                seen.add(low)

        # enforce at least one theme; use catch-all if none were chosen
        if not norm:
            return ["others"]

        return norm


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
    with prov_path.open("r") as file:
        for line in file:
            if not line.strip():
                continue
            data = json.loads(line)
            entries[int(data["index"])] = ProvenanceEntry(**data)
    return entries


def write_provenance(out_dir: Path, entries: Dict[int, ProvenanceEntry]) -> None:
    prov_path = out_dir / "provenance.jsonl"
    tmp_path = out_dir / "provenance.jsonl.tmp"
    with tmp_path.open("w") as file:
        for entry in entries.values():
            file.write(json.dumps(entry.__dict__) + "\n")
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

    # keep context light and predictable
    max_poem_chars = 8000
    max_interp_chars = 8000
    if len(poem) > max_poem_chars:
        poem = poem[:max_poem_chars]
    if len(interpretation) > max_interp_chars:
        interpretation = interpretation[:max_interp_chars]

    poem_missing = (not poem) or ("[mask" in poem.lower()) or ("<mask" in poem.lower())

    sys_prompt = f"""
ROLE: you are a careful literary annotator of poems.

STRICT OUTPUT CONTRACT:
- output ONLY one JSON object and nothing else (no markdown, no prose).
- the object must have exactly three keys:
  - "emotions": a list of 1 to 3 items chosen from {emotion_labels}
  - "sentiment": one of {sentiment_labels}
  - "themes": a list with ≥1 items chosen from {theme_labels}; use ["others"] if no specific theme fits.

SELECTION RULES:
- emotions: pick 1–3 dominant emotions in order of strength; avoid weak/tenuous ones.
- sentiment: overall valence of the whole poem.
- themes: broad motifs clearly supported by the text; if none fit, return ["others"].

EVIDENCE:
- base labels on the poem text only.
- if poem text is missing or masked, you may use the interpretation to decide conservatively.
""".strip()

    user_prompt = f"""
POEM METADATA
title: {title or "unknown"}
author: {author or "unknown"}
source: {row.get("source") or "unknown"}

POEM TEXT
{poem if poem else "[poem text missing]"}

INTERPRETATION (fallback only if poem missing)
{interpretation if poem_missing and interpretation else "[not provided]"}
""".strip()

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
    schema = {
        "name": "PoemAttrs",
        "schema": PoemAttrs.model_json_schema(),
        "strict": True,
    }

    try:
        doc: PoemAttrs = await backend.guardrail(
            messages=messages,
            response_model=PoemAttrs,
            json_schema=schema,
            max_retries=6,  # ↑ give the schema-off fallback time to work
            temperature=0.0,
            top_p=1.0,
            reasoning_effort="high",
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
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    prov = load_provenance(out_dir)

    ds = load_dataset("haining/poem_interpretation_corpus", split=args.split)
    total = len(ds)
    limit = int(args.limit)
    if limit > 0:
        total = min(total, limit)

    indexes_to_process: List[int] = []
    for index in range(total):
        entry = prov.get(index)
        out_file = out_dir / "rows" / f"{index}.json"
        if entry and entry.status == "success" and out_file.exists():
            continue
        indexes_to_process.append(index)

    logger.info(f"processing {len(indexes_to_process)}/{total} rows")

    backend = GuardedBackend(
        base_url=args.base_url,
        model=args.model,
        read_timeout=float(args.read_timeout),
    )

    semaphore = asyncio.Semaphore(int(args.max_concurrent))

    async def bounded(idx: int) -> None:
        async with semaphore:
            row = dict(ds[idx])
            await annotate_one(
                backend=backend,
                row=row,
                index=idx,
                out_dir=out_dir,
                prov=prov,
            )

    tasks = [bounded(i) for i in indexes_to_process]

    for fut in atqdm(
        asyncio.as_completed(tasks),
        total=len(tasks),
        desc="annotating",
    ):
        try:
            await fut
        except Exception as exc:
            logger.error(f"row failed: {exc}")

    await backend.close()
    logger.info("done")


def build_cli() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="add emotions/sentiment/themes to corpus")
    ap.add_argument(
        "--base_url", type=str, required=True, help="vllm openai api base url"
    )
    ap.add_argument("--model", type=str, default="openai/gpt-oss-120b")
    ap.add_argument("--read_timeout", type=float, default=1800.0)
    ap.add_argument("--out_dir", type=str, default="poem_attrs")
    ap.add_argument("--split", type=str, default="train")
    ap.add_argument("--max_concurrent", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0)
    return ap


def main() -> None:
    args = build_cli().parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(message)s")
    asyncio.run(run_all(args))


if __name__ == "__main__":
    main()
