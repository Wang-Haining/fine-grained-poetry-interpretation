"""
merge per-row or per-split json annotations into the original dataset.
writes split-wise parquet with five new columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_dataset


def _coerce_int(x: Any) -> Optional[int]:
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, (int,)):
            return int(x)
        if isinstance(x, float) and x.is_integer():
            return int(x)
        if isinstance(x, str) and x.strip().isdigit():
            return int(x.strip())
    except Exception:
        pass
    return None


def _extract_index(d: Dict[str, Any], fallback: int) -> int:
    # try common top level keys
    for k in ("row_index", "index", "row_id", "id"):
        v = d.get(k)
        idx = _coerce_int(v)
        if idx is not None:
            return idx
    # try nested meta
    meta = d.get("meta", {})
    if isinstance(meta, dict):
        for k in ("row_index", "index", "row_id", "id"):
            v = meta.get(k)
            idx = _coerce_int(v)
            if idx is not None:
                return idx
    return fallback


def load_split_payloads(attrs_root: Path, split: str) -> Dict[int, Dict[str, Any]]:
    """
    Load annotations for a split from either:
      1) per split jsonl at {attrs_root}/{split}.jsonl, or
      2) per row json files under {attrs_root}/{split}/rows/*.json

    If both exist, per row files take precedence for the same index.
    """
    out: Dict[int, Dict[str, Any]] = {}

    # case 1: jsonl
    jl = attrs_root / f"{split}.jsonl"
    if jl.exists():
        try:
            with jl.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        d = json.loads(line)
                    except Exception:
                        continue
                    idx = _extract_index(d, fallback=i)
                    out[idx] = d
        except Exception:
            pass

    # case 2: rows/*.json overrides
    rows_dir = attrs_root / split / "rows"
    if rows_dir.exists():
        for p in rows_dir.glob("*.json"):
            try:
                d = json.loads(p.read_text(encoding="utf-8"))
                stem_idx = _coerce_int(p.stem)
                idx = _extract_index(
                    d, fallback=stem_idx if stem_idx is not None else 0
                )
                out[idx] = d
            except Exception:
                continue

    return out


def merge_split(ds_id: str, split: str, attrs_root: Path, out_root: Path) -> Path:
    ds = load_dataset(ds_id, split=split)
    df = ds.to_pandas()
    df.columns = [c.lower() for c in df.columns]

    payloads = load_split_payloads(attrs_root, split)

    # init new columns
    for col in ("emotions", "primary_emotion", "sentiment", "themes", "themes_50"):
        if col not in df.columns:
            df[col] = None

    # pick a stable key on the dataframe side
    df_keys: List[Optional[int]] = []
    has_row_index = "row_index" in df.columns
    has_idx_level = "__index_level_0__" in df.columns
    has_id = "id" in df.columns

    for i in range(len(df)):
        key: Optional[int] = None
        if has_row_index:
            key = _coerce_int(df.iloc[i]["row_index"])
        if key is None and has_idx_level:
            key = _coerce_int(df.iloc[i]["__index_level_0__"])
        if key is None and has_id:
            key = _coerce_int(df.iloc[i]["id"])
        if key is None:
            key = i
        df_keys.append(key)

    # fill values from payloads
    for i, key in enumerate(df_keys):
        d = payloads.get(int(key))
        if not d:
            continue
        df.at[i, "emotions"] = d.get("emotions")
        df.at[i, "primary_emotion"] = d.get("primary_emotion")
        df.at[i, "sentiment"] = d.get("sentiment")
        df.at[i, "themes"] = d.get("themes")
        df.at[i, "themes_50"] = d.get("themes_50")

    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{split}.parquet"
    df.to_parquet(out_path, index=False)
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_id",
        type=str,
        default="haining/structured_poem_interpretation_corpus",
    )
    ap.add_argument("--attrs_dir", type=str, default="poem_attrs")
    ap.add_argument("--out_dir", type=str, default="poem_attrs/merged")
    ap.add_argument("--splits", type=str, default="train,validation,test")
    args = ap.parse_args()

    attrs_root = Path(args.attrs_dir)
    out_root = Path(args.out_dir)
    splits = [s.strip() for s in args.splits.split(",") if s.strip()]

    written: List[Path] = []
    for split in splits:
        p = merge_split(args.dataset_id, split, attrs_root, out_root)
        written.append(p)

    print("wrote:")
    for p in written:
        print(p)
