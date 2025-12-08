"""
merge per-row json annotations into the original dataset.
writes split-wise parquet with five new columns.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
from datasets import load_dataset


def load_row_payloads(rows_dir: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not rows_dir.exists():
        return out
    for p in rows_dir.glob("*.json"):
        try:
            d = json.loads(p.read_text())
            idx = int(d.get("row_index", d.get("index", p.stem)))
            out[idx] = d
        except Exception:
            continue
    return out


def merge_split(ds_id: str, split: str, attrs_root: Path, out_root: Path) -> Path:
    ds = load_dataset(ds_id, split=split)
    df = ds.to_pandas()
    df.columns = [c.lower() for c in df.columns]  # user prefers lower case cols

    rows_dir = attrs_root / split / "rows"
    payloads = load_row_payloads(rows_dir)

    # init new columns
    df["emotions"] = None
    df["primary_emotion"] = None
    df["sentiment"] = None
    df["themes"] = None
    df["themes_50"] = None

    # join by our stable integer key when available
    # fall back to enumerate index
    for i in range(len(df)):
        key = None
        for k in ("row_index", "__index_level_0__", "id"):
            if k in df.columns:
                try:
                    key = int(df.iloc[i][k])
                    break
                except Exception:
                    pass
        if key is None:
            key = i
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
        "--dataset_id", type=str, default="haining/poem_interpretation_corpus"
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
