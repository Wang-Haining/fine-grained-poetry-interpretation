"""
merge json rows from new annotator into the hf dataset.
layout expected: <attrs_dir>/<split>/rows/*.json
joins on 'row_index'. writes <out_dir>/<split>.parquet.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
from datasets import load_dataset

TARGET_COLS = ["emotions", "primary_emotion", "sentiment", "themes", "themes_50"]


def coerce_int(x: Any) -> Optional[int]:
    try:
        if isinstance(x, bool):
            return None
        if isinstance(x, int):
            return x
        if isinstance(x, float) and x.is_integer():
            return int(x)
        if isinstance(x, str):
            s = x.strip()
            if s.startswith("+"):
                s = s[1:]
            if s.isdigit():
                return int(s)
    except Exception:
        pass
    return None


def load_row_payloads(rows_dir: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    if not rows_dir.exists():
        return out
    for p in rows_dir.glob("*.json"):
        try:
            d = json.loads(p.read_text(encoding="utf-8"))
        except Exception:
            continue
        idx = coerce_int(d.get("row_index"))
        if idx is None:
            idx = coerce_int(p.stem)
        if idx is None:
            continue
        out[int(idx)] = d
    return out


def drop_and_recreate_cols(df: pd.DataFrame) -> None:
    for c in TARGET_COLS:
        if c in df.columns:
            df.drop(columns=[c], inplace=True)
        df[c] = pd.Series([pd.NA] * len(df), dtype="object")


def build_rowindex_to_pos(df: pd.DataFrame) -> Dict[int, int]:
    # prefer explicit row_index if present; else fall back to enumerate index
    if "row_index" in df.columns:
        keys = [coerce_int(v) for v in df["row_index"].tolist()]
    else:
        keys = list(range(len(df)))
    out: Dict[int, int] = {}
    for pos, k in enumerate(keys):
        kk = pos if k is None else int(k)
        out[kk] = pos
    return out


def merge_split(ds_id: str, split: str, attrs_root: Path, out_root: Path) -> Path:
    ds = load_dataset(ds_id, split=split)
    df = ds.to_pandas()
    df.columns = [c.lower() for c in df.columns]

    rows_dir = attrs_root / split / "rows"
    payloads = load_row_payloads(rows_dir)

    drop_and_recreate_cols(df)
    idx2pos = build_rowindex_to_pos(df)

    matched = 0
    for ridx, d in payloads.items():
        pos = idx2pos.get(int(ridx))
        if pos is None:
            continue
        df.at[pos, "emotions"] = d.get("emotions")
        df.at[pos, "primary_emotion"] = d.get("primary_emotion")
        df.at[pos, "sentiment"] = d.get("sentiment")
        df.at[pos, "themes"] = d.get("themes")
        df.at[pos, "themes_50"] = d.get("themes_50")
        matched += 1

    out_root.mkdir(parents=True, exist_ok=True)
    out_path = out_root / f"{split}.parquet"
    df.to_parquet(out_path, index=False)

    print(
        f"[merge] split={split} json_rows={len(payloads)} matched={matched} -> {out_path}"
    )
    return out_path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--dataset_id",
        type=str,
        default="haining/structured_poem_interpretation_corpus",
    )
    ap.add_argument(
        "--attrs_dir", type=str, default="poem_attrs"
    )  # expects <split>/rows/*.json
    ap.add_argument(
        "--out_dir", type=str, default="poem_attrs/merged"
    )  # writes <split>.parquet
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
