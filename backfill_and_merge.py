#!/usr/bin/env python3
# backfill_and_merge.py
# merge per-poem attributes into the HF catalog; supports id-join and (split,index) fallback

import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from datasets import load_dataset

# ---------- helpers ----------


def _snake(name: str) -> str:
    if name is None:
        return ""
    s = (
        unicodedata.normalize("NFKD", str(name))
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    s = re.sub(r"[^\w]+", "_", s)
    s = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s)
    return re.sub(r"_+", "_", s).strip("_").lower()


def norm_text(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.lower().replace("&", " and ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()


def poem_id(author: str, title: str, poem: str) -> str:
    key = f"{(author or '').strip()}\n{(title or '').strip()}\n{(poem or '').strip()[:200]}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def _force_id_str(df: pd.DataFrame, cols: Iterable[str]) -> pd.DataFrame:
    df = df.copy()
    for c in cols:
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().str.lower()
    return df


def normalize_for_poem_id_merge(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["__".join(map(str, t)).strip() for t in df.columns.to_list()]
    df.columns = [_snake(str(c).strip()) for c in df.columns]
    idx_names = [n for n in getattr(df.index, "names", []) if n is not None]
    if df.index.name == "poem_id" or ("poem_id" in idx_names):
        df = df.reset_index()
    pid_cols = [c for c in df.columns if c == "poem_id"]
    if len(pid_cols) > 1:
        base = df[pid_cols].bfill(axis=1).iloc[:, 0]
        keep = pid_cols[-1]
        drop = [c for c in pid_cols if c != keep]
        df = df.drop(columns=drop)
        df[keep] = base
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="last")]
    if (pd.Index(df.columns) == "poem_id").sum() != 1:
        raise AssertionError("poem_id label still ambiguous")
    df = _force_id_str(df, ["poem_id"])
    return df


def read_attrs(path: Path) -> pd.DataFrame:
    p = str(path)
    if p.endswith(".parquet"):
        df = pd.read_parquet(p)
    elif p.endswith(".csv"):
        df = pd.read_csv(p)
    elif p.endswith(".jsonl") or p.endswith(".ndjson"):
        df = pd.read_json(p, lines=True)
    else:
        raise ValueError(f"unsupported attrs format: {p}")
    df.columns = [_snake(c) for c in df.columns]
    df = _force_id_str(df, ["poem_id", "poem_id_file", "poem_id_json"])
    return df


def add_key_columns(df: pd.DataFrame, title_col: str, author_col: str) -> pd.DataFrame:
    df = df.copy()
    if "title_key" not in df.columns and title_col in df.columns:
        df["title_key"] = df[title_col].map(norm_text)
    if "author_key" not in df.columns and author_col in df.columns:
        df["author_key"] = df[author_col].map(norm_text)
    return df


def stringify_nested_objects(
    df: pd.DataFrame, candidates: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    # convert list/dict/ndarray to json strings; leave scalars as-is
    df = df.copy()
    cols = list(candidates) if candidates is not None else list(df.columns)
    for c in cols:
        if c not in df.columns:
            continue

        def _fix(v):
            if isinstance(v, (list, dict, np.ndarray)):
                try:
                    return json.dumps(v, ensure_ascii=False)
                except Exception:
                    return str(v)
            return v

        if pd.api.types.is_object_dtype(df[c]):
            df[c] = df[c].map(_fix)
    return df


def _emptyish(v) -> bool:
    # robust "is empty" for scalars, strings, lists, arrays
    if v is None:
        return True
    try:
        if pd.isna(v):
            return True
    except Exception:
        pass
    if isinstance(v, str):
        s = v.strip().lower()
        return s in {"", "nan", "none", "null"}
    if isinstance(v, (list, tuple, dict)):
        return len(v) == 0
    if isinstance(v, np.ndarray):
        return v.size == 0
    return False


HEX16 = re.compile(r"^[0-9a-f]{16}$")


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", default="haining/poem_interpretation_corpus")
    ap.add_argument("--attrs_path", required=True)
    ap.add_argument("--attrs_id_col", default="poem_id")
    ap.add_argument("--out_dir", default="poem_attrs")
    ap.add_argument("--merged_out", default="dist_v1/merged.parquet")
    ap.add_argument("--materialize_json", action="store_true")
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(args.merged_out).parent).mkdir(parents=True, exist_ok=True)

    # build catalog with deterministic poem_id + per-split row_index
    ds_dict = load_dataset(args.dataset_id)
    rows = []
    for split, ds in ds_dict.items():
        for i, r in enumerate(ds):
            author = (r.get("author") or "").strip()
            title = (r.get("title") or "").strip()
            poem = r.get("poem") or ""
            pid = poem_id(author, title, poem)
            rows.append(
                {
                    "split": split,
                    "row_index": i,  # within-split index
                    "poem_id": pid,
                    "author": author,
                    "title": title,
                    "poem": poem,
                    "interpretation": r.get("interpretation", ""),
                    "source": r.get("source", ""),
                }
            )
    catalog = pd.DataFrame(rows)
    catalog.columns = [_snake(c) for c in catalog.columns]
    catalog = add_key_columns(catalog, "title", "author")
    catalog = normalize_for_poem_id_merge(catalog)
    catalog["row_index"] = pd.array(catalog["row_index"], dtype="Int64")

    # read attrs
    attrs = read_attrs(Path(args.attrs_path))

    # honor provided id column
    have_pid = "poem_id" in attrs.columns
    if args.attrs_id_col in attrs.columns and args.attrs_id_col != "poem_id":
        attrs = attrs.rename(columns={args.attrs_id_col: "poem_id"})
        have_pid = True

    # normalize ids and keys
    if have_pid:
        attrs = normalize_for_poem_id_merge(attrs)
    attrs = add_key_columns(attrs, "title", "author")

    # coerce index to numeric for fallback
    if "index" in attrs.columns:
        attrs["index"] = pd.to_numeric(attrs["index"], errors="coerce")
        attrs["index"] = pd.array(attrs["index"], dtype="Int64")
    has_split = "split" in attrs.columns

    # choose value columns to import (drop meta/id-ish)
    meta_drop = {
        "split",
        "poem",
        "interpretation",
        "title",
        "author",
        "source",
        "title_src",
        "author_src",
        "poem_src",
        "title_key",
        "author_key",
        "row_index",
        "index",
        "poem_id_json",
        "poem_id_file",
        "poem_id",
    }
    candidate = [c for c in attrs.columns if c not in meta_drop]
    # keep columns that have any non-null data
    value_cols = [c for c in candidate if attrs[c].notna().any()]

    merged = catalog.copy()

    # ---------- pass 1: direct id join on 16-hex poem_id ----------
    matched_by_id = pd.Series(False, index=merged.index)
    if have_pid and value_cols:
        attrs_hex = attrs[attrs["poem_id"].str.fullmatch(HEX16.pattern, na=False)]
        if not attrs_hex.empty:
            right = attrs_hex[["poem_id"] + value_cols].drop_duplicates("poem_id")
            merged = merged.merge(right, on="poem_id", how="left")
            matched_by_id = merged[value_cols].notna().any(axis=1)

        inter = (
            len(
                pd.Index(catalog["poem_id"]).intersection(
                    pd.Index(attrs_hex["poem_id"])
                )
            )
            if not attrs_hex.empty
            else 0
        )
        print(f"id (hex) intersection with catalog: {inter}/{len(catalog)}")

    # ensure value columns exist post-merge (if pass 1 had no matches)
    for c in value_cols:
        if c not in merged.columns:
            merged[c] = pd.NA

    # ---------- pass 2: (split, index) → (split, row_index) fallback ----------
    need_fallback = ~matched_by_id
    did_fallback = pd.Series(False, index=merged.index)
    if has_split and ("index" in attrs.columns) and value_cols:
        left_sub = merged.loc[need_fallback, ["poem_id", "split", "row_index"]].copy()
        left_sub["row_index"] = pd.array(left_sub["row_index"], dtype="Int64")

        right_sub = (
            attrs[["split", "index"] + value_cols]
            .dropna(subset=["index"])
            .drop_duplicates(subset=["split", "index"])
        )
        if not right_sub.empty:
            glued = left_sub.merge(
                right_sub,
                left_on=["split", "row_index"],
                right_on=["split", "index"],
                how="left",
            ).set_index("poem_id")
            for c in value_cols:
                prev_na = merged.loc[need_fallback, c].isna()
                fill = merged.loc[need_fallback, "poem_id"].map(glued[c])
                merged.loc[need_fallback, c] = merged.loc[
                    need_fallback, c
                ].combine_first(fill)
                did_fallback.loc[need_fallback] |= (
                    prev_na & merged.loc[need_fallback, c].notna()
                )

    # coverage report
    have_any = (
        merged[value_cols].notna().any(axis=1)
        if value_cols
        else pd.Series(False, index=merged.index)
    )
    print(f"total rows: {len(merged)}")
    print(f"with attrs: {int(have_any.sum())}")
    print(f"coverage   : {have_any.mean():.2%}")
    if value_cols:
        print(f"  matched                  by_id: +{int(matched_by_id.sum())}")
        print(f"  filled by (split,index) fallback: +{int(did_fallback.sum())}")

    # small peek
    try:
        cols_show = ["poem_id", "split", "row_index"] + value_cols[:6]
        print("\npeek with attrs:")
        print(merged.loc[have_any, cols_show].head(5).to_string(index=False))
    except Exception:
        pass

    # write parquet
    merged.to_parquet(args.merged_out, index=False)

    # optional per-row attribute jsons
    if args.materialize_json and value_cols:
        for split, g in merged.groupby("split", sort=False):
            base = Path(args.out_dir) / split / "rows"
            base.mkdir(parents=True, exist_ok=True)
            att_only = g[["poem_id"] + value_cols]
            for r in att_only.to_dict(orient="records"):
                pid = r.pop("poem_id")
                if all(_emptyish(v) for v in r.values()):
                    continue
                (base / f"{pid}.json").write_text(json.dumps(r, ensure_ascii=False))
        print("materialized json rows under:", args.out_dir)


if __name__ == "__main__":
    main()
