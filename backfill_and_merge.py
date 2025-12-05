#!/usr/bin/env python3
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

# ------------------ helpers ------------------


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

    # flatten multiindex columns if present
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["__".join(map(str, t)).strip() for t in df.columns.to_list()]

    # ensure plain string labels then snake_case
    df.columns = [_snake(str(c).strip()) for c in df.columns]

    # if poem_id is in the index, bring it back as a column
    idx_names = [n for n in getattr(df.index, "names", []) if n is not None]
    if df.index.name == "poem_id" or ("poem_id" in idx_names):
        df = df.reset_index()

    # coalesce duplicate poem_id labels
    pid_cols = [c for c in df.columns if c == "poem_id"]
    if len(pid_cols) > 1:
        base = df[pid_cols].bfill(axis=1).iloc[:, 0]
        keep = pid_cols[-1]
        drop = [c for c in pid_cols if c != keep]
        df = df.drop(columns=drop)
        df[keep] = base

    # drop duplicate column labels (last wins)
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="last")]

    # final sanity: exactly one poem_id column
    if (pd.Index(df.columns) == "poem_id").sum() != 1:
        raise AssertionError("poem_id label still ambiguous")

    # ensure id is normalized string
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

    # snake all column names now (handles camelCase, spaces, etc.)
    df.columns = [_snake(c) for c in df.columns]
    # ensure any id-like columns are strings now (before any other ops)
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
    """
    Convert list/dict/ndarray cells to JSON strings so Parquet preserves them.
    Leave scalars (numbers/strings) as-is.
    """
    df = df.copy()
    cols = list(candidates) if candidates is not None else list(df.columns)
    for c in cols:
        if c not in df.columns:
            continue
        if pd.api.types.is_object_dtype(df[c]):

            def _fix(v):
                if isinstance(v, (list, dict, np.ndarray)):
                    try:
                        return json.dumps(v, ensure_ascii=False)
                    except Exception:
                        return str(v)
                return v

            df[c] = df[c].map(_fix)
    return df


# ------------------ main ------------------


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

    # ---------- build catalog from HF dataset ----------
    ds_dict = load_dataset(args.dataset_id)
    rows = []
    for split, ds in ds_dict.items():
        for r in ds:
            author = (r.get("author") or "").strip()
            title = (r.get("title") or "").strip()
            poem = r.get("poem") or ""
            pid = poem_id(author, title, poem)
            rows.append(
                {
                    "split": split,
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
    # normalize/force poem_id string/lower
    catalog = normalize_for_poem_id_merge(catalog)

    # ---------- read attrs ----------
    attrs = read_attrs(Path(args.attrs_path))

    # prefer explicit attrs id if provided (e.g., poem_id_file)
    have_pid = "poem_id" in attrs.columns
    if args.attrs_id_col in attrs.columns and args.attrs_id_col != "poem_id":
        attrs = attrs.rename(columns={args.attrs_id_col: "poem_id"})
        have_pid = True

    # fallback: derive poem_id from author/title/poem if missing and available
    if not have_pid and all(c in attrs.columns for c in ("author", "title", "poem")):
        attrs["poem_id"] = [
            poem_id(a or "", t or "", p or "")
            for a, t, p in zip(attrs["author"], attrs["title"], attrs["poem"])
        ]
        have_pid = True

    # normalize ids on attrs if we have them
    attrs = normalize_for_poem_id_merge(attrs) if have_pid else attrs
    attrs = add_key_columns(attrs, "title", "author")

    # stringify nested in candidate columns so Parquet stores them nicely
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
    }
    idish = {"poem_id", "poem_id_json", "poem_id_file"}

    candidate = [c for c in attrs.columns if c not in meta_drop]
    # do not stringify id columns
    candidate_no_ids = [c for c in candidate if c not in idish]
    attrs = stringify_nested_objects(attrs, candidates=candidate_no_ids)

    # actual value columns = non-id, non-meta that have at least one non-null
    value_cols = [c for c in candidate_no_ids if attrs[c].notna().any()]

    # ---------- sanity: show id intersection ----------
    inter = (
        len(pd.Index(catalog["poem_id"]).intersection(pd.Index(attrs["poem_id"])))
        if have_pid
        else 0
    )
    if have_pid:
        print(f"id intersection with catalog: {inter}/{len(catalog)}")

    # ---------- primary: id-based merge ----------
    merged = catalog.copy()
    matched_by_id = pd.Series(False, index=merged.index)
    if have_pid and value_cols:
        right = attrs[["poem_id"] + value_cols].drop_duplicates("poem_id")
        # force id strings (defensive)
        right = _force_id_str(right, ["poem_id"])
        merged = _force_id_str(merged, ["poem_id"]).merge(
            right, on="poem_id", how="left"
        )
        matched_by_id = merged[value_cols].notna().any(axis=1)

    # ---------- fallback: exact author_key + title_key ----------
    need_fallback = ~matched_by_id
    if (
        need_fallback.any()
        and {"author_key", "title_key"}.issubset(attrs.columns)
        and value_cols
    ):
        left_sub = merged.loc[need_fallback, ["poem_id", "author_key", "title_key"]]
        right_sub = attrs[["author_key", "title_key"] + value_cols].drop_duplicates()
        glued = left_sub.merge(
            right_sub, on=["author_key", "title_key"], how="left"
        ).set_index("poem_id")
        for c in value_cols:
            if c not in merged.columns:
                merged[c] = pd.NA
            merged.loc[need_fallback, c] = merged.loc[need_fallback, c].combine_first(
                merged.loc[need_fallback, "poem_id"].map(glued[c])
            )

    # ---------- coverage report (value columns only) ----------
    has_any_value = (
        merged[value_cols].notna().any(axis=1)
        if value_cols
        else pd.Series(False, index=merged.index)
    )
    print(f"total rows: {len(merged)}")
    print(f"with attrs: {int(has_any_value.sum())}")
    print(f"coverage   : {has_any_value.mean():.2%}")
    if have_pid and value_cols:
        print(f"  matched                  by_id: +{int(matched_by_id.sum())}")
    if need_fallback.any() and value_cols:
        print(
            f"  filled by author+title fallback: +{int((has_any_value & ~matched_by_id).sum())}"
        )

    # ---------- write parquet ----------
    merged.to_parquet(args.merged_out, index=False)

    # ---------- optional per-row attribute jsons ----------
    if args.materialize_json and value_cols:
        for split, g in merged.groupby("split", sort=False):
            base = Path(args.out_dir) / split / "rows"
            base.mkdir(parents=True, exist_ok=True)
            att_only = g[["poem_id"] + value_cols]
            for r in att_only.to_dict(orient="records"):
                pid = r.pop("poem_id")
                if all(pd.isna(v) for v in r.values()):
                    continue
                (base / f"{pid}.json").write_text(json.dumps(r, ensure_ascii=False))
        print("materialized json rows under:", args.out_dir)


if __name__ == "__main__":
    main()
