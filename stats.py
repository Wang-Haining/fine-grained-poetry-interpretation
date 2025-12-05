#!/usr/bin/env python3
# stats.py
# minimal deps: pandas>=2.0, pyarrow or fastparquet

import argparse
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd


def to_lower_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).lower() for c in df.columns]
    return df


def _maybe_parse_json_like(x):
    if isinstance(x, (dict, list)):
        return x
    if isinstance(x, str):
        s = x.strip()
        if (s.startswith("{") and s.endswith("}")) or (
            s.startswith("[") and s.endswith("]")
        ):
            try:
                return json.loads(s)
            except Exception:
                return None
    return None


def expand_jsonish_columns(
    df: pd.DataFrame, candidates: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """
    try to flatten simple json-like columns.
    - dict -> new columns with prefix {col}__{key}
    - list -> new numeric column {col}__len
    only adds numeric-ish fields. leaves original column in place.
    """
    if candidates is None:
        candidates = [c for c in df.columns if df[c].dtype == "object"]

    for c in candidates:
        s = df[c]
        if s.isna().all():
            continue
        # peek a few non-null samples
        sample = s.dropna().head(50)
        parsed = sample.map(_maybe_parse_json_like)
        if parsed.notna().mean() < 0.6:
            continue  # not really json-like
        # parse whole column where possible
        full = s.map(_maybe_parse_json_like)
        # dict case
        dict_mask = full.map(lambda v: isinstance(v, dict))
        if dict_mask.any():
            keys = set()
            for d in full[dict_mask].head(200):
                keys.update(
                    [
                        k
                        for k, v in d.items()
                        if isinstance(v, (int, float))
                        or str(v).strip().replace(".", "", 1).lstrip("-").isdigit()
                    ]
                )
            for k in sorted(keys):
                newc = f"{c}__{k}".lower()
                df[newc] = full.map(
                    lambda d: (d.get(k) if isinstance(d, dict) else np.nan)
                )
                # coerce to numeric where possible
                df[newc] = pd.to_numeric(df[newc], errors="coerce")
        # list case
        list_mask = full.map(lambda v: isinstance(v, list))
        if list_mask.any():
            newc = f"{c}__len".lower()
            df[newc] = full.map(lambda v: (len(v) if isinstance(v, list) else np.nan))
    return df


def derive_text_lengths(df: pd.DataFrame, text_cols: Iterable[str]) -> pd.DataFrame:
    for c in text_cols:
        if c in df.columns:
            s = df[c].astype("string")
            df[f"{c}_len_chars"] = s.str.len()
            df[f"{c}_len_words"] = s.str.split().map(
                lambda xs: len(xs) if isinstance(xs, list) else np.nan
            )
    return df


def detect_and_coerce_numeric(
    df: pd.DataFrame, min_ratio: float = 0.6, exclude: Optional[Iterable[str]] = None
) -> List[str]:
    """
    treat a column as numeric if >= min_ratio of non-null values coerce to numbers
    """
    exclude = set([c for c in (exclude or [])])
    numeric_cols: List[str] = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            numeric_cols.append(c)
            continue
        if df[c].dtype == "object":
            ser = pd.to_numeric(df[c], errors="coerce")
            if ser.notna().mean() >= min_ratio:
                df[c] = ser
                numeric_cols.append(c)
    # drop junk columns that look numeric but are all nan
    numeric_cols = [c for c in numeric_cols if df[c].notna().any()]
    return numeric_cols


def summarize_numeric(df: pd.DataFrame, numeric_cols: List[str]) -> pd.DataFrame:
    if not numeric_cols:
        return pd.DataFrame(
            columns=[
                "column",
                "count",
                "mean",
                "std",
                "min",
                "p5",
                "median",
                "p95",
                "max",
            ]
        )
    g = df[numeric_cols]
    out = pd.DataFrame(
        {
            "column": numeric_cols,
            "count": [int(g[c].notna().sum()) for c in numeric_cols],
            "mean": [g[c].mean() for c in numeric_cols],
            "std": [g[c].std() for c in numeric_cols],
            "min": [g[c].min() for c in numeric_cols],
            "p5": [g[c].quantile(0.05) for c in numeric_cols],
            "median": [g[c].median() for c in numeric_cols],
            "p95": [g[c].quantile(0.95) for c in numeric_cols],
            "max": [g[c].max() for c in numeric_cols],
        }
    )
    return out


def coverage_overall(df: pd.DataFrame) -> pd.DataFrame:
    nn = df.notna().mean(numeric_only=False) * 100.0
    out = nn.reset_index()
    out.columns = ["column", "pct_non_null"]
    out["pct_non_null"] = out["pct_non_null"].round(2)
    return out.sort_values("pct_non_null", ascending=False, kind="mergesort")


def coverage_by_split(df: pd.DataFrame, split_col: str = "split") -> pd.DataFrame:
    if split_col not in df.columns:
        return pd.DataFrame(columns=[split_col, "column", "pct_non_null"])
    records = []
    for split_value, g in df.groupby(split_col, dropna=False):
        nn = g.notna().mean(numeric_only=False) * 100.0
        rec = nn.to_frame(name="pct_non_null").reset_index(names="column")
        rec[split_col] = split_value
        records.append(rec)
    out = pd.concat(records, ignore_index=True)
    out["pct_non_null"] = out["pct_non_null"].round(2)
    return out[[split_col, "column", "pct_non_null"]].sort_values(
        [split_col, "pct_non_null"], ascending=[True, False], kind="mergesort"
    )


def missingness_table(df: pd.DataFrame) -> pd.DataFrame:
    non_null = df.notna().sum()
    total = len(df)
    out = pd.DataFrame(
        {
            "column": non_null.index,
            "dtype": [str(df[c].dtype) for c in df.columns],
            "non_null": non_null.values,
            "null": (total - non_null.values),
            "pct_non_null": (non_null.values / total * 100).round(2),
        }
    )
    return out.sort_values("pct_non_null", ascending=False, kind="mergesort")


def top_values_overall(
    df: pd.DataFrame, exclude: Iterable[str], topn: int
) -> pd.DataFrame:
    out_rows = []
    for c in df.columns:
        if c in exclude:
            continue
        if pd.api.types.is_numeric_dtype(df[c]):
            continue
        vc = df[c].value_counts(dropna=False).head(topn)
        for v, cnt in vc.items():
            out_rows.append({"column": c, "value": v, "count": int(cnt)})
    return pd.DataFrame(out_rows)


def top_values_by_split(
    df: pd.DataFrame, split_col: str, exclude: Iterable[str], topn: int
) -> pd.DataFrame:
    if split_col not in df.columns:
        return pd.DataFrame(columns=[split_col, "column", "value", "count"])
    out_rows = []
    for split_val, g in df.groupby(split_col, dropna=False):
        for c in g.columns:
            if c in exclude or pd.api.types.is_numeric_dtype(g[c]):
                continue
            vc = g[c].value_counts(dropna=False).head(topn)
            for v, cnt in vc.items():
                out_rows.append(
                    {split_col: split_val, "column": c, "value": v, "count": int(cnt)}
                )
    return pd.DataFrame(out_rows)


def ensure_out_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--merged_path", required=True, help="path to merged parquet")
    parser.add_argument("--out_dir", required=True, help="directory for csv outputs")
    parser.add_argument("--topn", type=int, default=25)
    parser.add_argument("--by_split", action="store_true")
    args = parser.parse_args()

    df = pd.read_parquet(args.merged_path)
    df = to_lower_columns(df)

    # drop accidental duplicate column labels
    df = df.loc[:, ~df.columns.duplicated(keep="last")]

    # move index back if it leaked as a named index
    if df.index.name and df.index.name not in df.columns:
        df = df.reset_index()

    # try to flatten simple json-like columns before computing stats
    df = expand_jsonish_columns(df)

    # derive robust numeric features even if attrs are missing
    text_candidates = [
        c for c in ["poem", "interpretation", "title"] if c in df.columns
    ]
    df = derive_text_lengths(df, text_candidates)

    # detect numeric columns with coercion, ignore id-ish columns
    exclude = {
        "poem_id",
        "split",
        "author",
        "title",
        "poem",
        "interpretation",
        "source",
        "index",
    }
    numeric_cols = detect_and_coerce_numeric(df, min_ratio=0.6, exclude=exclude)

    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)

    # missingness and coverage
    missing = missingness_table(df)
    missing.to_csv(out_dir / "missingness.csv", index=False)

    cov_all = coverage_overall(df)
    cov_all.to_csv(out_dir / "coverage_overall.csv", index=False)

    if args.by_split and "split" in df.columns:
        cov_split = coverage_by_split(df, split_col="split")
        cov_split.to_csv(out_dir / "coverage_by_split.csv", index=False)

    # top values
    tv_overall = top_values_overall(
        df, exclude=exclude.union(numeric_cols), topn=args.topn
    )
    tv_overall.to_csv(out_dir / "top_values_overall.csv", index=False)
    if args.by_split and "split" in df.columns:
        tv_split = top_values_by_split(
            df, "split", exclude=exclude.union(numeric_cols), topn=args.topn
        )
        tv_split.to_csv(out_dir / "top_values_by_split.csv", index=False)

    # numeric summaries
    num_all = summarize_numeric(df, numeric_cols)
    num_all.to_csv(out_dir / "numeric_summary_overall.csv", index=False)

    if args.by_split and "split" in df.columns and numeric_cols:
        parts = []
        for s, g in df.groupby("split", dropna=False):
            summ = summarize_numeric(g, numeric_cols)
            summ.insert(0, "split", s)
            parts.append(summ)
        if parts:
            pd.concat(parts, ignore_index=True).to_csv(
                out_dir / "numeric_summary_by_split.csv", index=False
            )

    # tiny console hints
    print(f"rows: {len(df)}")
    print(f"numeric columns detected: {len(numeric_cols)}")
    if not numeric_cols:
        print(
            "no numeric columns found after coercion, only derived lengths will appear if present"
        )


if __name__ == "__main__":
    main()
