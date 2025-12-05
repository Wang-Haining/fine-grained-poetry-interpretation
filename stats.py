#!/usr/bin/env python3
# stats.py

# compute basic coverage, top values for string-like columns, and numeric summaries
# outputs are written as csv files under --out_dir

import argparse
from pathlib import Path

import pandas as pd
from pandas.api.types import (
    CategoricalDtype,
    is_numeric_dtype,
    is_object_dtype,
    is_string_dtype,
)


def detect_stringy_columns(df: pd.DataFrame) -> list[str]:
    # choose columns that are string/object/categorical; exclude ids and split
    skip = {"poem_id", "split"}
    cols = []
    for c in df.columns:
        if c in skip:
            continue
        dt = df[c].dtype
        if (
            is_string_dtype(dt)
            or is_object_dtype(dt)
            or isinstance(dt, CategoricalDtype)
        ):
            cols.append(c)
    return cols


def detect_numeric_columns(df: pd.DataFrame) -> list[str]:
    # choose numeric columns
    cols = [c for c in df.columns if is_numeric_dtype(df[c])]
    return cols


def ensure_out_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def coverage_overall(df: pd.DataFrame) -> pd.DataFrame:
    # fraction non-null per column
    cov = df.notna().mean(numeric_only=False) * 100.0
    cov = (
        cov.round(2)
        .rename("coverage_pct")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    return cov


def coverage_by_split(df: pd.DataFrame, split_col: str = "split") -> pd.DataFrame:
    # wide table: one row per split, columns are original columns
    cov = (
        df.groupby(split_col, dropna=False)
        .apply(lambda g: g.notna().mean(numeric_only=False))
        .mul(100.0)
        .round(2)
    )
    cov.index.name = split_col
    cov = cov.reset_index()
    return cov


def top_values_overall(df: pd.DataFrame, cols: list[str], topn: int) -> pd.DataFrame:
    # top n value counts for each column
    out = []
    n = len(df)
    for c in cols:
        vc = df[c].dropna()
        # turn unhashable objects (lists/dicts) into strings safely
        vc = vc.astype(str)
        vc = vc.value_counts().head(topn)
        if vc.empty:
            continue
        temp = vc.to_frame("count").reset_index().rename(columns={"index": "value"})
        temp["pct"] = (temp["count"] / n * 100.0).round(2)
        temp.insert(0, "column", c)
        out.append(temp)
    return (
        pd.concat(out, ignore_index=True)
        if out
        else pd.DataFrame(columns=["column", "value", "count", "pct"])
    )


def top_values_by_split(
    df: pd.DataFrame, cols: list[str], topn: int, split_col: str = "split"
) -> pd.DataFrame:
    # top n per split
    out = []
    for split, g in df.groupby(split_col, dropna=False):
        m = len(g)
        for c in cols:
            vc = g[c].dropna().astype(str).value_counts().head(topn)
            if vc.empty:
                continue
            temp = vc.to_frame("count").reset_index().rename(columns={"index": "value"})
            temp["pct"] = (temp["count"] / m * 100.0).round(2)
            temp.insert(0, split_col, split)
            temp.insert(1, "column", c)
            out.append(temp)
    return (
        pd.concat(out, ignore_index=True)
        if out
        else pd.DataFrame(columns=[split_col, "column", "value", "count", "pct"])
    )


def numeric_summary_overall(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(
            columns=[
                "column",
                "count",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ]
        )
    desc = (
        df[cols]
        .describe(percentiles=[0.25, 0.5, 0.75])
        .T.reset_index()
        .rename(columns={"index": "column"})
    )
    # keep a stable column order
    order = ["column", "count", "mean", "std", "min", "25%", "50%", "75%", "max"]
    for col in order:
        if col not in desc.columns:
            desc[col] = pd.NA
    return desc[order]


def numeric_summary_by_split(
    df: pd.DataFrame, cols: list[str], split_col: str = "split"
) -> pd.DataFrame:
    if not cols:
        return pd.DataFrame(
            columns=[
                split_col,
                "column",
                "count",
                "mean",
                "std",
                "min",
                "25%",
                "50%",
                "75%",
                "max",
            ]
        )
    parts = []
    for split, g in df.groupby(split_col, dropna=False):
        d = (
            g[cols]
            .describe(percentiles=[0.25, 0.5, 0.75])
            .T.reset_index()
            .rename(columns={"index": "column"})
        )
        d.insert(0, split_col, split)
        parts.append(d)
    out = pd.concat(parts, ignore_index=True)
    order = [
        split_col,
        "column",
        "count",
        "mean",
        "std",
        "min",
        "25%",
        "50%",
        "75%",
        "max",
    ]
    for col in order:
        if col not in out.columns:
            out[col] = pd.NA
    return out[order]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="compute coverage, top values, and numeric summaries from merged parquet"
    )
    p.add_argument("--merged_path", required=True, help="path to merged parquet file")
    p.add_argument("--out_dir", required=True, help="output directory for stats csvs")
    p.add_argument(
        "--topn", type=int, default=20, help="top n values for categorical columns"
    )
    p.add_argument(
        "--by_split",
        action="store_true",
        help="also compute per-split stats if a split column exists",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    merged_path = Path(args.merged_path)
    out_dir = Path(args.out_dir)
    ensure_out_dir(out_dir)

    df = pd.read_parquet(merged_path)

    has_split = "split" in df.columns
    if args.by_split and not has_split:
        print(
            "warning: --by_split requested but no 'split' column found; computing overall stats only"
        )

    # coverage
    cov_overall = coverage_overall(df)
    cov_overall.to_csv(out_dir / "coverage_overall.csv", index=False)

    if args.by_split and has_split:
        cov_split = coverage_by_split(df, split_col="split")
        cov_split.to_csv(out_dir / "coverage_by_split.csv", index=False)

    # top values for string-like columns
    stringy = detect_stringy_columns(df)
    tv_overall = top_values_overall(df, stringy, args.topn)
    tv_overall.to_csv(out_dir / "top_values_overall.csv", index=False)

    if args.by_split and has_split:
        tv_split = top_values_by_split(df, stringy, args.topn, split_col="split")
        tv_split.to_csv(out_dir / "top_values_by_split.csv", index=False)

    # numeric summaries
    numeric_cols = detect_numeric_columns(df)
    num_overall = numeric_summary_overall(df, numeric_cols)
    num_overall.to_csv(out_dir / "numeric_summary_overall.csv", index=False)

    if args.by_split and has_split:
        num_split = numeric_summary_by_split(df, numeric_cols, split_col="split")
        num_split.to_csv(out_dir / "numeric_summary_by_split.csv", index=False)

    print("done. wrote stats to", out_dir)


if __name__ == "__main__":
    main()
