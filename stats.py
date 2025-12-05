#!/usr/bin/env python3
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# ------------------------
# helpers
# ------------------------


def is_categorical_dtype(dtype) -> bool:
    # treat object, boolean, and pandas categorical as categorical-like
    return (
        pd.api.types.is_object_dtype(dtype)
        or pd.api.types.is_bool_dtype(dtype)
        or isinstance(dtype, pd.CategoricalDtype)
    )


def cat_cols(df: pd.DataFrame, exclude=None):
    exclude = set(exclude or [])
    return [
        c for c in df.columns if c not in exclude and is_categorical_dtype(df[c].dtype)
    ]


def num_cols(df: pd.DataFrame, exclude=None):
    exclude = set(exclude or [])
    return [
        c
        for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c].dtype)
    ]


def safe_value_counts(s: pd.Series, topn: int):
    # handle unhashable objects by stringifying
    try:
        vc = s.value_counts(dropna=False)
    except TypeError:
        vc = s.astype(str).value_counts(dropna=False)
    return vc.head(topn)


# ------------------------
# coverage
# ------------------------


def coverage_overall(df: pd.DataFrame, exclude=None) -> pd.DataFrame:
    exclude = set(exclude or [])
    cols = [c for c in df.columns if c not in exclude]
    cov = df[cols].notna().mean(numeric_only=False)
    out = cov.reset_index()
    out.columns = ["column", "coverage"]
    out["coverage"] = out["coverage"].astype(float)
    return out.sort_values("column").reset_index(drop=True)


def coverage_by_split(
    df: pd.DataFrame, split_col="split", exclude=None
) -> pd.DataFrame:
    exclude = set(exclude or [])
    # build a 0/1 mask for non-nulls on attribute columns only
    attrs = [c for c in df.columns if c not in exclude | {split_col}]
    mask = df[attrs].notna().astype(float)
    # attach split for grouping; do not use apply to avoid the warning and duplicate columns
    mask[split_col] = df[split_col].values
    cov = mask.groupby(split_col, dropna=False).mean(numeric_only=False).reset_index()
    # wide -> long
    long = cov.melt(id_vars=[split_col], var_name="column", value_name="coverage")
    long["coverage"] = long["coverage"].astype(float)
    return long.sort_values([split_col, "column"]).reset_index(drop=True)


# ------------------------
# top values
# ------------------------


def top_values_overall(df: pd.DataFrame, topn=25, exclude=None) -> pd.DataFrame:
    exclude = set(exclude or [])
    cols = cat_cols(df, exclude=exclude)
    rows = []
    for c in cols:
        vc = safe_value_counts(df[c], topn)
        total = len(df)
        for val, cnt in vc.items():
            rows.append(
                {
                    "column": c,
                    "value": val,
                    "count": int(cnt),
                    "freq": float(cnt) / float(total) if total else np.nan,
                }
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["column", "count"], ascending=[True, False])
        .reset_index(drop=True)
    )


def top_values_by_split(
    df: pd.DataFrame, split_col="split", topn=25, exclude=None
) -> pd.DataFrame:
    exclude = set(exclude or [])
    cols = cat_cols(df, exclude=exclude | {split_col})
    rows = []
    for c in cols:
        for split_val, g in df.groupby(split_col, dropna=False, sort=False):
            vc = safe_value_counts(g[c], topn)
            total = len(g)
            for val, cnt in vc.items():
                rows.append(
                    {
                        split_col: split_val,
                        "column": c,
                        "value": val,
                        "count": int(cnt),
                        "freq": float(cnt) / float(total) if total else np.nan,
                    }
                )
    return (
        pd.DataFrame(rows)
        .sort_values([split_col, "column", "count"], ascending=[True, True, False])
        .reset_index(drop=True)
    )


# ------------------------
# numeric summaries
# ------------------------


def numeric_summary_overall(df: pd.DataFrame, exclude=None) -> pd.DataFrame:
    exclude = set(exclude or [])
    cols = num_cols(df, exclude=exclude)
    if not cols:
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
    desc = (
        df[cols]
        .agg(
            [
                "count",
                "mean",
                "std",
                "min",
                lambda x: x.quantile(0.05),
                "median",
                lambda x: x.quantile(0.95),
                "max",
            ]
        )
        .T.reset_index()
    )
    desc.columns = [
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
    return desc


def numeric_summary_by_split(
    df: pd.DataFrame, split_col="split", exclude=None
) -> pd.DataFrame:
    exclude = set(exclude or [])
    cols = num_cols(df, exclude=exclude | {split_col})
    if not cols:
        return pd.DataFrame(
            columns=[
                split_col,
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
    g = df.groupby(split_col, dropna=False)
    pieces = []
    for sv, sub in g:
        stats = numeric_summary_overall(sub, exclude=None)
        stats.insert(0, split_col, sv)
        pieces.append(stats)
    return (
        pd.concat(pieces, ignore_index=True)
        if pieces
        else pd.DataFrame(
            columns=[
                split_col,
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
    )


# ------------------------
# main
# ------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_path", required=True)
    ap.add_argument("--out_dir", default="stats_out")
    ap.add_argument("--topn", type=int, default=25)
    ap.add_argument("--by_split", action="store_true")
    ap.add_argument("--split_col", default="split")
    ap.add_argument(
        "--id_cols",
        default="poem_id",
        help="comma-separated id columns to exclude from stats",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_parquet(args.merged_path)

    # normalize column names if needed (no rename by default; keep original)
    id_cols = [c.strip() for c in args.id_cols.split(",") if c.strip()]
    exclude = set(id_cols)

    # coverage
    cov_all = coverage_overall(df, exclude=exclude)
    cov_all.to_csv(out_dir / "coverage_overall.csv", index=False)

    if args.by_split and args.split_col in df.columns:
        cov_split = coverage_by_split(df, split_col=args.split_col, exclude=exclude)
        cov_split.to_csv(out_dir / "coverage_by_split.csv", index=False)

    # top values (categorical-like)
    top_all = top_values_overall(df, topn=args.topn, exclude=exclude | {args.split_col})
    top_all.to_csv(out_dir / "top_values_overall.csv", index=False)

    if args.by_split and args.split_col in df.columns:
        top_split = top_values_by_split(
            df, split_col=args.split_col, topn=args.topn, exclude=exclude
        )
        top_split.to_csv(out_dir / "top_values_by_split.csv", index=False)

    # numeric summaries
    num_all = numeric_summary_overall(df, exclude=exclude | {args.split_col})
    num_all.to_csv(out_dir / "numeric_summary_overall.csv", index=False)

    if args.by_split and args.split_col in df.columns:
        num_split = numeric_summary_by_split(
            df, split_col=args.split_col, exclude=exclude
        )
        num_split.to_csv(out_dir / "numeric_summary_by_split.csv", index=False)

    print(f"wrote outputs to {out_dir}")


if __name__ == "__main__":
    main()
