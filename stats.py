# stats for merged annotations
# usage:
#   python stats.py --merged_path dist_v1/merged.parquet --out_dir stats_out --topn 25 --by_split
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument(
    "--merged_path",
    required=True,
    help="path to merged parquet (e.g., dist_v1/merged.parquet)",
)
parser.add_argument(
    "--out_dir", default="stats_out", help="directory to write summaries"
)
parser.add_argument(
    "--topn",
    type=int,
    default=20,
    help="top-n values to keep for categorical summaries",
)
parser.add_argument(
    "--by_split",
    action="store_true",
    help="also compute per-split summaries when split column exists",
)
args = parser.parse_args()

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)

# load and normalize
df = pd.read_parquet(args.merged_path)
df.columns = [str(c).strip().lower() for c in df.columns]  # lower case columns
has_split = "split" in df.columns

# quick health checks
total_rows = len(df)
n_unique_poem = df["poem_id"].nunique() if "poem_id" in df.columns else None
dup_poem_ids = int(total_rows - (n_unique_poem or 0))

# split counts
split_counts = {}
if has_split:
    sc = (
        df["split"]
        .value_counts(dropna=False)
        .rename_axis("split")
        .reset_index(name="count")
    )
    sc["pct"] = (sc["count"] / total_rows * 100).round(2)
    split_counts = {
        r["split"]: {"count": int(r["count"]), "pct": float(r["pct"])}
        for _, r in sc.iterrows()
    }
    sc.to_csv(out_dir / "split_counts.csv", index=False)

# dtype helpers
is_numeric = {c: pd.api.types.is_numeric_dtype(df[c]) for c in df.columns}
is_stringy = {
    c: (
        pd.api.types.is_string_dtype(df[c])
        or pd.api.types.is_object_dtype(df[c])
        or pd.api.types.is_categorical_dtype(df[c])
    )
    for c in df.columns
}

# missingness summary
miss_rows = []
for c in df.columns:
    non_null = int(df[c].notna().sum())
    null = int(total_rows - non_null)
    pct_non_null = round(non_null / total_rows * 100, 2) if total_rows else 0.0
    miss_rows.append(
        {
            "column": c,
            "dtype": str(df[c].dtype),
            "non_null": non_null,
            "null": null,
            "pct_non_null": pct_non_null,
        }
    )
miss_df = pd.DataFrame(miss_rows)
miss_df.to_csv(out_dir / "missingness.csv", index=False)

# coverage by split (for every column)
if has_split:
    cov_split = (
        df.assign(_nn=1 - df.isna())
        .groupby("split", dropna=False)
        .apply(lambda g: (1 - g.isna()).mean(numeric_only=False))
    )
    # drop non-column rows introduced by apply nuances
    if isinstance(cov_split.index, pd.MultiIndex):
        cov_split.index = cov_split.index.get_level_values(0)
    cov_split = cov_split.drop(columns=["split"], errors="ignore")
    cov_split = (cov_split * 100).round(2)
    cov_split.reset_index().to_csv(out_dir / "coverage_by_split.csv", index=False)

# numeric summary
num_cols = [c for c, flag in is_numeric.items() if flag and c != "poem_id"]
if num_cols:
    desc = df[num_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
    desc = desc.rename(
        columns={
            "count": "count",
            "mean": "mean",
            "std": "std",
            "min": "min",
            "5%": "p05",
            "25%": "p25",
            "50%": "p50",
            "75%": "p75",
            "95%": "p95",
            "max": "max",
        }
    )
    desc = desc.round(4).reset_index().rename(columns={"index": "column"})
    desc.to_csv(out_dir / "numeric_summary.csv", index=False)

    if has_split and args.by_split:
        # per-split numeric describe
        parts = []
        for s, g in df.groupby("split", dropna=False):
            sub = g[num_cols].describe(percentiles=[0.05, 0.25, 0.5, 0.75, 0.95]).T
            sub = sub.rename(
                columns={
                    "5%": "p05",
                    "25%": "p25",
                    "50%": "p50",
                    "75%": "p75",
                    "95%": "p95",
                }
            )
            sub = sub.round(4).reset_index().rename(columns={"index": "column"})
            sub.insert(0, "split", s)
            parts.append(sub)
        pd.concat(parts, ignore_index=True).to_csv(
            out_dir / "numeric_summary_by_split.csv", index=False
        )

# categorical top-k (strings and categoricals)
cat_cols = [c for c, flag in is_stringy.items() if flag and c not in {"poem_id"}]
cat_rows = []
for c in cat_cols:
    vc = df[c].value_counts(dropna=False).head(args.topn)
    total = int(df[c].shape[0])
    for val, cnt in vc.items():
        pct = round(cnt / total * 100, 2) if total else 0.0
        cat_rows.append(
            {
                "column": c,
                "value": ("" if pd.isna(val) else str(val))[:500],
                "count": int(cnt),
                "pct": float(pct),
            }
        )
if cat_rows:
    pd.DataFrame(cat_rows).to_csv(out_dir / "categorical_top.csv", index=False)

# optional: per-split top-k for categoricals
if has_split and args.by_split and cat_cols:
    parts = []
    for s, g in df.groupby("split", dropna=False):
        for c in cat_cols:
            vc = g[c].value_counts(dropna=False).head(args.topn)
            total = int(g[c].shape[0])
            for val, cnt in vc.items():
                pct = round(cnt / total * 100, 2) if total else 0.0
                parts.append(
                    {
                        "split": s,
                        "column": c,
                        "value": ("" if pd.isna(val) else str(val))[:500],
                        "count": int(cnt),
                        "pct": float(pct),
                    }
                )
    if parts:
        pd.DataFrame(parts).to_csv(
            out_dir / "categorical_top_by_split.csv", index=False
        )

# lightweight json fields inventory (counts of dict/list objects)
json_like = {}
sample_n = min(2000, total_rows)
sample = df.sample(n=sample_n, random_state=0) if total_rows > sample_n else df
for c in df.columns:
    try:
        n_jsonish = int(sample[c].apply(lambda x: isinstance(x, (dict, list))).sum())
        if n_jsonish > 0:
            json_like[c] = {
                "jsonish_in_sample": n_jsonish,
                "sample_size": int(len(sample)),
                "est_pct_in_sample": round(n_jsonish / len(sample) * 100, 2),
            }
    except Exception:
        pass
(out_dir / "json_fields.json").write_text(json.dumps(json_like, indent=2))

# write a compact run summary
summary = {
    "merged_path": str(args.merged_path),
    "out_dir": str(out_dir),
    "total_rows": int(total_rows),
    "unique_poem_id": int(n_unique_poem) if n_unique_poem is not None else None,
    "duplicate_poem_id_rows": int(dup_poem_ids) if n_unique_poem is not None else None,
    "has_split": bool(has_split),
    "split_counts": split_counts,
    "n_numeric_cols": int(len(num_cols)),
    "n_categorical_cols": int(len(cat_cols)),
    "outputs": [
        "split_counts.csv" if has_split else None,
        "missingness.csv",
        "coverage_by_split.csv" if has_split else None,
        "numeric_summary.csv" if num_cols else None,
        (
            "numeric_summary_by_split.csv"
            if (num_cols and has_split and args.by_split)
            else None
        ),
        "categorical_top.csv" if cat_cols else None,
        (
            "categorical_top_by_split.csv"
            if (cat_cols and has_split and args.by_split)
            else None
        ),
        "json_fields.json",
    ],
}
summary["outputs"] = [p for p in summary["outputs"] if p is not None]
(out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

# console recap
print("rows:", total_rows)
if n_unique_poem is not None:
    print("unique poem_id:", n_unique_poem, "| duplicates:", dup_poem_ids)
if has_split:
    print("splits:", json.dumps(split_counts, indent=2))
print("wrote summaries to:", out_dir.resolve())
