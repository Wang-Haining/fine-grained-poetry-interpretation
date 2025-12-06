#!/usr/bin/env python3
# stats.py
# minimal deps: pandas>=2.0, pyarrow or fastparquet

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

# ---------- utils ----------


def to_lower_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [str(c).lower() for c in df.columns]
    return df


def _ensure_list_str(v):
    # normalize to list[str]
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return []
    if isinstance(v, list):
        return [str(x).strip().lower() for x in v if str(x).strip() != ""]
    if isinstance(v, tuple):
        return [str(x).strip().lower() for x in v if str(x).strip() != ""]
    if isinstance(v, str):
        s = v.strip()
        if s == "" or s.lower() == "nan":
            return []
        if s.startswith("[") and s.endswith("]"):
            try:
                j = json.loads(s)
                if isinstance(j, list):
                    return [str(x).strip().lower() for x in j if str(x).strip() != ""]
            except Exception:
                pass
        return [s.lower()]
    return [str(v).strip().lower()]


def _word_count(s: Optional[str]) -> int:
    if s is None:
        return 0
    s = str(s)
    if not s.strip():
        return 0
    return len(s.split())


def _unique_authors(series: pd.Series) -> int:
    return (
        series.fillna("")
        .astype(str)
        .str.normalize("NFKD")
        .str.encode("ascii", "ignore")
        .str.decode("ascii")
        .str.strip()
        .str.lower()
        .replace("", np.nan)
        .nunique(dropna=True)
    )


# ---------- table 1: overview by source ----------


def table1_overview_by_source(df: pd.DataFrame) -> pd.DataFrame:
    need_cols = {"source", "author", "interpretation"}
    missing = need_cols - set(df.columns)
    if missing:
        raise ValueError(f"missing required columns for table1: {missing}")

    d = df.copy()
    d["interp_wc"] = d["interpretation"].apply(_word_count)

    records = []
    for src, g in d.groupby("source", dropna=False):
        records.append(
            {
                "dataset": str(src),
                "total_poems": int(len(g)),
                "unique_poets": int(_unique_authors(g["author"])),
                "avg_interpretation_word_count": float(g["interp_wc"].mean()),
            }
        )
    out = pd.DataFrame(records).sort_values("dataset")
    # round for readability
    out["avg_interpretation_word_count"] = out["avg_interpretation_word_count"].round(2)
    return out


# ---------- table 2: literary device frequency (simple lexical scan over interpretations) ----------

# keep patterns minimal and word-bounded; counted on interpretation text only
DEVICE_PATTERNS: List[Tuple[str, str]] = [
    ("imagery", r"\bimagery\b"),
    ("symbolism", r"\bsymbolism\b"),
    ("rhyme", r"\brhyme\b|\brhymes\b|\brhyming\b"),
    ("metaphor", r"\bmetaphor(s)?\b"),
    ("personification", r"\bpersonification\b"),
    ("meter", r"\bmeter\b"),  # american spelling
    ("enjambment", r"\benjambment\b"),
    ("simile", r"\bsimile(s)?\b"),
    ("irony", r"\birony\b|\bironic\b"),
    ("alliteration", r"\balliteration\b"),
    ("hyperbole", r"\bhyperbole\b"),
    ("symbol", r"\bsymbol(s)?\b"),
    ("caesura", r"\bcaesura\b"),
    ("sonnet", r"\bsonnet(s)?\b"),
    ("paradox", r"\bparadox(es)?\b"),
    ("onomatopoeia", r"\bonomatopoeia\b"),
    ("apostrophe", r"\bapostrophe\b"),
    ("anaphora", r"\banaphora\b"),
    ("oxymoron", r"\boxymoron\b"),
    ("assonance", r"\bassonance\b"),
    ("allegory", r"\ballegor(y|ies)\b"),
    ("consonance", r"\bconsonance\b"),
    ("metre", r"\bmetre\b"),  # british spelling
]


def table2_device_freq(df: pd.DataFrame) -> pd.DataFrame:
    if "interpretation" not in df.columns:
        raise ValueError("missing column 'interpretation' for device counts")
    s = df["interpretation"].fillna("").astype(str).str.lower()

    out_rows = []
    for name, pat in DEVICE_PATTERNS:
        cnt = int(s.str.contains(pat, regex=True, na=False).sum())
        out_rows.append({"device": name, "poems_with_term": cnt})
    out = pd.DataFrame(out_rows).sort_values("poems_with_term", ascending=False)
    return out


# ---------- table 3: tag distributions (emotions, themes, sentiment) ----------


def table3_emotions(df: pd.DataFrame) -> pd.DataFrame:
    if "emotions" not in df.columns:
        raise ValueError("missing column 'emotions'")
    n_rows = len(df)
    emo_lists = df["emotions"].apply(_ensure_list_str)

    # per-emotion row presence
    all_labels = sorted({e for xs in emo_lists for e in xs})
    rows = []
    for lab in all_labels:
        present = emo_lists.apply(lambda xs: lab in set(xs)).sum()
        rows.append(
            {
                "emotion": lab,
                "count": int(present),
                "pct_rows_with_emotion": round(present / n_rows * 100.0, 2),
            }
        )
    return pd.DataFrame(rows).sort_values(["count", "emotion"], ascending=[False, True])


def table3_themes_top(df: pd.DataFrame, topn: int = 20) -> pd.DataFrame:
    if "themes" not in df.columns:
        raise ValueError("missing column 'themes'")
    n_rows = len(df)
    lists = df["themes"].apply(_ensure_list_str).apply(lambda xs: sorted(set(xs)))
    exploded = lists.explode().dropna()
    vc = exploded.value_counts()
    vc = vc.head(topn)
    out = pd.DataFrame(
        {
            "theme": vc.index.to_list(),
            "count": vc.values.astype(int),
            "pct_rows_with_theme": (vc.values / n_rows * 100.0).round(2),
        }
    )
    return out


def table3_sentiment(df: pd.DataFrame) -> pd.DataFrame:
    if "sentiment" not in df.columns:
        raise ValueError("missing column 'sentiment'")
    n_rows = len(df)
    s = df["sentiment"].astype(str).str.strip().str.lower().replace({"nan": None})
    vc = s.value_counts(dropna=True)
    out = pd.DataFrame(
        {
            "sentiment": vc.index.to_list(),
            "count": vc.values.astype(int),
            "pct": (vc.values / n_rows * 100.0).round(2),
        }
    ).sort_values("count", ascending=False)
    return out


# ---------- main ----------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_path", required=True, help="path to merged parquet")
    ap.add_argument("--out_dir", required=True, help="directory for csv outputs")
    ap.add_argument("--topn", type=int, default=20, help="top-n themes to list")
    args = ap.parse_args()

    df = pd.read_parquet(args.merged_path)
    df = to_lower_columns(df)
    df = df.loc[:, ~df.columns.duplicated(keep="last")]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # table 1
    t1 = table1_overview_by_source(df)
    t1.to_csv(out_dir / "table1_overview_by_source.csv", index=False)

    # table 2
    t2 = table2_device_freq(df)
    t2.to_csv(out_dir / "table2_device_freq.csv", index=False)

    # table 3
    t3e = table3_emotions(df)
    t3e.to_csv(out_dir / "table3_emotions.csv", index=False)

    t3t = table3_themes_top(df, topn=args.topn)
    t3t.to_csv(out_dir / "table3_themes_top.csv", index=False)

    t3s = table3_sentiment(df)
    t3s.to_csv(out_dir / "table3_sentiment.csv", index=False)

    # tiny console summary
    print("rows:", len(df))
    print("wrote:")
    for p in [
        "table1_overview_by_source.csv",
        "table2_device_freq.csv",
        "table3_emotions.csv",
        "table3_themes_top.csv",
        "table3_sentiment.csv",
    ]:
        print(" -", out_dir / p)


if __name__ == "__main__":
    main()
