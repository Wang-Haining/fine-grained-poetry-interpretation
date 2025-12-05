"""
merge interpretations and tags, mask poetry foundation poems for public release,
and compute summary tables (per source and aggregated) with mean and 95% ci.

inputs
  --catalog: main corpus file (csv/parquet/jsonl) with at least: author,title,poem,source
  --attrs: attributes file with id and columns: emotion, sentiment, themes, themes_50
  --interps: optional interpretations file with id and column: interpretation
  --catalog_id_col / --attrs_id_col / --interps_id_col: id columns; if missing, a stable poem_id is created
  note: all columns are normalized to lowercase

outputs (written to --out_dir)
  merged_full.parquet           # full internal use
  merged_public.parquet         # poem blanked where source is poetry foundation
  table1_overview.csv           # counts and interpretation word stats with 95% ci
  table2_devices.csv            # literary device counts per source and total
  table3_emotions.csv           # emotion distribution
  table3_sentiment.csv          # sentiment distribution
  table3_themes50.csv           # theme (fixed-50) distribution
  stats_summary.md              # small markdown with the above tables
"""

from __future__ import annotations

import argparse
import hashlib
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def read_any(path: Path) -> pd.DataFrame:
    suf = path.suffix.lower()
    if suf in {".csv"}:
        df = pd.read_csv(path)
    elif suf in {".parquet"}:
        df = pd.read_parquet(path)
    elif suf in {".jsonl", ".json"}:
        rows = []
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rows.append(pd.read_json(line, typ="series").to_dict())
        df = pd.DataFrame(rows)
    else:
        raise ValueError(f"unsupported file type: {suf}")
    df.columns = [c.lower() for c in df.columns]
    return df


def stable_poem_id(author: str, title: str, poem: str) -> str:
    key = f"{(author or '').strip().lower()}||{(title or '').strip().lower()}||{(poem or '').strip().lower()[:200]}"
    return hashlib.sha1(key.encode("utf-8")).hexdigest()


def ensure_id(df: pd.DataFrame, id_col: Optional[str]) -> Tuple[pd.DataFrame, str]:
    if id_col and id_col in df.columns:
        return df, id_col
    # create poem_id deterministically
    for col in ["author", "title", "poem"]:
        if col not in df.columns:
            df[col] = ""
    df["poem_id"] = [
        stable_poem_id(a, t, p)
        for a, t, p in zip(df["author"], df["title"], df["poem"])
    ]
    return df, "poem_id"


def normalize_source(df: pd.DataFrame) -> pd.Series:
    s = df.get("source", pd.Series(["unknown"] * len(df))).astype(str).str.lower()
    is_pf = s.str.contains("poetry") & s.str.contains("foundation")
    is_gut = s.str.contains("gutenberg")
    out = pd.Series(["other"] * len(df))
    out[is_pf] = "poetry_foundation"
    out[is_gut] = "public_domain"
    return out


def parse_semicolon_list(x: Optional[str]) -> List[str]:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return []
    xs = [t.strip().lower() for t in str(x).split(";")]
    return [t for t in xs if t]


def word_count(text: Optional[str]) -> float:
    if not isinstance(text, str) or not text.strip():
        return np.nan
    return float(len(re.findall(r"\b\w+\b", text)))


def mean_ci95(series: pd.Series) -> Tuple[float, float, float]:
    vals = series.dropna().astype(float).values
    n = len(vals)
    if n == 0:
        return (float("nan"), float("nan"), float("nan"))
    m = float(np.mean(vals))
    sd = float(np.std(vals, ddof=1)) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n > 1 else 0.0
    ci = 1.96 * se if n > 1 else 0.0
    return (m, m - ci, m + ci)


def devices_table(df: pd.DataFrame) -> pd.DataFrame:
    # simple keyword list, counted once per poem
    devices = [
        "metaphor",
        "simile",
        "alliteration",
        "assonance",
        "consonance",
        "imagery",
        "personification",
        "symbolism",
        "hyperbole",
        "irony",
        "onomatopoeia",
        "oxymoron",
        "paradox",
        "allegory",
        "apostrophe",
        "enjambment",
        "caesura",
        "anaphora",
        "repetition",
        "parallelism",
        "rhyme",
        "meter",
        "blank verse",
        "free verse",
        "iambic pentameter",
        "sonnet",
        "volta",
        "couplet",
        "quatrain",
        "sestet",
        "tercet",
        "villanelle",
        "sestina",
        "terza rima",
        "ballad",
        "ode",
        "elegy",
        "haiku",
        "limerick",
    ]
    patt = {d: re.compile(rf"\b{re.escape(d)}\b", flags=re.I) for d in devices}

    def has_any(text: str, rx) -> int:
        if not isinstance(text, str):
            return 0
        return 1 if rx.search(text) else 0

    out_rows = []
    for group_name, gdf in [
        ("public_domain", df[df["source_norm"] == "public_domain"]),
        ("poetry_foundation", df[df["source_norm"] == "poetry_foundation"]),
        ("all", df),
    ]:
        row = {"group": group_name, "n_poems": len(gdf)}
        interp = gdf.get("interpretation", pd.Series([None] * len(gdf)))
        for d, rx in patt.items():
            row[d] = int(interp.apply(lambda t: has_any(t, rx)).sum())
        out_rows.append(row)
    out = pd.DataFrame(out_rows)
    return out


def table1_overview(df: pd.DataFrame) -> pd.DataFrame:
    def one_group(g: pd.DataFrame, name: str) -> Dict[str, object]:
        cnt = len(g)
        poets = g["author"].dropna().astype(str).str.strip().str.lower().nunique()
        m, lo, hi = mean_ci95(g["interp_words"])
        return {
            "group": name,
            "n_poems": int(cnt),
            "n_unique_authors": int(poets),
            "interp_words_mean": round(m, 2) if not math.isnan(m) else np.nan,
            "interp_words_ci95_low": round(lo, 2) if not math.isnan(lo) else np.nan,
            "interp_words_ci95_high": round(hi, 2) if not math.isnan(hi) else np.nan,
        }

    rows = [
        one_group(df[df["source_norm"] == "public_domain"], "public_domain"),
        one_group(df[df["source_norm"] == "poetry_foundation"], "poetry_foundation"),
        one_group(df, "all"),
    ]
    return pd.DataFrame(rows)


def table3_distributions(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    # emotions
    emo_rows = []
    for group_name, gdf in [
        ("public_domain", df[df["source_norm"] == "public_domain"]),
        ("poetry_foundation", df[df["source_norm"] == "poetry_foundation"]),
        ("all", df),
    ]:
        emo_lists = gdf["emotion_list"].explode().dropna()
        emo_counts = emo_lists.value_counts().reset_index()
        emo_counts.columns = ["emotion", "count"]
        emo_counts.insert(0, "group", group_name)
        emo_rows.append(emo_counts)
    emotions = (
        pd.concat(emo_rows, ignore_index=True)
        if emo_rows
        else pd.DataFrame(columns=["group", "emotion", "count"])
    )

    # sentiment
    sent_rows = []
    for group_name, gdf in [
        ("public_domain", df[df["source_norm"] == "public_domain"]),
        ("poetry_foundation", df[df["source_norm"] == "poetry_foundation"]),
        ("all", df),
    ]:
        s_counts = gdf["sentiment"].dropna().str.lower().value_counts().reset_index()
        s_counts.columns = ["sentiment", "count"]
        s_counts.insert(0, "group", group_name)
        sent_rows.append(s_counts)
    sentiments = (
        pd.concat(sent_rows, ignore_index=True)
        if sent_rows
        else pd.DataFrame(columns=["group", "sentiment", "count"])
    )

    # themes_50
    th_rows = []
    for group_name, gdf in [
        ("public_domain", df[df["source_norm"] == "public_domain"]),
        ("poetry_foundation", df[df["source_norm"] == "poetry_foundation"]),
        ("all", df),
    ]:
        th_counts = gdf["themes50_list"].explode().dropna()
        th_counts = th_counts[th_counts != ""]
        th_counts = th_counts.value_counts().reset_index()
        th_counts.columns = ["theme", "count"]
        th_counts.insert(0, "group", group_name)
        th_rows.append(th_counts)
    themes50 = (
        pd.concat(th_rows, ignore_index=True)
        if th_rows
        else pd.DataFrame(columns=["group", "theme", "count"])
    )

    return emotions, sentiments, themes50


def write_markdown_summary(
    out_dir: Path,
    t1: pd.DataFrame,
    devices: pd.DataFrame,
    emo: pd.DataFrame,
    sent: pd.DataFrame,
    th50: pd.DataFrame,
) -> None:
    def df_to_md(df: pd.DataFrame) -> str:
        return df.to_markdown(index=False)

    parts = []
    parts.append("# statistical summary\n")
    parts.append("## table 1: overview\n")
    parts.append(df_to_md(t1))
    parts.append("\n\n## table 2: literary device counts\n")
    parts.append(
        df_to_md(devices.iloc[:, :20])
    )  # show first 20 columns to keep it readable
    parts.append("\n\n## table 3a: emotion distribution\n")
    parts.append(df_to_md(emo.head(50)))
    parts.append("\n\n## table 3b: sentiment distribution\n")
    parts.append(df_to_md(sent))
    parts.append("\n\n## table 3c: top themes (fixed 50)\n")
    top = (
        th50.sort_values(["group", "count"], ascending=[True, False])
        .groupby("group")
        .head(20)
    )
    parts.append(df_to_md(top))
    (out_dir / "stats_summary.md").write_text("\n".join(parts), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--catalog",
        required=True,
        help="csv/parquet/jsonl with author,title,poem,source and id if available",
    )
    ap.add_argument(
        "--attrs",
        required=True,
        help="attributes csv/parquet/jsonl with id, emotion, sentiment, themes, themes_50",
    )
    ap.add_argument(
        "--interps",
        default=None,
        help="optional interpretations file with id and interpretation",
    )
    ap.add_argument("--catalog_id_col", default=None)
    ap.add_argument("--attrs_id_col", default=None)
    ap.add_argument("--interps_id_col", default=None)
    ap.add_argument("--out_dir", default="merged_out")
    ap.add_argument(
        "--mask_columns",
        default="poem",
        help="comma separated columns to blank for poetry foundation in public release",
    )
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    df_cat = read_any(Path(args.catalog))
    df_cat, cat_id = ensure_id(df_cat, args.catalog_id_col)

    df_attrs = read_any(Path(args.attrs))
    df_attrs, attrs_id = ensure_id(df_attrs, args.attrs_id_col)

    if args.interps:
        df_int = read_any(Path(args.interps))
        df_int, int_id = ensure_id(df_int, args.interps_id_col)
    else:
        df_int = None
        int_id = None

    # select and normalize attrs columns
    for col in ["emotion", "sentiment", "themes", "themes_50"]:
        if col not in df_attrs.columns:
            raise ValueError(f"missing column in attrs: {col}")

    # merge catalog + interps (if provided) + attrs
    df = df_cat.copy()
    if df_int is not None:
        keep_int = [int_id, "interpretation"]
        for col in keep_int:
            if col not in df_int.columns:
                raise ValueError(f"missing column in interps: {col}")
        df = df.merge(df_int[keep_int], left_on=cat_id, right_on=int_id, how="left")
        if int_id in df.columns and int_id != cat_id:
            df = df.drop(columns=[int_id])

    keep_attrs = [attrs_id, "emotion", "sentiment", "themes", "themes_50"]
    df = df.merge(df_attrs[keep_attrs], left_on=cat_id, right_on=attrs_id, how="left")
    if attrs_id in df.columns and attrs_id != cat_id:
        df = df.drop(columns=[attrs_id])

    # normalize types and helper lists for stats
    df["source_norm"] = normalize_source(df)
    if "interpretation" not in df.columns:
        df["interpretation"] = None
    df["interp_words"] = df["interpretation"].apply(word_count)

    df["emotion_list"] = (
        df["emotion"].apply(parse_semicolon_list)
        if "emotion" in df.columns
        else [[]] * len(df)
    )
    df["themes50_list"] = (
        df["themes_50"].apply(parse_semicolon_list)
        if "themes_50" in df.columns
        else [[]] * len(df)
    )

    # save full internal version
    full_path = out_dir / "merged_full.parquet"
    df.to_parquet(full_path, index=False)

    # build public masked version
    mask_cols = [c.strip().lower() for c in args.mask_columns.split(",") if c.strip()]
    df_public = df.copy()
    pf_mask = df_public["source_norm"] == "poetry_foundation"
    for col in mask_cols:
        if col in df_public.columns:
            df_public.loc[pf_mask, col] = ""
    public_path = out_dir / "merged_public.parquet"
    df_public.to_parquet(public_path, index=False)

    # compute tables
    t1 = table1_overview(df_public)
    t1.to_csv(out_dir / "table1_overview.csv", index=False)

    dev = devices_table(df_public)
    dev.to_csv(out_dir / "table2_devices.csv", index=False)

    emo, sent, th50 = table3_distributions(df_public)
    emo.to_csv(out_dir / "table3_emotions.csv", index=False)
    sent.to_csv(out_dir / "table3_sentiment.csv", index=False)
    th50.to_csv(out_dir / "table3_themes50.csv", index=False)

    write_markdown_summary(out_dir, t1, dev, emo, sent, th50)

    print(
        f"[done] wrote:\n- {full_path}\n- {public_path}\n- {out_dir}/table1_overview.csv\n- {out_dir}/table2_devices.csv\n- {out_dir}/table3_emotions.csv\n- {out_dir}/table3_sentiment.csv\n- {out_dir}/table3_themes50.csv\n- {out_dir}/stats_summary.md"
    )


if __name__ == "__main__":
    main()
