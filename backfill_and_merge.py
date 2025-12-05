#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# --- helpers ---------------------------------------------------------------


def norm_text(s: str | None) -> str | None:
    # normalize to ascii, lowercase, collapse whitespace, strip punctuation
    if s is None:
        return None
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("ascii")
    s = s.lower().replace("&", " and ")
    s = re.sub(r"\s+", " ", s)
    s = re.sub(r"[^\w\s]", "", s)
    return s.strip()


def poem_id(author: str, title: str, poem: str) -> str:
    # deterministic 16-hex id based on author, title, and first 200 chars of poem
    key = f"{(author or '').strip()}\n{(title or '').strip()}\n{(poem or '').strip()[:200]}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


def normalize_for_poem_id_merge(df: pd.DataFrame) -> pd.DataFrame:
    # make columns a simple, unique, lowercase index with exactly one poem_id column
    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["__".join(map(str, t)).strip() for t in df.columns.to_list()]

    df.columns = [str(c).strip() for c in df.columns]

    # if poem_id is in the index, bring it back as a column
    index_names = [n for n in getattr(df.index, "names", []) if n is not None]
    if df.index.name == "poem_id" or ("poem_id" in index_names):
        df = df.reset_index()

    # coalesce any columns that are literally named poem_id (case-insensitive duplicates)
    # after this, we will standardize the label to lowercase "poem_id"
    colmap = {c: c for c in df.columns}
    # lower all columns (user preference)
    df.columns = [c.lower() for c in df.columns]

    # gather poem_id-likes that are exactly "poem_id"
    poemish = [c for c in df.columns if c == "poem_id"]
    if len(poemish) > 1:
        base = df[poemish].bfill(axis=1).iloc[:, 0]
        keep = poemish[-1]
        drop = [c for c in poemish if c != keep]
        df = df.drop(columns=drop)
        df[keep] = base

    # drop duplicate column labels (last wins)
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="last")]

    # final sanity: exactly one poem_id column
    if (pd.Index(df.columns) == "poem_id").sum() != 1:
        raise AssertionError("poem_id label still ambiguous")

    return df


def read_attrs(path: Path) -> pd.DataFrame:
    # load attrs in parquet/csv/jsonl; lower-case columns
    p = str(path)
    if p.endswith(".parquet"):
        df = pd.read_parquet(p)
    elif p.endswith(".csv"):
        df = pd.read_csv(p)
    elif p.endswith(".jsonl") or p.endswith(".ndjson"):
        df = pd.read_json(p, lines=True)
    else:
        raise ValueError(f"unsupported attrs format: {p}")
    df.columns = [c.lower() for c in df.columns]
    return df


def add_key_columns(df: pd.DataFrame, title_col: str, author_col: str) -> pd.DataFrame:
    # add normalized join keys for author and title if missing
    df = df.copy()
    if "title_key" not in df.columns:
        df["title_key"] = (
            df[title_col].map(norm_text) if title_col in df.columns else None
        )
    if "author_key" not in df.columns:
        df["author_key"] = (
            df[author_col].map(norm_text) if author_col in df.columns else None
        )
    return df


# --- main -----------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", default="haining/poem_interpretation_corpus")
    ap.add_argument(
        "--attrs_path", required=True, help="attrs csv/parquet/jsonl already produced"
    )
    ap.add_argument(
        "--attrs_id_col", default="poem_id", help="id column in attrs if present"
    )
    ap.add_argument(
        "--out_dir", default="poem_attrs", help="where to materialize per-row jsons"
    )
    ap.add_argument("--merged_out", default="dist_v1/merged.parquet")
    ap.add_argument(
        "--materialize_json",
        action="store_true",
        help="write poem_attrs/<split>/rows/*.json",
    )
    args = ap.parse_args()

    Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    Path(Path(args.merged_out).parent).mkdir(parents=True, exist_ok=True)

    # load hf dataset and build catalog with deterministic poem_id + keys
    ds_dict = load_dataset(args.dataset_id)
    rows = []
    for split, ds in ds_dict.items():
        for r in ds:
            author = r.get("author", "") or ""
            title = r.get("title", "") or ""
            poem = r.get("poem", "") or ""
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
    catalog.columns = [c.lower() for c in catalog.columns]
    catalog = add_key_columns(catalog, "title", "author")
    catalog = normalize_for_poem_id_merge(catalog)

    # read attrs and try to preserve id if supplied; otherwise prepare key-based fallback
    attrs = read_attrs(Path(args.attrs_path))

    # support alternative source column names: author_src/title_src if present
    author_col = (
        "author_src"
        if "author_src" in attrs.columns
        else ("author" if "author" in attrs.columns else None)
    )
    title_col = (
        "title_src"
        if "title_src" in attrs.columns
        else ("title" if "title" in attrs.columns else None)
    )
    poem_col = (
        "poem_src"
        if "poem_src" in attrs.columns
        else ("poem" if "poem" in attrs.columns else None)
    )

    # compute or validate poem_id on attrs
    if args.attrs_id_col in attrs.columns:
        # standardize to poem_id name
        if args.attrs_id_col != "poem_id":
            attrs = attrs.rename(columns={args.attrs_id_col: "poem_id"})
    else:
        # derive poem_id only if we have author+title+poem; otherwise we'll rely on key join
        if (
            author_col
            and title_col
            and poem_col
            and all(c in attrs.columns for c in [author_col, title_col, poem_col])
        ):
            attrs["poem_id"] = [
                poem_id(a, t, p)
                for a, t, p in zip(attrs[author_col], attrs[title_col], attrs[poem_col])
            ]
        else:
            # no id derivation possible; we'll just create keys for fallback join
            pass

    # add normalized keys for fallback matching
    if author_col or title_col:
        attrs = add_key_columns(attrs, title_col or "title", author_col or "author")

    # decide which columns are attributes (exclude ids/meta/text we don't want to overwrite)
    meta_drop = {
        "split",
        "poem",
        "interpretation",
        "title",
        "author",
        "title_src",
        "author_src",
        "poem_src",
        "source",
        "title_key",
        "author_key",
    }
    attr_cols = [c for c in attrs.columns if c not in meta_drop]
    # ensure poem_id is present in attr_cols when available
    if "poem_id" in attrs.columns and "poem_id" not in attr_cols:
        attr_cols = ["poem_id"] + attr_cols

    # normalize both sides for a clean merge
    attrs = normalize_for_poem_id_merge(attrs) if "poem_id" in attrs.columns else attrs
    attrs = attrs.loc[:, ~pd.Index(attrs.columns).duplicated(keep="last")]

    # 1) id-based merge if possible
    if "poem_id" in attrs.columns:
        merged = catalog.merge(
            attrs[["poem_id"] + [c for c in attr_cols if c != "poem_id"]],
            on="poem_id",
            how="left",
        )
        matched_by_id = (
            merged[[c for c in attr_cols if c != "poem_id"]].notna().any(axis=1)
        )
    else:
        merged = catalog.copy()
        matched_by_id = pd.Series(False, index=merged.index)

    # 2) fallback: exact author_key + title_key join for rows still without any attrs
    need_fallback = ~matched_by_id
    key_cols = ["author_key", "title_key"]

    if need_fallback.any() and set(key_cols).issubset(attrs.columns):
        # left: rows still missing attrs, keyed by normalized author/title
        left_sub = merged.loc[need_fallback, ["poem_id", *key_cols]].drop_duplicates()

        # right: one row per (author_key, title_key), keep last when multiple
        # exclude id and key columns from the attribute payload
        right_cols = [c for c in attrs.columns if c not in {"poem_id", *key_cols}]
        right_sub = (
            attrs.loc[:, [*key_cols, *right_cols]]
            .sort_values(key_cols)
            .drop_duplicates(subset=key_cols, keep="last")
        )

        # join on keys and map back by poem_id only for rows that still lack attrs
        m_key = left_sub.merge(right_sub, on=key_cols, how="left").set_index("poem_id")

        attr_targets = right_cols  # payload only, keys already excluded
        for c in attr_targets:
            if c not in merged.columns:
                merged[c] = pd.NA
            merged.loc[need_fallback, c] = merged.loc[need_fallback, c].combine_first(
                merged.loc[need_fallback, "poem_id"].map(m_key[c])
            )

    # coverage report
    # infer attribute fields actually added (exclude catalog columns)
    catalog_cols = {
        "split",
        "poem_id",
        "author",
        "title",
        "poem",
        "interpretation",
        "source",
        "author_key",
        "title_key",
    }
    candidate_attr_cols = [c for c in merged.columns if c not in catalog_cols]
    has_any_attr = (
        merged[candidate_attr_cols].notna().any(axis=1)
        if candidate_attr_cols
        else pd.Series(False, index=merged.index)
    )

    print(f"total rows: {len(merged)}")
    print(f"with attrs: {int(has_any_attr.sum())}")
    print(f"coverage   : {has_any_attr.mean():.2%}")

    # save final parquet
    # drop helper keys unless you want to keep for debugging
    keep_cols = [c for c in merged.columns if c not in {"author_key", "title_key"}]
    merged[keep_cols].to_parquet(args.merged_out, index=False)

    # optionally materialize per-row jsons with only the attribute fields
    if args.materialize_json and candidate_attr_cols:
        for split, g in merged.groupby("split", sort=False):
            base = Path(args.out_dir) / split / "rows"
            base.mkdir(parents=True, exist_ok=True)
            # write only non-catalog columns
            att_only = g[["poem_id"] + candidate_attr_cols]
            for r in att_only.to_dict(orient="records"):
                pid = r.pop("poem_id")
                # skip rows with no attrs at all
                if all(pd.isna(v) for v in r.values()):
                    continue
                (base / f"{pid}.json").write_text(json.dumps(r, ensure_ascii=False))
        print("materialized json rows under:", args.out_dir)


if __name__ == "__main__":
    main()
