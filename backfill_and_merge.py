#!/usr/bin/env python3
import argparse
import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# --------------------- helpers ---------------------


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

    # lower all columns
    df.columns = [c.lower() for c in df.columns]

    # coalesce any duplicate poem_id columns (exact name match)
    poemish = [c for c in df.columns if c == "poem_id"]
    if len(poemish) > 1:
        base = df[poemish].bfill(axis=1).iloc[:, 0]
        keep = poemish[-1]
        drop = [c for c in poemish if c != keep]
        df = df.drop(columns=drop)
        df[keep] = base

    # drop duplicate column labels (last wins)
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="last")]

    # final sanity: at most one poem_id column
    if (pd.Index(df.columns) == "poem_id").sum() > 1:
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
    if "title_key" not in df.columns and title_col in df.columns:
        df["title_key"] = df[title_col].map(norm_text)
    if "author_key" not in df.columns and author_col in df.columns:
        df["author_key"] = df[author_col].map(norm_text)
    return df


def collapse_on_keys(
    df: pd.DataFrame, keys: list[str], payload_cols: list[str]
) -> pd.DataFrame:
    # stable collapse without hashing payloads (avoids unhashable list/ndarray errors)
    cols = [*keys, *payload_cols]
    cols = [c for c in cols if c in df.columns]
    if not cols:
        return pd.DataFrame(columns=keys + payload_cols)
    out = (
        df[cols]
        .groupby(keys, as_index=False, dropna=False)
        .last()  # keep last occurrence
    )
    return out


# --------------------- main ---------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", default="haining/poem_interpretation_corpus")
    ap.add_argument(
        "--attrs_path", required=True, help="attrs csv/parquet/jsonl already produced"
    )
    ap.add_argument(
        "--attrs_id_col",
        default="poem_id",
        help="id column in attrs if present (e.g., poem_id_file)",
    )
    ap.add_argument(
        "--out_dir", default="poem_attrs", help="where to materialize per-row jsons"
    )
    ap.add_argument("--merged_out", default="dist_v1/merged.parquet")
    ap.add_argument(
        "--materialize_json",
        action="store_true",
        help="write poem_attrs/<split>/rows/*.json with only attribute fields",
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

    # read attrs
    attrs = read_attrs(Path(args.attrs_path))
    attrs = normalize_for_poem_id_merge(attrs)

    # prefer explicit attrs id if present
    if args.attrs_id_col in attrs.columns and args.attrs_id_col != "poem_id":
        attrs = attrs.rename(columns={args.attrs_id_col: "poem_id"})

    # detect optional source column names
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

    # add normalized keys when we have source columns
    if author_col or title_col:
        attrs = add_key_columns(attrs, title_col or "title", author_col or "author")

    # meta we never overwrite into catalog
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
        "poem_id_json",
    }

    # payload candidates (attributes)
    payload_cols_all = [c for c in attrs.columns if c not in meta_drop]

    # ensure no duplicate column labels
    attrs = attrs.loc[:, ~pd.Index(attrs.columns).duplicated(keep="last")]

    # start with a clean merged frame
    merged = catalog.copy()
    for c in payload_cols_all:
        if c not in merged.columns:
            merged[c] = pd.NA

    def has_any_attr_mask(df: pd.DataFrame) -> pd.Series:
        base_cols = {
            "split",
            "poem_id",
            "author",
            "title",
            "poem",
            "interpretation",
            "source",
        }
        cand = [c for c in df.columns if c not in base_cols]
        if not cand:
            return pd.Series(False, index=df.index)
        return df[cand].notna().any(axis=1)

    matched = has_any_attr_mask(merged)

    # ---------- strategy 1: id join on (split, poem_id) if attrs has poem_id ----------
    matched_counts = {}

    if "poem_id" in attrs.columns:
        # collapse attrs by keys first
        right = collapse_on_keys(
            attrs,
            keys=["split", "poem_id"],
            payload_cols=[c for c in payload_cols_all if c != "poem_id"],
        )
        left = merged.loc[~matched, ["split", "poem_id"]].drop_duplicates()
        m = left.merge(right, on=["split", "poem_id"], how="left").set_index(
            merged.loc[~matched, "poem_id"].index
        )
        # map by positional alignment of left subset
        fill_cols = [c for c in right.columns if c not in {"split", "poem_id"}]
        for c in fill_cols:
            merged.loc[~matched, c] = merged.loc[~matched, c].combine_first(m[c])
        matched = has_any_attr_mask(merged)
        matched_counts["by_id"] = int(matched.sum())

    # ---------- strategy 2: json id (poem_id_json) -> catalog poem_id ----------
    if "poem_id_json" in attrs.columns and (~matched).any():
        right = collapse_on_keys(
            attrs.rename(columns={"poem_id_json": "rid"}),
            keys=["split", "rid"],
            payload_cols=[
                c for c in payload_cols_all if c not in {"poem_id", "poem_id_json"}
            ],
        )
        left = merged.loc[~matched, ["split", "poem_id"]].rename(
            columns={"poem_id": "rid"}
        )
        m = left.merge(right, on=["split", "rid"], how="left")
        fill_cols = [c for c in right.columns if c not in {"split", "rid"}]
        for c in fill_cols:
            merged.loc[~matched, c] = merged.loc[~matched, c].combine_first(m[c])
        matched = has_any_attr_mask(merged)
        matched_counts["by_json_id"] = int(matched.sum())

    # ---------- strategy 3: exact (author, title) ----------
    if author_col and title_col and (~matched).any():
        right = collapse_on_keys(
            attrs.rename(
                columns={author_col: "author_src__", title_col: "title_src__"}
            ),
            keys=["author_src__", "title_src__"],
            payload_cols=[c for c in payload_cols_all if c not in {"poem_id"}],
        )
        left = merged.loc[~matched, ["author", "title"]].rename(
            columns={"author": "author_src__", "title": "title_src__"}
        )
        m = left.merge(right, on=["author_src__", "title_src__"], how="left")
        fill_cols = [
            c for c in right.columns if c not in {"author_src__", "title_src__"}
        ]
        for c in fill_cols:
            merged.loc[~matched, c] = merged.loc[~matched, c].combine_first(m[c])
        matched = has_any_attr_mask(merged)
        matched_counts["by_exact_author_title"] = int(matched.sum())

    # ---------- strategy 4: normalized (author_key, title_key) ----------
    if {"author_key", "title_key"}.issubset(attrs.columns) and (~matched).any():
        right = collapse_on_keys(
            attrs,
            keys=["author_key", "title_key"],
            payload_cols=[c for c in payload_cols_all if c not in {"poem_id"}],
        )
        left = merged.loc[~matched, ["author_key", "title_key"]]
        m = left.merge(right, on=["author_key", "title_key"], how="left")
        fill_cols = [c for c in right.columns if c not in {"author_key", "title_key"}]
        for c in fill_cols:
            merged.loc[~matched, c] = merged.loc[~matched, c].combine_first(m[c])
        matched = has_any_attr_mask(merged)
        matched_counts["by_norm_author_title"] = int(matched.sum())

    # coverage report
    total = len(merged)
    any_attr = has_any_attr_mask(merged)
    print(f"total rows: {total}")
    print(f"with attrs: {int(any_attr.sum())}")
    print(f"coverage   : {any_attr.mean():.2%}")
    if matched_counts:
        # incremental counts for transparency
        prev = 0
        for k, v in matched_counts.items():
            print(f"  matched {k:>22}: +{v - prev}")
            prev = v

    # save final parquet (drop internal keys)
    keep_cols = [c for c in merged.columns if c not in {"author_key", "title_key"}]
    Path(args.merged_out).parent.mkdir(parents=True, exist_ok=True)
    merged[keep_cols].to_parquet(args.merged_out, index=False)

    # optionally materialize per-row jsons with only the attribute fields
    base_cols = {
        "split",
        "poem_id",
        "author",
        "title",
        "poem",
        "interpretation",
        "source",
    }
    candidate_attr_cols = [c for c in merged.columns if c not in base_cols]
    if args.materialize_json and candidate_attr_cols:
        for split, g in merged.groupby("split", sort=False):
            base = Path(args.out_dir) / split / "rows"
            base.mkdir(parents=True, exist_ok=True)
            att_only = g[["poem_id"] + candidate_attr_cols]
            for r in att_only.to_dict(orient="records"):
                pid = r.pop("poem_id")
                if all(pd.isna(v) for v in r.values()):
                    continue
                (base / f"{pid}.json").write_text(json.dumps(r, ensure_ascii=False))
        print("materialized json rows under:", args.out_dir)


if __name__ == "__main__":
    main()
