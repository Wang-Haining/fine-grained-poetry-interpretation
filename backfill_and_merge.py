#!/usr/bin/env python3
import argparse
import hashlib
import json
from pathlib import Path

# --- add near the top ---
import pandas as pd
from datasets import load_dataset


def _normalize_for_poem_id_merge(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # flatten multiindex columns if any
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["__".join(map(str, t)).strip() for t in df.columns.to_list()]

    # normalize names
    df.columns = [str(c).strip() for c in df.columns]

    # if poem_id appears in the index, bring it back as a column
    index_names = [n for n in getattr(df.index, "names", []) if n is not None]
    if df.index.name == "poem_id" or ("poem_id" in index_names):
        df = df.reset_index()

    # coalesce any poem_id-like duplicates (e.g., poem_id, poem_id.1)
    poemish = [c for c in df.columns if c.strip().lower() == "poem_id"]
    if len(poemish) > 1:
        base = df[poemish].bfill(axis=1).iloc[:, 0]
        keep = poemish[-1]
        drop = [c for c in poemish if c != keep]
        df = df.drop(columns=drop)
        df[keep] = base

    # drop duplicate column labels (last wins)
    df = df.loc[:, ~pd.Index(df.columns).duplicated(keep="last")]

    # final sanity: exactly one poem_id column
    assert (
        pd.Index(df.columns) == "poem_id"
    ).sum() == 1, "poem_id label still ambiguous"
    return df


def poem_id(author: str, title: str, poem: str) -> str:
    key = f"{(author or '').strip()}\n{(title or '').strip()}\n{(poem or '').strip()[:200]}"
    return hashlib.sha1(key.encode()).hexdigest()[:16]


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
    df.columns = [c.lower() for c in df.columns]
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_id", default="haining/poem_interpretation_corpus")
    ap.add_argument(
        "--attrs_path", required=True, help="attrs csv/parquet/jsonl already produced"
    )
    ap.add_argument(
        "--attrs_id_col", default="poem_id", help="if attrs already has an id, set here"
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

    # load hf splits and build catalog
    ds_dict = load_dataset(args.dataset_id)
    rows = []
    for split, ds in ds_dict.items():
        for r in ds:
            pid = poem_id(r.get("author", ""), r.get("title", ""), r.get("poem", ""))
            rows.append(
                {
                    "split": split,
                    "poem_id": pid,
                    "author": r.get("author", ""),
                    "title": r.get("title", ""),
                    "poem": r.get("poem", ""),
                    "interpretation": r.get("interpretation", ""),
                    "source": r.get("source", ""),
                }
            )
    catalog = pd.DataFrame(rows)
    catalog.columns = [c.lower() for c in catalog.columns]

    # read attrs and ensure id
    attrs = read_attrs(Path(args.attrs_path))
    if args.attrs_id_col not in attrs.columns:
        # try to compute poem_id if attrs has author/title/poem
        need = {"author", "title", "poem"}
        if need.issubset(set(attrs.columns)):
            attrs["poem_id"] = [
                poem_id(a, t, p)
                for a, t, p in zip(attrs["author"], attrs["title"], attrs["poem"])
            ]
        else:
            raise ValueError(
                f"attrs has no '{args.attrs_id_col}' and lacks columns to derive it"
            )

    # keep only id + attrs columns (drop huge text if present)
    drop_like = {"poem", "interpretation", "title", "author", "source", "split"}
    attr_cols = ["poem_id"] + [c for c in attrs.columns if c not in drop_like]
    attrs = attrs[attr_cols].drop_duplicates("poem_id")

    # merge
    catalog = _normalize_for_poem_id_merge(catalog)
    attrs = _normalize_for_poem_id_merge(attrs)
    merged = catalog.merge(attrs, on="poem_id", how="left")
    merged.to_parquet(args.merged_out, index=False)

    # quick coverage report
    cov = merged.assign(has_attrs=merged[attr_cols[1:]].notna().any(axis=1))
    print("total rows:", len(cov))
    print("with attrs:", int(cov["has_attrs"].sum()))
    print("coverage   :", f"{cov['has_attrs'].mean():.2%}")

    # optionally materialize per-row jsons for downstream tools
    if args.materialize_json:
        for split, g in merged.groupby("split"):
            base = Path(args.out_dir) / split / "rows"
            base.mkdir(parents=True, exist_ok=True)
            # write only the attribute fields
            att_only = g[["poem_id"] + [c for c in attr_cols if c != "poem_id"]]
            for r in att_only.to_dict(orient="records"):
                pid = r.pop("poem_id")
                (base / f"{pid}.json").write_text(json.dumps(r, ensure_ascii=False))
        print("materialized json rows under:", args.out_dir)


if __name__ == "__main__":
    main()
