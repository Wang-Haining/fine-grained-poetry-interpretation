#!/usr/bin/env python3
"""
restore_pf_from_local_source.py

creates two local outputs from the now-canonical corpus by restoring Poetry Foundation
interpretations (and optionally PF poem text for internal use) from a local artifact.

outputs:
- internal_full/data/{train,validation,test}-00000-of-00001.parquet
- public_masked/data/{train,validation,test}-00000-of-00001.parquet

recommended PF source:
- a datasets.save_to_disk() directory (often looks like *.hf)
- or a directory with data/{train,validation,test}-*.parquet
- or a single parquet/csv with PF rows
"""

from __future__ import annotations

import argparse
import hashlib
import os
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
from datasets import Dataset, DatasetDict, load_dataset, load_from_disk
from huggingface_hub import login, snapshot_download
from tqdm.auto import tqdm


def try_login() -> None:
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        # if the dataset is public, this is harmless; if private, it will prompt
        try:
            login()
        except Exception:
            pass


def norm_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def make_key(author: Optional[str], title: Optional[str], source: Optional[str]) -> str:
    base = "||".join([norm_text(author), norm_text(title), norm_text(source)])
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def is_missing_text(x: Any) -> bool:
    if x is None:
        return True
    s = str(x).strip()
    return s == ""


def load_hf_dataset_with_parquet_fallback(repo_id: str) -> DatasetDict:
    try:
        return load_dataset(repo_id)
    except Exception:
        snap_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
        data_dir = Path(snap_dir) / "data"
        data_files: Dict[str, str] = {}
        for split in ["train", "validation", "test"]:
            files = sorted(data_dir.glob(f"{split}-*.parquet"))
            if files:
                data_files[split] = str(files[0])
        if not data_files:
            raise RuntimeError(f"no parquet files found in snapshot for {repo_id}")
        return load_dataset("parquet", data_files=data_files)


def wrap_as_datasetdict(obj) -> DatasetDict:
    if isinstance(obj, DatasetDict):
        return obj
    if isinstance(obj, Dataset):
        return DatasetDict({"train": obj})
    raise TypeError(f"unsupported dataset object type: {type(obj)}")


def load_local_pf_source(path: str) -> DatasetDict:
    p = Path(path)

    if not p.exists():
        raise FileNotFoundError(f"pf source path not found: {p}")

    # 1) datasets.save_to_disk directory (often named *.hf)
    if p.is_dir():
        try:
            obj = load_from_disk(str(p))
            return wrap_as_datasetdict(obj)
        except Exception:
            # 2) parquet directory layouts
            data_dir = p / "data" if (p / "data").exists() else p
            data_files: Dict[str, str] = {}
            for split in ["train", "validation", "test"]:
                files = sorted(data_dir.glob(f"{split}-*.parquet"))
                if files:
                    data_files[split] = str(files[0])
            if data_files:
                return load_dataset("parquet", data_files=data_files)
            raise RuntimeError(
                f"directory {p} is neither a datasets.save_to_disk dir nor contains split parquets"
            )

    # 3) single file (parquet/csv/jsonl)
    suf = p.suffix.lower()
    if suf == ".parquet":
        ds = load_dataset("parquet", data_files={"train": str(p)})
        return DatasetDict({"train": ds["train"]})
    if suf == ".csv":
        ds = load_dataset("csv", data_files={"train": str(p)})
        return DatasetDict({"train": ds["train"]})
    if suf in {".jsonl", ".json"}:
        ds = load_dataset("json", data_files={"train": str(p)})
        return DatasetDict({"train": ds["train"]})

    raise RuntimeError(f"unsupported pf source file type: {p}")


def count_pf_missing(dsd: DatasetDict) -> Dict[str, int]:
    total = 0
    poem_missing = 0
    interp_missing = 0
    for split in dsd.keys():
        for row in dsd[split]:
            if row.get("source") != "poetry_foundation":
                continue
            total += 1
            poem_missing += int(is_missing_text(row.get("poem")))
            interp_missing += int(is_missing_text(row.get("interpretation")))
    return {
        "pf_total": total,
        "pf_poem_missing": poem_missing,
        "pf_interpretation_missing": interp_missing,
    }


def build_pf_map(
    dsd: DatasetDict,
) -> Tuple[Dict[str, Dict[str, Optional[str]]], Dict[str, int]]:
    """
    key -> {"poem": str|None, "interpretation": str|None} for PF rows.
    keep longer values on duplicates.
    """
    mapping: Dict[str, Dict[str, Optional[str]]] = {}
    stats = {
        "pf_rows_seen": 0,
        "pf_rows_with_poem": 0,
        "pf_rows_with_interpretation": 0,
        "key_duplicates": 0,
    }

    for split in dsd.keys():
        for row in tqdm(dsd[split], desc=f"index pf source:{split}"):
            if row.get("source") != "poetry_foundation":
                continue

            stats["pf_rows_seen"] += 1

            poem_val = (
                None if is_missing_text(row.get("poem")) else str(row.get("poem"))
            )
            interp_val = (
                None
                if is_missing_text(row.get("interpretation"))
                else str(row.get("interpretation"))
            )

            if poem_val is not None:
                stats["pf_rows_with_poem"] += 1
            if interp_val is not None:
                stats["pf_rows_with_interpretation"] += 1

            k = make_key(row.get("author"), row.get("title"), row.get("source"))
            if k not in mapping:
                mapping[k] = {"poem": poem_val, "interpretation": interp_val}
                continue

            stats["key_duplicates"] += 1
            if poem_val and (
                not mapping[k].get("poem")
                or len(poem_val) > len(mapping[k]["poem"] or "")
            ):
                mapping[k]["poem"] = poem_val
            if interp_val and (
                not mapping[k].get("interpretation")
                or len(interp_val) > len(mapping[k]["interpretation"] or "")
            ):
                mapping[k]["interpretation"] = interp_val

    return mapping, stats


def apply_restore(
    canonical: DatasetDict,
    pf_map: Dict[str, Dict[str, Optional[str]]],
    *,
    mode: str,  # internal|public
    overwrite: bool,
    restore_pf_poem_in_internal: bool,
) -> DatasetDict:
    def update_batch(batch):
        authors = batch["author"]
        titles = batch["title"]
        sources = batch["source"]
        poems = batch["poem"]
        interps = batch["interpretation"]

        for i in range(len(authors)):
            if sources[i] != "poetry_foundation":
                continue

            k = make_key(authors[i], titles[i], sources[i])
            rec = pf_map.get(k)

            # restore interpretation
            if rec and rec.get("interpretation"):
                if overwrite or is_missing_text(interps[i]):
                    interps[i] = rec["interpretation"]

            # handle poem text
            if mode == "public":
                poems[i] = None  # mask PF poem only
            else:
                if restore_pf_poem_in_internal and rec and rec.get("poem"):
                    if overwrite or is_missing_text(poems[i]):
                        poems[i] = rec["poem"]

        batch["poem"] = poems
        batch["interpretation"] = interps
        return batch

    fixed = DatasetDict()
    for split in canonical.keys():
        fixed[split] = canonical[split].map(update_batch, batched=True, batch_size=1000)
    return fixed


def write_parquet_splits(dsd: DatasetDict, out_dir: Path) -> None:
    out_data = out_dir / "data"
    out_data.mkdir(parents=True, exist_ok=True)
    for split in ["train", "validation", "test"]:
        if split in dsd:
            dsd[split].to_parquet(str(out_data / f"{split}-00000-of-00001.parquet"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo_canonical", default="haining/structured_poem_interpretation_corpus"
    )
    ap.add_argument(
        "--pf_source_path",
        required=True,
        help="local artifact containing PF interpretation (and optionally PF poem)",
    )
    ap.add_argument("--out_internal_dir", default="internal_full")
    ap.add_argument("--out_public_dir", default="public_masked")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--restore_pf_poem_in_internal", action="store_true")
    args = ap.parse_args()

    try_login()

    print("loading canonical:", args.repo_canonical)
    canonical = load_hf_dataset_with_parquet_fallback(args.repo_canonical)
    print("canonical PF before:", count_pf_missing(canonical))

    print("loading PF source from:", args.pf_source_path)
    pf_source = load_local_pf_source(args.pf_source_path)
    print("pf source PF status:", count_pf_missing(pf_source))

    pf_map, pf_stats = build_pf_map(pf_source)
    print("pf map stats:", pf_stats)
    usable_interp = sum(1 for v in pf_map.values() if v.get("interpretation"))
    usable_poem = sum(1 for v in pf_map.values() if v.get("poem"))
    print(f"pf map usable: interpretation={usable_interp:,} poem={usable_poem:,}")

    if usable_interp == 0:
        raise RuntimeError(
            "PF source contains zero PF interpretations; cannot restore."
        )

    internal = apply_restore(
        canonical,
        pf_map,
        mode="internal",
        overwrite=args.overwrite,
        restore_pf_poem_in_internal=args.restore_pf_poem_in_internal,
    )
    public = apply_restore(
        canonical,
        pf_map,
        mode="public",
        overwrite=args.overwrite,
        restore_pf_poem_in_internal=False,
    )

    print("internal PF after:", count_pf_missing(internal))
    print("public PF after  :", count_pf_missing(public))

    out_internal = Path(args.out_internal_dir)
    out_public = Path(args.out_public_dir)
    out_internal.mkdir(parents=True, exist_ok=True)
    out_public.mkdir(parents=True, exist_ok=True)

    write_parquet_splits(internal, out_internal)
    write_parquet_splits(public, out_public)

    print("wrote internal parquet to:", (out_internal / "data").as_posix())
    print("wrote public parquet to  :", (out_public / "data").as_posix())


if __name__ == "__main__":
    main()
