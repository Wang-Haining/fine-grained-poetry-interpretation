"""
restore_pf_from_stale_revision.py

creates two local datasets from the now-canonical corpus by restoring Poetry Foundation
interpretations (and optionally PF poem text for internal use) from an older revision
of the stale dataset.

outputs:
- internal_full/data/{train,validation,test}-00000-of-00001.parquet
- public_masked/data/{train,validation,test}-00000-of-00001.parquet

defaults assume:
- canonical: haining/structured_poem_interpretation_corpus
- stale:     haining/structured_poem_interpretation_corpus_stale
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
import pyarrow.parquet as pq
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi, hf_hub_download, login, snapshot_download
from tqdm.auto import tqdm


def try_login() -> None:
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        # for private datasets, this prompts
        login()


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


def load_dataset_with_fallback(
    repo_id: str, revision: Optional[str] = None
) -> DatasetDict:
    # prefer standard loading if possible
    try:
        return (
            load_dataset(repo_id, revision=revision)
            if revision
            else load_dataset(repo_id)
        )
    except Exception:
        # fallback: load parquets from a snapshot
        snap_dir = snapshot_download(
            repo_id=repo_id, repo_type="dataset", revision=revision
        )
        data_dir = Path(snap_dir) / "data"
        data_files: Dict[str, str] = {}
        for split in ["train", "validation", "test"]:
            files = sorted(data_dir.glob(f"{split}-*.parquet"))
            if files:
                data_files[split] = str(files[0])
        if not data_files:
            raise RuntimeError(
                f"no parquet files found in snapshot for {repo_id}@{revision}"
            )
        return load_dataset("parquet", data_files=data_files)


def probe_pf_counts_from_parquet(parquet_path: str) -> Tuple[int, int, int]:
    """
    return (pf_total, pf_poem_nonempty, pf_interp_nonempty) for a single parquet file.
    """
    table = pq.read_table(parquet_path, columns=["source", "poem", "interpretation"])
    src = table["source"].to_pylist()
    poem = table["poem"].to_pylist()
    interp = table["interpretation"].to_pylist()

    pf_total = 0
    pf_poem_nonempty = 0
    pf_interp_nonempty = 0
    for s, p, it in zip(src, poem, interp):
        if s != "poetry_foundation":
            continue
        pf_total += 1
        if not is_missing_text(p):
            pf_poem_nonempty += 1
        if not is_missing_text(it):
            pf_interp_nonempty += 1
    return pf_total, pf_poem_nonempty, pf_interp_nonempty


def find_best_stale_revision(
    repo_id: str,
    *,
    prefer_poem: bool,
    max_commits: int,
    probe_split: str,
) -> Optional[str]:
    """
    scans stale repo commits newest->oldest, probes a small parquet file, and returns
    the first revision with PF interpretation available (and PF poem if prefer_poem).
    """
    api = HfApi()
    commits = api.list_repo_commits(repo_id=repo_id, repo_type="dataset")
    commits = commits[:max_commits]

    for c in commits:
        rev = (
            getattr(c, "commit_id", None)
            or getattr(c, "oid", None)
            or getattr(c, "sha", None)
        )
        if not rev:
            continue

        # download just one split parquet for probing (smallest is validation/test usually)
        # we try preferred split first; fallback to any available parquet in that revision.
        try:
            snap = snapshot_download(
                repo_id=repo_id,
                repo_type="dataset",
                revision=rev,
                allow_patterns=[f"data/{probe_split}-*.parquet"],
            )
            data_dir = Path(snap) / "data"
            files = sorted(data_dir.glob(f"{probe_split}-*.parquet"))
            if not files:
                continue
            probe_file = str(files[0])
        except Exception:
            continue

        pf_total, pf_poem_nonempty, pf_interp_nonempty = probe_pf_counts_from_parquet(
            probe_file
        )

        # we care about PF interpretation existing
        if pf_interp_nonempty <= 0:
            continue
        if prefer_poem and pf_poem_nonempty <= 0:
            continue

        print(
            f"picked stale revision {rev} based on {probe_split} probe: "
            f"pf_total={pf_total}, pf_poem_nonempty={pf_poem_nonempty}, pf_interp_nonempty={pf_interp_nonempty}"
        )
        return rev

    return None


def build_pf_map(source: DatasetDict) -> Dict[str, Dict[str, Optional[str]]]:
    """
    key -> {"poem": str|None, "interpretation": str|None} for PF rows only.
    keeps the longer value on duplicates.
    """
    mapping: Dict[str, Dict[str, Optional[str]]] = {}

    for split in source.keys():
        for row in tqdm(source[split], desc=f"index source:{split}"):
            if row.get("source") != "poetry_foundation":
                continue

            k = make_key(row.get("author"), row.get("title"), row.get("source"))

            poem = None if is_missing_text(row.get("poem")) else str(row.get("poem"))
            interp = (
                None
                if is_missing_text(row.get("interpretation"))
                else str(row.get("interpretation"))
            )

            if k not in mapping:
                mapping[k] = {"poem": poem, "interpretation": interp}
                continue

            # keep longer non-empty
            if poem and (
                not mapping[k].get("poem") or len(poem) > len(mapping[k]["poem"] or "")
            ):
                mapping[k]["poem"] = poem
            if interp and (
                not mapping[k].get("interpretation")
                or len(interp) > len(mapping[k]["interpretation"] or "")
            ):
                mapping[k]["interpretation"] = interp

    return mapping


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


def apply_restore(
    canonical: DatasetDict,
    pf_map: Dict[str, Dict[str, Optional[str]]],
    *,
    mode: str,  # "internal" | "public"
    overwrite: bool,
    restore_pf_poem_in_internal: bool,
) -> DatasetDict:
    if mode not in {"internal", "public"}:
        raise ValueError("mode must be 'internal' or 'public'")

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

            # interpretation: always try to restore
            if rec and rec.get("interpretation"):
                if overwrite or is_missing_text(interps[i]):
                    interps[i] = rec["interpretation"]

            # poem: public masks; internal optionally restores
            if mode == "public":
                poems[i] = None
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
        if split not in dsd:
            continue
        path = out_data / f"{split}-00000-of-00001.parquet"
        dsd[split].to_parquet(str(path))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo_canonical", default="haining/structured_poem_interpretation_corpus"
    )
    ap.add_argument(
        "--repo_stale", default="haining/structured_poem_interpretation_corpus_stale"
    )

    ap.add_argument("--out_internal_dir", default="internal_full")
    ap.add_argument("--out_public_dir", default="public_masked")

    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--restore_pf_poem_in_internal", action="store_true")

    ap.add_argument(
        "--stale_revision", default=None, help="optional: pin a known good revision sha"
    )
    ap.add_argument(
        "--auto_find_revision",
        action="store_true",
        help="auto-search stale history for PF interpretation",
    )
    ap.add_argument(
        "--prefer_poem",
        action="store_true",
        help="when auto-finding, require PF poem to exist too",
    )
    ap.add_argument("--max_commits", type=int, default=50)
    ap.add_argument(
        "--probe_split", default="validation", choices=["train", "validation", "test"]
    )

    args = ap.parse_args()

    try_login()

    print(f"loading canonical: {args.repo_canonical}")
    canonical = load_dataset_with_fallback(args.repo_canonical)

    print("canonical PF status before:", count_pf_missing(canonical))

    stale_rev = args.stale_revision
    if not stale_rev and args.auto_find_revision:
        print(
            f"auto-finding a stale revision in {args.repo_stale} (max_commits={args.max_commits}, probe_split={args.probe_split})"
        )
        stale_rev = find_best_stale_revision(
            args.repo_stale,
            prefer_poem=args.prefer_poem,
            max_commits=args.max_commits,
            probe_split=args.probe_split,
        )

    if stale_rev:
        print(f"loading stale (revision={stale_rev}): {args.repo_stale}")
    else:
        print(f"loading stale (HEAD): {args.repo_stale}")

    stale = load_dataset_with_fallback(args.repo_stale, revision=stale_rev)

    print("stale PF status (source):", count_pf_missing(stale))

    pf_map = build_pf_map(stale)
    print("pf map keys:", len(pf_map))

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

    print("internal PF status after:", count_pf_missing(internal))
    print("public PF status after  :", count_pf_missing(public))

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
