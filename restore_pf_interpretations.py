"""
restore_pf_interpretations_and_make_two_local_versions.py

goal
- keep the now-canonical dataset as the base (themes_50, etc)
- restore poetry foundation interpretations (and optionally poem text for internal use)
  from the stale dataset
- write two local versions:
  1) internal_full: poetry_foundation poem + interpretation retained
  2) public_masked: poetry_foundation poem masked (null), interpretation retained

assumes you renamed:
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
from datasets import DatasetDict, load_dataset
from huggingface_hub import login, snapshot_download
from tqdm.auto import tqdm


def try_login() -> None:
    token = os.environ.get("HF_TOKEN")
    if token:
        login(token=token)
    else:
        # for public datasets, auth is not required
        # for private datasets, this will prompt interactively
        try:
            login()
        except Exception:
            pass


def load_dataset_with_fallback(repo_id: str) -> DatasetDict:
    # try normal loading first
    try:
        return load_dataset(repo_id)
    except Exception:
        # fallback: load the parquet files directly from a snapshot
        snap_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
        data_dir = Path(snap_dir) / "data"
        data_files: Dict[str, str] = {}
        for split in ["train", "validation", "test"]:
            files = sorted(data_dir.glob(f"{split}-*.parquet"))
            if files:
                data_files[split] = str(files[0])
        if not data_files:
            raise RuntimeError(f"could not find parquet files in {repo_id} snapshot")
        return load_dataset("parquet", data_files=data_files)


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


def build_pf_map(
    stale: DatasetDict,
) -> Tuple[Dict[str, Dict[str, Optional[str]]], Dict[str, int]]:
    # map key -> {"poem": ..., "interpretation": ...} from poetry_foundation rows
    mapping: Dict[str, Dict[str, Optional[str]]] = {}
    stats = {
        "pf_rows_seen": 0,
        "pf_rows_with_poem": 0,
        "pf_rows_with_interpretation": 0,
        "key_conflicts": 0,
        "key_duplicates": 0,
    }

    for split in stale.keys():
        for row in tqdm(stale[split], desc=f"index stale:{split}"):
            if row.get("source") != "poetry_foundation":
                continue

            stats["pf_rows_seen"] += 1
            poem = row.get("poem")
            interp = row.get("interpretation")

            if not is_missing_text(poem):
                stats["pf_rows_with_poem"] += 1
            if not is_missing_text(interp):
                stats["pf_rows_with_interpretation"] += 1

            k = make_key(row.get("author"), row.get("title"), row.get("source"))
            rec = {"poem": None, "interpretation": None}
            if not is_missing_text(poem):
                rec["poem"] = str(poem)
            if not is_missing_text(interp):
                rec["interpretation"] = str(interp)

            if k not in mapping:
                mapping[k] = rec
                continue

            stats["key_duplicates"] += 1
            existing = mapping[k]

            # conflict bookkeeping (different non-empty values)
            if (
                existing.get("poem")
                and rec.get("poem")
                and existing["poem"] != rec["poem"]
            ) or (
                existing.get("interpretation")
                and rec.get("interpretation")
                and existing["interpretation"] != rec["interpretation"]
            ):
                stats["key_conflicts"] += 1

            # keep the longer poem/interpretation when both exist
            if rec.get("poem") and (
                not existing.get("poem")
                or len(rec["poem"]) > len(existing["poem"] or "")
            ):
                existing["poem"] = rec["poem"]
            if rec.get("interpretation") and (
                not existing.get("interpretation")
                or len(rec["interpretation"]) > len(existing["interpretation"] or "")
            ):
                existing["interpretation"] = rec["interpretation"]

            mapping[k] = existing

    return mapping, stats


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
    mode: str,
    overwrite: bool,
    restore_pf_poem_in_internal: bool,
) -> DatasetDict:
    # mode: "internal" or "public"
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

            # always restore interpretation if missing (or overwrite)
            if rec and rec.get("interpretation"):
                if overwrite or is_missing_text(interps[i]):
                    interps[i] = rec["interpretation"]

            if mode == "public":
                # mask poem only for public version
                poems[i] = None
            else:
                # internal: keep poem if you choose to restore it from stale
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
        # datasets supports to_parquet in recent versions
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
    ap.add_argument(
        "--restore_pf_poem_in_internal",
        action="store_true",
        help="also restore PF poem text for internal_full",
    )
    ap.add_argument(
        "--save_to_disk",
        action="store_true",
        help="also save arrow format via save_to_disk()",
    )

    args = ap.parse_args()

    try_login()

    print(f"loading canonical: {args.repo_canonical}")
    canonical = load_dataset_with_fallback(args.repo_canonical)

    print(f"loading stale: {args.repo_stale}")
    stale = load_dataset_with_fallback(args.repo_stale)

    before = count_pf_missing(canonical)
    print("canonical pf status before:", before)

    pf_map, pf_stats = build_pf_map(stale)
    print("stale pf map stats:", pf_stats)
    print("pf map keys:", len(pf_map))

    # internal version: restore interpretations; optionally restore PF poem text
    internal = apply_restore(
        canonical,
        pf_map,
        mode="internal",
        overwrite=args.overwrite,
        restore_pf_poem_in_internal=args.restore_pf_poem_in_internal,
    )

    # public version: restore interpretations; mask PF poem text only
    public = apply_restore(
        canonical,
        pf_map,
        mode="public",
        overwrite=args.overwrite,
        restore_pf_poem_in_internal=False,
    )

    after_internal = count_pf_missing(internal)
    after_public = count_pf_missing(public)

    print("internal pf status after:", after_internal)
    print("public pf status after  :", after_public)

    # public sanity: PF poem should be fully masked
    if (
        after_public["pf_total"] > 0
        and after_public["pf_poem_missing"] != after_public["pf_total"]
    ):
        print("warning: public_masked does not fully mask PF poem text (unexpected)")

    # write outputs
    out_internal = Path(args.out_internal_dir)
    out_public = Path(args.out_public_dir)
    out_internal.mkdir(parents=True, exist_ok=True)
    out_public.mkdir(parents=True, exist_ok=True)

    write_parquet_splits(internal, out_internal)
    write_parquet_splits(public, out_public)

    if args.save_to_disk:
        internal.save_to_disk(str(out_internal / "arrow"))
        public.save_to_disk(str(out_public / "arrow"))

    print("wrote internal parquet to:", (out_internal / "data").as_posix())
    print("wrote public parquet to  :", (out_public / "data").as_posix())
    if args.save_to_disk:
        print(
            "also wrote arrow dirs to :",
            (out_internal / "arrow").as_posix(),
            "and",
            (out_public / "arrow").as_posix(),
        )


if __name__ == "__main__":
    main()
