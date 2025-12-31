"""
restore_pf_interpretations.py

restore Poetry Foundation interpretations in the canonical dataset by copying them
from the stale dataset, while keeping Poetry Foundation poem text masked.

defaults assume you renamed:
- canonical: haining/structured_poem_interpretation_corpus
- stale:     haining/structured_poem_interpretation_corpus_stale
"""

from __future__ import annotations

import argparse
import hashlib
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from datasets import DatasetDict, load_dataset
from huggingface_hub import HfApi, hf_hub_download, login
from tqdm.auto import tqdm


def norm_text(s: Optional[str]) -> str:
    if s is None:
        return ""
    s = str(s)
    s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("ascii")
    s = s.lower().strip()
    s = re.sub(r"\s+", " ", s)
    return s


def make_key(author: Optional[str], title: Optional[str], source: Optional[str]) -> str:
    # stable join key for the same record across repos
    base = "||".join([norm_text(author), norm_text(title), norm_text(source)])
    return hashlib.md5(base.encode("utf-8")).hexdigest()


def load_dataset_safely(repo_id: str) -> DatasetDict:
    # assumes metadata loads cleanly (your canonical now does)
    # if you ever need a parquet fallback again, we can add it.
    return load_dataset(repo_id)


def build_pf_interp_map(stale: DatasetDict) -> Tuple[Dict[str, str], int]:
    """
    build mapping key -> interpretation for PF rows.
    return (map, conflict_count) where conflicts mean same key has different interpretations.
    """
    mapping: Dict[str, str] = {}
    conflicts = 0

    for split in stale.keys():
        for row in tqdm(stale[split], desc=f"index stale:{split}"):
            if row.get("source") != "poetry_foundation":
                continue
            interp = row.get("interpretation")
            if interp is None or str(interp).strip() == "":
                continue
            k = make_key(row.get("author"), row.get("title"), row.get("source"))
            if k in mapping and mapping[k] != interp:
                conflicts += 1
                # keep the longer interpretation
                if len(str(interp)) > len(str(mapping[k])):
                    mapping[k] = str(interp)
            else:
                mapping[k] = str(interp)

    return mapping, conflicts


def count_pf_missing_interpretation(dsd: DatasetDict) -> Tuple[int, int]:
    total = 0
    missing = 0
    for split in dsd.keys():
        for row in dsd[split]:
            if row.get("source") != "poetry_foundation":
                continue
            total += 1
            interp = row.get("interpretation")
            if interp is None or str(interp).strip() == "":
                missing += 1
    return total, missing


def fill_canonical(
    canonical: DatasetDict, pf_map: Dict[str, str], *, overwrite: bool = False
) -> DatasetDict:
    """
    fill PF interpretations in canonical from pf_map.
    always mask PF poem text (poem=None).
    overwrite=False will only fill when canonical interpretation is missing/empty.
    """

    def fill_batch(batch):
        authors = batch["author"]
        titles = batch["title"]
        sources = batch["source"]
        poems = batch["poem"]
        interps = batch["interpretation"]

        for i in range(len(authors)):
            if sources[i] != "poetry_foundation":
                continue

            # mask poem text only
            poems[i] = None

            need_fill = overwrite or (
                interps[i] is None or str(interps[i]).strip() == ""
            )
            if need_fill:
                k = make_key(authors[i], titles[i], sources[i])
                if k in pf_map:
                    interps[i] = pf_map[k]

        batch["poem"] = poems
        batch["interpretation"] = interps
        return batch

    fixed = DatasetDict()
    for split in canonical.keys():
        fixed[split] = canonical[split].map(fill_batch, batched=True, batch_size=1000)
    return fixed


def patch_hf_readme(repo_id: str) -> None:
    """
    patch the masking sentence in README.md to say we mask poem only, not interpretation.
    if the exact sentence isn't found, append a short corrected paragraph under a new heading.
    """
    readme_path = hf_hub_download(
        repo_id=repo_id, repo_type="dataset", filename="README.md"
    )
    text = Path(readme_path).read_text(encoding="utf-8")

    replaced = False

    # common wrong phrasing variants
    patterns = [
        r"For rows where `source == \"poetry_foundation\"`, the `poem` and `interpretation` fields are set to `null`[^.]*\.",
        r"Rows with `source == \"poetry_foundation\"` have `poem` and `interpretation` set to `null`[^.]*\.",
        r"`poem` and `interpretation` (are|were) set to (null|`null`)[^.]*\.",
    ]

    replacement = (
        'For rows where `source == "poetry_foundation"`, the `poem` field is set to `null` '
        "in this release to respect content licensing. The `interpretation` field (machine-generated) "
        "and all categorical annotations and metadata remain available."
    )

    for pat in patterns:
        new_text, n = re.subn(pat, replacement, text, flags=re.IGNORECASE)
        if n > 0:
            text = new_text
            replaced = True
            break

    if not replaced:
        # append a corrected masking note
        text += "\n\n## Masking policy (Poetry Foundation)\n\n" + replacement + "\n"

    tmp = Path("README.patched.md")
    tmp.write_text(text, encoding="utf-8")

    api = HfApi()
    api.upload_file(
        path_or_fileobj=str(tmp),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
        commit_message="Fix masking policy: mask poem text only; keep interpretations",
    )


def listlike(x) -> List[str]:
    if x is None:
        return []
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (list, tuple)):
        return list(x)
    return []


def write_stats_and_sample(dsd: DatasetDict, out_dir: Path) -> None:
    out_stats = out_dir / "stats"
    out_data = out_dir / "data"
    out_stats.mkdir(parents=True, exist_ok=True)
    out_data.mkdir(parents=True, exist_ok=True)

    # split sizes + masking rates (poem only should be null for PF; interpretation should NOT)
    rows = []
    mask_rows = []
    for split in ["train", "validation", "test"]:
        ds = dsd[split]
        src_ctr = Counter()
        poem_null = Counter()
        interp_null = Counter()
        for row in ds:
            src = row.get("source")
            src_ctr[src] += 1
            poem_null[src] += int(
                row.get("poem") is None or str(row.get("poem")).strip() == ""
            )
            interp_null[src] += int(
                row.get("interpretation") is None
                or str(row.get("interpretation")).strip() == ""
            )

        for src, n in src_ctr.items():
            rows.append({"split": split, "source": src, "rows": int(n)})
            mask_rows.append(
                {
                    "split": split,
                    "source": src,
                    "poem_null_rate": poem_null[src] / n,
                    "interpretation_null_rate": interp_null[src] / n,
                    "rows": int(n),
                }
            )

    split_source_counts = pd.DataFrame(rows)
    masking_rates = pd.DataFrame(mask_rows)
    split_source_counts.to_csv(out_stats / "split_source_counts.csv", index=False)
    masking_rates.to_csv(out_stats / "masking_rates.csv", index=False)

    # text length stats (public domain only)
    def wc(s: Optional[str]) -> int:
        if s is None:
            return 0
        s = str(s).strip()
        return len(s.split()) if s else 0

    len_rows = []
    for split in ["train", "validation", "test"]:
        ds = dsd[split]
        poem_wc = []
        interp_wc = []
        for row in ds:
            if row.get("source") != "public_domain_poetry":
                continue
            poem_wc.append(wc(row.get("poem")))
            interp_wc.append(wc(row.get("interpretation")))

        def q(arr: List[int]) -> Dict[str, float]:
            a = np.array(arr, dtype=float)
            return {
                "mean": float(a.mean()) if len(a) else float("nan"),
                "median": float(np.median(a)) if len(a) else float("nan"),
                "p05": float(np.percentile(a, 5)) if len(a) else float("nan"),
                "p95": float(np.percentile(a, 95)) if len(a) else float("nan"),
                "max": float(a.max()) if len(a) else float("nan"),
            }

        len_rows.append({"split": split, "field": "poem_words", **q(poem_wc)})
        len_rows.append(
            {"split": split, "field": "interpretation_words", **q(interp_wc)}
        )

    text_length = pd.DataFrame(len_rows)
    text_length.to_csv(out_stats / "text_length_summary_public_domain.csv", index=False)

    # label coverage + distributions (avoid pandas list/ndarray pitfalls)
    cov_rows = []
    sent_rows = []
    themes50_ctr = Counter()

    for split in ["train", "validation", "test"]:
        ds = dsd[split]
        n = len(ds)

        emo_empty = 0
        themes_empty = 0
        themes50_empty = 0
        themes50_lens = []
        sent_ctr = Counter()

        for row in ds:
            emotions = listlike(row.get("emotions"))
            themes = listlike(row.get("themes"))
            themes_50 = listlike(row.get("themes_50"))

            emo_empty += int(len(emotions) == 0)
            themes_empty += int(len(themes) == 0)
            themes50_empty += int(len(themes_50) == 0)
            themes50_lens.append(len(themes_50))

            sent_ctr.update([row.get("sentiment")])
            themes50_ctr.update(
                [
                    str(x).strip().lower()
                    for x in themes_50
                    if x is not None and str(x).strip() != ""
                ]
            )

        themes50_lens = np.array(themes50_lens, dtype=int)

        cov_rows.append(
            {
                "split": split,
                "emotions_empty_rate": emo_empty / n,
                "themes_empty_rate": themes_empty / n,
                "themes_50_empty_rate": themes50_empty / n,
                "themes_50_len_median": float(np.median(themes50_lens)),
                "themes_50_len_p95": float(np.percentile(themes50_lens, 95)),
            }
        )

        for lab, cnt in sent_ctr.items():
            sent_rows.append(
                {
                    "split": split,
                    "sentiment": str(lab),
                    "count": int(cnt),
                    "pct": cnt / n,
                }
            )

    coverage = pd.DataFrame(cov_rows)
    sentiment_dist = pd.DataFrame(sent_rows).sort_values(["split", "sentiment"])
    top_themes50 = pd.DataFrame(
        themes50_ctr.most_common(50), columns=["theme_50", "count"]
    )

    coverage.to_csv(out_stats / "label_coverage.csv", index=False)
    sentiment_dist.to_csv(out_stats / "sentiment_distribution.csv", index=False)
    top_themes50.to_csv(out_stats / "themes_50_top50.csv", index=False)

    # sample.csv: 600 rows (200 per split), mask poem only for PF
    sample_cols = [
        "author",
        "title",
        "poem",
        "interpretation",
        "source",
        "emotions",
        "primary_emotion",
        "sentiment",
        "themes",
        "themes_50",
    ]
    parts = []
    for split in ["train", "validation", "test"]:
        ds = dsd[split].select_columns(sample_cols).to_pandas()
        ds["split"] = split

        pd_part = ds[ds["source"] == "public_domain_poetry"].sample(
            n=min(120, (ds["source"] == "public_domain_poetry").sum()),
            random_state=0,
        )
        pf_part = (
            ds[ds["source"] == "poetry_foundation"]
            .sample(
                n=min(80, (ds["source"] == "poetry_foundation").sum()),
                random_state=0,
            )
            .copy()
        )

        # mask poem only
        pf_part["poem"] = None

        parts += [pd_part, pf_part]

    sample_df = pd.concat(parts, ignore_index=True)

    # list columns -> readable strings
    def join_list(x):
        if x is None:
            return ""
        if isinstance(x, np.ndarray):
            x = x.tolist()
        if isinstance(x, (list, tuple)):
            return ";".join(
                [str(v) for v in x if v is not None and str(v).strip() != ""]
            )
        return str(x)

    for c in ["emotions", "themes", "themes_50"]:
        sample_df[c] = sample_df[c].apply(join_list)

    sample_df.to_csv(out_data / "sample.csv", index=False)

    # README-ready markdown
    md = []
    md.append("## Dataset statistics\n")
    md.append("### Split sizes by source\n")
    md.append(split_source_counts.to_markdown(index=False))
    md.append("\n\n### Masking rates (Poetry Foundation)\n")
    md.append(masking_rates.to_markdown(index=False))
    md.append("\n\n### Text length (public-domain only; word counts)\n")
    md.append(text_length.to_markdown(index=False))
    md.append("\n\n### Label coverage\n")
    md.append(coverage.to_markdown(index=False))
    md.append("\n\n### Sentiment distribution\n")
    md.append(sentiment_dist.to_markdown(index=False))
    md.append("\n\n### Top themes_50 (overall)\n")
    md.append(top_themes50.head(20).to_markdown(index=False))
    (out_stats / "readme_stats.md").write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--repo_current", default="haining/structured_poem_interpretation_corpus"
    )
    ap.add_argument(
        "--repo_stale", default="haining/structured_poem_interpretation_corpus_stale"
    )
    ap.add_argument(
        "--overwrite", action="store_true", help="overwrite existing PF interpretations"
    )
    ap.add_argument(
        "--push",
        action="store_true",
        help="push the fixed dataset back to repo_current",
    )
    ap.add_argument(
        "--patch_readme",
        action="store_true",
        help="patch README.md masking paragraph on hub",
    )
    ap.add_argument(
        "--write_artifacts",
        action="store_true",
        help="write stats/ and data/sample.csv locally",
    )
    ap.add_argument("--out_dir", default="repo_artifacts")
    args = ap.parse_args()

    login()  # uses HF_TOKEN if available; otherwise interactive

    print("loading canonical:", args.repo_current)
    canonical = load_dataset_safely(args.repo_current)

    print("loading stale:", args.repo_stale)
    stale = load_dataset_safely(args.repo_stale)

    pf_total_before, pf_missing_before = count_pf_missing_interpretation(canonical)
    print(
        f"canonical PF rows={pf_total_before:,} missing interpretation={pf_missing_before:,}"
    )

    pf_map, conflicts = build_pf_interp_map(stale)
    print(f"stale PF interpretation map size={len(pf_map):,} conflicts={conflicts:,}")

    fixed = fill_canonical(canonical, pf_map, overwrite=args.overwrite)

    pf_total_after, pf_missing_after = count_pf_missing_interpretation(fixed)
    print(
        f"fixed PF rows={pf_total_after:,} missing interpretation={pf_missing_after:,}"
    )
    print(f"filled interpretations: {pf_missing_before - pf_missing_after:,}")

    # quick spot-check
    for i in range(3):
        row = fixed["train"][i]
        if row.get("source") == "poetry_foundation":
            print("spot-check PF:", row.get("author"), "-", row.get("title"))
            print("poem is None?", row.get("poem") is None)
            print(
                "interpretation len:",
                (
                    0
                    if row.get("interpretation") is None
                    else len(str(row.get("interpretation")))
                ),
            )
            break

    if args.push:
        print("pushing fixed dataset to hub:", args.repo_current)
        fixed.push_to_hub(
            args.repo_current,
            commit_message="Restore Poetry Foundation interpretations; mask poem text only",
        )

    if args.patch_readme:
        print("patching README on hub:", args.repo_current)
        patch_hf_readme(args.repo_current)

    if args.write_artifacts:
        out_dir = Path(args.out_dir)
        print("writing artifacts to:", out_dir)
        write_stats_and_sample(fixed, out_dir)
        print("wrote:", out_dir / "stats/readme_stats.md")
        print("wrote:", out_dir / "data/sample.csv")


if __name__ == "__main__":
    main()
