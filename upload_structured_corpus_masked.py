from pathlib import Path

import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder

# local v2 artifacts (written by merge_poem_attrs.py)
DATA_DIR = (
    "poem_attrs/merged"  # expects train.parquet / validation.parquet / test.parquet
)

# target hf dataset repo
REPO_ID = "haining/structured_poem_interpretation_corpus_v2"


def build_masked_dataset(data_dir: str) -> DatasetDict:
    dsd = DatasetDict()
    root = Path(data_dir)

    for split in ("train", "validation", "test"):
        p = root / f"{split}.parquet"
        if not p.exists():
            continue

        df = pd.read_parquet(p)

        # mask poetry foundation text in-place
        if "source" in df.columns:
            is_pf = df["source"].eq("poetry_foundation")
            for col in ("poem", "interpretation"):
                if col in df.columns:
                    df.loc[is_pf, col] = None

        dsd[split] = Dataset.from_pandas(
            df.reset_index(drop=True),
            preserve_index=False,
        )

    return dsd


def main() -> None:
    print("building masked DatasetDict…")
    dsd = build_masked_dataset(DATA_DIR)
    print({k: len(v) for k, v in dsd.items()})

    token = HfFolder.get_token()
    if not token:
        raise RuntimeError(
            "no HF token found. run `huggingface-cli login` or set HUGGINGFACE_HUB_TOKEN"
        )

    api = HfApi(token=token)
    api.create_repo(
        repo_id=REPO_ID,
        repo_type="dataset",
        private=False,
        exist_ok=True,
    )
    print(f"repo ready: https://huggingface.co/datasets/{REPO_ID}")

    print("pushing masked dataset to hub…")
    dsd.push_to_hub(
        repo_id=REPO_ID,
        token=token,
        commit_message="public v2: add emotions/sentiment/themes; mask PF poem/interpretation with null",
    )

    CARD = """---
pretty_name: Structured Poem Interpretation Corpus (v2, PF text masked)
license: cc-by-4.0
language:
- en
tags:
- poetry
- literary-analysis
- nlp
- annotations
- emotions
- sentiment
- themes
task_categories:
- text-classification
- text-generation
---

**Masking policy:** For rows with `source == "poetry_foundation"`, the `poem` and
`interpretation` fields are set to **null** to respect content licensing. Public-domain
entries (`source == "public_domain_poetry"`) include full text. All categorical annotations
(`emotions`, `primary_emotion`, `sentiment`, `themes`, `themes_50`) and metadata remain available.
"""
    api.upload_file(
        path_or_fileobj=CARD.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print(f"done. view at https://huggingface.co/datasets/{REPO_ID}")


if __name__ == "__main__":
    main()
