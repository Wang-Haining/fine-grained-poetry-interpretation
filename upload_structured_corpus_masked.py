# upload_structured_corpus_masked.py
import pandas as pd
from datasets import Dataset, DatasetDict
from huggingface_hub import HfApi, HfFolder

DATA_PARQUET = "dist_v1/merged_v1.parquet"  # final local parquet
REPO_NAME = "structured_poem_interpretation_corpus"  # HF dataset repo name


def build_masked_dataset(parquet_path: str) -> DatasetDict:
    df = pd.read_parquet(parquet_path)

    # mask PF text in-place
    is_pf = df["source"].eq("poetry_foundation")
    for col in ("poem", "interpretation"):
        if col in df.columns:
            df.loc[is_pf, col] = None  # null out PF content

    # split to DatasetDict without filtering overhead
    splits = [
        s for s in ("train", "validation", "test") if s in set(df["split"].unique())
    ]
    dsd = DatasetDict(
        {
            s: Dataset.from_pandas(
                df[df["split"].eq(s)].reset_index(drop=True), preserve_index=False
            )
            for s in splits
        }
    )
    return dsd


def main():
    print("building masked DatasetDict…")
    dsd = build_masked_dataset(DATA_PARQUET)
    print({k: len(v) for k, v in dsd.items()})

    token = HfFolder.get_token()
    if not token:
        raise RuntimeError(
            "no HF token found. run `huggingface-cli login` or set HUGGINGFACE_HUB_TOKEN"
        )
    api = HfApi(token=token)
    user = api.whoami(token=token)
    username = user["name"]
    repo_id = f"{username}/{REPO_NAME}"
    print(f"authenticated as {username}")

    # create **public** dataset repo
    api.create_repo(repo_id=repo_id, repo_type="dataset", private=False, exist_ok=True)
    print(f"repo ready: https://huggingface.co/datasets/{repo_id}")

    print("pushing masked dataset to hub…")
    dsd.push_to_hub(
        repo_id=repo_id,
        token=token,
        commit_message="public v1: mask PF poem/interpretation with null",
    )

    # dataset card notes masking-by-nulls
    CARD = """---
pretty_name: Structured Poem Interpretation Corpus (v1, PF text masked)
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
        repo_id=repo_id,
        repo_type="dataset",
    )
    print(f"done. view at https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
