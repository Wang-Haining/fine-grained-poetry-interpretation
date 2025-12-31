import os

from datasets import load_dataset
from huggingface_hub import login

login(token=os.environ["HF_TOKEN"])

repo_id = "haining/structured_poem_interpretation_corpus"

data_files = {
    "train": "public_masked/data/train-*.parquet",
    "validation": "public_masked/data/validation-*.parquet",
    "test": "public_masked/data/test-*.parquet",
}

dsd = load_dataset("parquet", data_files=data_files)


# sanity checks: PF poem masked, PF interpretation present
def is_missing(x):
    return x is None or str(x).strip() == ""


pf_total = pf_poem_missing = pf_interp_missing = 0
for split in ["train", "validation", "test"]:
    for r in dsd[split]:
        if r["source"] != "poetry_foundation":
            continue
        pf_total += 1
        pf_poem_missing += int(is_missing(r["poem"]))
        pf_interp_missing += int(is_missing(r["interpretation"]))

print(
    {
        "pf_total": pf_total,
        "pf_poem_missing": pf_poem_missing,
        "pf_interpretation_missing": pf_interp_missing,
    }
)
assert pf_poem_missing == pf_total, "public version should mask ALL PF poems"
assert pf_interp_missing == 0, "public version should keep ALL PF interpretations"

# push to hub (updates parquet + schema metadata)
dsd.push_to_hub(
    repo_id,
    commit_message="Restore Poetry Foundation interpretations; mask Poetry Foundation poem text only",
)
