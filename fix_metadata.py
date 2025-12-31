from pathlib import Path

from datasets import Features, Sequence, Value, load_dataset
from huggingface_hub import login, snapshot_download

login()

repo_id = "haining/structured_poem_interpretation_corpus_v2"

# load via parquet (bypasses the current stale metadata)
snap_dir = snapshot_download(repo_id=repo_id, repo_type="dataset")
data_dir = Path(snap_dir) / "data"
data_files = {
    "train": str(next(data_dir.glob("train-*.parquet"))),
    "validation": str(next(data_dir.glob("validation-*.parquet"))),
    "test": str(next(data_dir.glob("test-*.parquet"))),
}

# current v2 schema (includes __index_level_0__)
features_with_index = Features(
    {
        "author": Value("string"),
        "title": Value("string"),
        "poem": Value("string"),
        "interpretation": Value("string"),
        "source": Value("string"),
        "__index_level_0__": Value("int64"),
        "emotions": Sequence(Value("string")),
        "primary_emotion": Value("string"),
        "sentiment": Value("string"),
        "themes": Sequence(Value("string")),
        "themes_50": Sequence(Value("string")),
    }
)

ds = load_dataset("parquet", data_files=data_files, features=features_with_index)

# drop the unwanted column
if "__index_level_0__" in ds["train"].column_names:
    ds = ds.remove_columns("__index_level_0__")

# push back with the cleaned schema (this also fixes hub metadata)
ds.push_to_hub(
    repo_id,
    private=True,
    commit_message="Drop __index_level_0__ and fix dataset schema metadata",
)

# verify
ds_check = load_dataset(repo_id, download_mode="force_redownload")
print(ds_check["train"].column_names)
