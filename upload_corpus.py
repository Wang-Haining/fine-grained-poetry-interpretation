from datasets import load_from_disk
from huggingface_hub import HfApi, HfFolder
import os

# set the path to your dataset and repository name
POEM_INTERPRETATION_CORPUS = 'interpretation/poem_interpretation_corpus_v001.hf'
repo_name = "structured_poem_interpretation_corpus"  # Name for the dataset repository on Hugging Face

# load the dataset
print("Loading dataset from disk...")
ds = load_from_disk(POEM_INTERPRETATION_CORPUS)

# authenticate with Hugging Face
api = HfApi()
user = api.whoami()
username = user["name"]
print(f"Authenticated as {username}.")

# create a private repository on Hugging Face
repo_url = api.create_repo(
    repo_id=f"{username}/{repo_name}",  # Use repo_id instead of name
    token=HfFolder.get_token(),
    repo_type="dataset",
    private=True,  # make the repository private
    exist_ok=True,  # Do not raise an error if the repo already exists
)
print(f"Repository created at {repo_url}")

# save the dataset to Hugging Face
print("Pushing dataset to Hugging Face...")
ds.push_to_hub(repo_id=f"{username}/{repo_name}")

print(f"Dataset successfully uploaded to {repo_url}")
