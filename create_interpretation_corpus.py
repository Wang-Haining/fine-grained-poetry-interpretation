from datasets import Dataset, DatasetDict, concatenate_datasets
import pandas as pd
from nltk.tokenize import casual_tokenize
import numpy as np
from sklearn.model_selection import train_test_split

POETRY_INTERPRETATION_CORPUS = 'interpretation/poetry_interpretation_corpus_v001.hf'

# load datasets
pf_ds = pd.read_parquet('interpretation/poetry_foundation.parquet')  # 13,854
pdp_ds = pd.read_parquet('interpretation/public_domain_poetry.parquet')  # 38,499
pf_ds = Dataset.from_pandas(pf_ds)
pdp_ds = Dataset.from_pandas(pdp_ds)

# add source identifier (for stats)
pf_ds = pf_ds.map(lambda x: {**x, 'source': 'poetry_foundation'})
pdp_ds = pdp_ds.map(lambda x: {**x, 'source': 'public_domain_poetry'})

# combine
dataset = concatenate_datasets([pf_ds, pdp_ds])

# remove duplicates, based on author and title
dataset = dataset.to_pandas().drop_duplicates(subset=['author', 'title'])
print(f'The raw combined dataset has {dataset.shape[0]} entries before duplication.')

# shuffle and split using train_test_split
train_data, val_test_data = train_test_split(dataset, test_size=0.1, random_state=42)
valid_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)

# convert back to Dataset
train_data = Dataset.from_pandas(train_data)
valid_data = Dataset.from_pandas(valid_data)
test_data = Dataset.from_pandas(test_data)

# create a DatasetDict
dataset_dict = DatasetDict({
    'train': train_data,
    'validation': valid_data,
    'test': test_data
})

# save
dataset_dict.save_to_disk(POETRY_INTERPRETATION_CORPUS)

# corpus statistics
corpus_stats = {
    'number_of_entries': dataset.groupby('source').size().to_dict(),
    'number_of_authors': dataset['author'].nunique(),
    'percentile_length_distribution': {}
}

# length distributions
poem_lengths = dataset.apply(lambda x: len(casual_tokenize(x['poem'])), axis=1)
interpretation_lengths = dataset.apply(lambda x: len(casual_tokenize(x['interpretation'])), axis=1)

# calculate percentiles
poem_percentiles = np.percentile(poem_lengths, [25, 50, 75])
interpretation_percentiles = np.percentile(interpretation_lengths, [25, 50, 75])

corpus_stats['percentile_length_distribution']['poem'] = poem_percentiles.tolist()
corpus_stats['percentile_length_distribution']['interpretation'] = interpretation_percentiles.tolist()

# print statistics
print('Poetry Interpretation Corpus Statistics:')
for key, value in corpus_stats.items():
    print(f"{key}: {value}")
