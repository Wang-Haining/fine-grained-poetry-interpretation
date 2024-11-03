from datasets import load_from_disk, concatenate_datasets, Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from nltk.tokenize import casual_tokenize
import numpy as np

POETRY_INTERPRETATION_CORPUS = 'interpretation/poetry_interpretation_corpus_v001.hf'

# load datasets
pf_ds = load_from_disk('interpretation/poetry_foundation.parquet')
pdp_ds = load_from_disk('public_domain_poetry.parquet')

# add source identifier (for stats)
pf_ds = pf_ds.map(lambda x: {**x, 'source': 'poetry_foundation'})
pdp_ds = pdp_ds.map(lambda x: {**x, 'source': 'public_domain_poetry'})

# combine
dataset = concatenate_datasets([pf_ds, pdp_ds])

# remove duplicates, based on author and title
dataset = dataset.to_pandas().drop_duplicates(subset=['author', 'title'])
dataset = Dataset.from_pandas(dataset)
print(f'The raw combined dataset has {dataset.shape[0]} entries before duplication.')

# shuffle and split
train_data, val_test_data = train_test_split(dataset, test_size=0.1, random_state=42)
valid_data, test_data = train_test_split(val_test_data, test_size=0.5, random_state=42)

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
    'number_of_entries': dataset.to_pandas().groupby('source').size().to_dict(),
    'number_of_authors': dataset.to_pandas()['author'].nunique(),
    'percentile_length_distribution': {}
}

# length distributions
poem_lengths = dataset.map(lambda x: {'poem_length': len(casual_tokenize(x['poem']))})['poem_length']
interpretation_lengths = dataset.map(lambda x: {'interpretation_length': len(casual_tokenize(x['interpretation']))})['interpretation_length']

# calculate percentiles
poem_percentiles = np.percentile(poem_lengths, [25, 50, 75])
interpretation_percentiles = np.percentile(interpretation_lengths, [25, 50, 75])

corpus_stats['percentile_length_distribution']['poem'] = poem_percentiles.tolist()
corpus_stats['percentile_length_distribution']['interpretation'] = interpretation_percentiles.tolist()

# print statistics
print('Poetry Interpretation Corpus Statistics:')
for key, value in corpus_stats.items():
    print(f"{key}: {value}")
