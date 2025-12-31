# Structured Poem Interpretation Corpus

A large-scale corpus of English poems paired with structured, machine-generated interpretations and categorical tags for computational literary studies and NLP.

* **Total rows:** 51,356
* **Splits:** train 46,220 | validation 2,568 | test 2,568
* **Sources:** 37,554 public-domain poems and 13,802 Poetry Foundation poems (text masked)

This repository also includes:

* **Sample data:** `data/sample.csv` (easy-to-read subset)
* **Basic statistics:** `stats/readme_stats.md` (split sizes, masking rates, label summaries)

## What is in the dataset

Each row corresponds to a poem record. For public-domain items, we provide the full poem text and a structured interpretation. For Poetry Foundation items, we provide metadata and categorical tags, but the poem text and interpretation are masked (see the masking policy below).

### Source corpora

1. **Public Domain Poetry**

* Source: `DanFosing/public-domain-poetry` (Hugging Face)
* Rows in this release: **37,554**

2. **Poetry Foundation**

* Source: Poetry Foundation poem metadata (via a public dataset mirror)
* Rows in this release: **13,802**

### Masking policy (Poetry Foundation)

For rows where `source == "poetry_foundation"`, the `poem` and `interpretation` fields are set to `null` in this release to respect content licensing. All categorical annotations and metadata remain available.

Users who have independent access to the Poetry Foundation text can recover poem content by using `author` and `title` to locate the poem on poetryfoundation.org.

## Fields

| Field             | Type           | Description                                                                          |
| ----------------- | -------------- | ------------------------------------------------------------------------------------ |
| `author`          | string         | Poet name.                                                                           |
| `title`           | string         | Poem title.                                                                          |
| `poem`            | string or null | Full poem text (null for Poetry Foundation rows).                                    |
| `interpretation`  | string or null | Machine-generated interpretation (null for Poetry Foundation rows).                  |
| `source`          | string         | `public_domain_poetry` or `poetry_foundation`.                                       |
| `emotions`        | list[string]   | 1–3 labels from {anger, anticipation, disgust, fear, joy, sadness, surprise, trust}. |
| `primary_emotion` | string         | The first item of `emotions`.                                                        |
| `sentiment`       | string         | One of {positive, neutral, negative}.                                                |
| `themes`          | list[string]   | Open-vocabulary themes (0–5 concise tags, 1–3 words each).                           |
| `themes_50`       | list[string]   | Themes chosen from a fixed 50-item lexicon (typically up to 5).                      |

### Fixed 50-theme lexicon

`themes_50` uses the following fixed set (lowercased):

```text
nature, body, death, love, existential, identity, self, beauty, america,
loss, animals, history, memories, family, writing, ancestry, thought,
landscapes, war, time, religion, grief, violence, aging, childhood, desire,
night, mothers, language, birds, social justice, music, flowers, politics,
hope, heartache, fathers, gender, environment, spirituality, loneliness,
oceans, dreams, survival, cities, earth, despair, anxiety, weather, illness,
home
```

## How the annotations were produced (high level)

* **Interpretations:** generated offline via structured prompting.
* **Categorical tags:** produced with a guardrailed LLM pipeline that enforces a strict JSON schema, followed by normalization (lowercasing, deduplication, and length limits).

The goal is to support both open-ended analysis (`themes`) and controlled-category evaluation (`emotions`, `sentiment`, `themes_50`).

## Quick dataset statistics

The following files are generated for reviewer-friendly inspection:

* `stats/readme_stats.md`: split sizes by source, masking rates, public-domain text length summaries, and label distribution tables.
* `data/sample.csv`: a small, human-readable subset of rows from all splits. Poetry Foundation rows have `poem` and `interpretation` masked.

## Usage

### Load from Hugging Face

```python
from datasets import load_dataset

dsd = load_dataset("haining/structured_poem_interpretation_corpus")
train = dsd["train"]

# public-domain rows with full text
pd_train = train.filter(lambda r: r["source"] == "public_domain_poetry")

# Poetry Foundation rows (text masked, annotations available)
pf_train = train.filter(lambda r: r["source"] == "poetry_foundation")
```

[//]: # (## Citation)

[//]: # ()
[//]: # (TBD)

## License

* Public-domain poem text is included where permitted.
* Poetry Foundation text is masked in this release.
* Annotations and derived metadata are released under the MIT license.
