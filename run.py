"""
This script generates structured interpretations of poems by leveraging the OpenAI API.
It supports two data sources:
- 'poets_org' - a dataset containing poems from poets.org in Parquet format.
- 'public_domain_poetry' - a dataset containing public domain poems in JSON format.

The script fetches a set number of poems, formats them into prompts, and sends the
prompts to OpenAI's GPT model for analysis. The structured interpretation for each poem
is saved along with the poem's original details (author, title, poem text) in Parquet
format.

Make sure OpenAI API key has been set up to work properly.
```bash
# for example, add the below line to ~/.bashrc
export OPENAI_API_KEY="[your_openai_api_key]"
```
"""

__author__ = "hw56@indiana.edu"
__version__ = "0.0.1"
__license__ = "0BSD"

import os
import pandas as pd
from openai import OpenAI


OPENAI_API_PROJECT_KEY_ = 'OPENAI_API_KEY_POETRY'
# MODELS = ["gpt-4o-2024-05-13", "gpt-3.5-turbo-0125"]
MODEL = "gpt-4o-2024-05-13"
NUM_EXAMPLES = 50
SYSTEM_PROMPT = "You are an assistant that strictly follows user instructions. Provide only the analysis and content requested by the user without any greetings, closing remarks, or unnecessary additions. Do not include any extra text beyond what is required."

PROMPT_TEMPLATE = """
Analyze the poem "{title}" by {author}. Please provide a structured interpretation following these steps:

1. Summary: Give a brief overview of the poem's main themes and ideas. Summarize the key emotions or messages conveyed by the poet.

2. Stanza-by-Stanza Breakdown:
For each stanza:
- Summarize the content.
- Discuss any literary devices used (such as metaphor, simile, personification, etc.).
- Explain how the stanza contributes to the overall theme of the poem.

3. Structure and Form:
Discuss the poem's structure (e.g., number of stanzas, rhyme scheme, meter) and how the form contributes to the poem's meaning or emotional effect.

4. Imagery and Symbolism:
Analyze the imagery, symbols, and figurative language. Explain how these elements enhance the meaning of the poem.

5. Tone and Mood:
Describe the tone (the poet’s attitude) and the mood (the emotion created for the reader) throughout the poem.

Provide your response in a clear and detailed manner, as if explaining the poem to someone unfamiliar with it.

This is the poem: {poem}.
"""


def save_interpretations_to_parquet(df, source, num_examples):
    """Saves the entire batch of interpretations as a Parquet file."""
    if not os.path.exists('interpretation'):
        os.makedirs('interpretation')

    file_name = f"{source}_first_{num_examples}.parquet"
    file_path = os.path.join('interpretation', file_name)

    df.to_parquet(file_path, index=False)
    print(f"Saved to {file_path}")


def get_interpretation_from_openai(source,
                                   num_examples=NUM_EXAMPLES,
                                   model=MODEL,
                                   system_prompt=SYSTEM_PROMPT,
                                   prompt_template=PROMPT_TEMPLATE,
                                   verbose=True):

    if source == 'poets_org':
        df = pd.read_parquet('data/poets_org.parquet')
        # relevant columns: author, title, poem_text
        title_key = 'title'
        author_key = 'author'
        poem_key = 'poem_text'
    elif source == 'public_domain_poetry':
        df = pd.read_json("hf://datasets/DanFosing/public-domain-poetry/poems.json")
        # relevant columns: Title, Author, text (yes, lower case)
        title_key = 'Title'
        author_key = 'Author'
        poem_key = 'text'
    else:
        raise ValueError('Source must be either "poets_org" or "public_domain_poetry"')
    generated_data = []

    for i, (_, item) in enumerate(df.head(num_examples).iterrows()):
        title = item[title_key]
        author = item[author_key]
        poem = item[poem_key]

        prompt = prompt_template.format(title=title, author=author, poem=poem)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system",
                 "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            seed=42,
            n=1,
        )
        generated_text = response.choices[0].message.content

        # collect the generated interpretation and the original poem information
        generated_data.append({
            'author': author,
            'title': title,
            'poem': poem,
            'interpretation': generated_text
        })

        if verbose:
            print("*" * 90)
            print(f"Generated interpretation for {title} by {author}: {generated_text}")
            print("*" * 90)

    interpretations_df = pd.DataFrame(generated_data)
    save_interpretations_to_parquet(interpretations_df, source, num_examples)



if __name__ == '__main__':

    client = OpenAI(
        api_key=os.environ.get(OPENAI_API_PROJECT_KEY_, "Set up OPENAI API Key"))

    get_interpretation_from_openai(source='poets_org', num_examples=50)

    get_interpretation_from_openai(source='public_domain_poetry', num_examples=50)
