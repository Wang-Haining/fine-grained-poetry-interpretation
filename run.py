import os
import pandas as pd
from openai import OpenAI
import argparse
from tqdm import tqdm


OPENAI_API_PROJECT_KEY_ = 'OPENAI_API_KEY_POETRY'
MODEL = "gpt-4o-2024-05-13"

SYSTEM_PROMPT = "You are an assistant that strictly follows user instructions. Provide only the analysis and content requested by the user without any greetings, closing remarks, or unnecessary additions. Do not include any extra text beyond what is required."

PROMPT_TEMPLATE = """
Analyze the poem "{title}" by {author}. Please provide a structured, markdown-formatted interpretation following these steps:

1. Summary: Give a brief overview of the poem's main themes and ideas. Summarize the key emotions or messages conveyed by the poet.

2. Stanza-by-Stanza Breakdown:
For each stanza:
- Summarize the content.
- Discuss any literary devices used (such as metaphor, simile, personification, etc.).
- Explain how the stanza contributes to the overall theme of the poem.
- If the poem does not have a traditional stanza structure, adapt the analysis accordingly.

3. Structure and Form:
Discuss the poem's structure (e.g., number of stanzas, rhyme scheme, meter) and how the form contributes to the poem's meaning or emotional effect. Mention any significant literary devices like repetition, enjambment, or caesura if relevant.

4. Imagery, Symbolism, and Figurative Language:
Analyze the imagery, symbols, and figurative language (metaphors, similes, personification, etc.). Explain how these elements enhance the meaning of the poem.

5. Tone and Mood:
Describe the tone (the poet’s attitude towards the subject) and the mood (the emotion created for the reader) throughout the poem.

Provide your response in a clear and detailed manner, as if explaining the poem to someone unfamiliar with it.

This is the poem: {poem}.
"""

def save_interpretations_incrementally(df, source, num_examples):
    """Saves the batch of interpretations incrementally to a Parquet file."""
    file_name = f"{source}_first_{num_examples}.parquet" if num_examples != 'all' else f"{source}.parquet"
    file_path = os.path.join('interpretation', file_name)

    if not os.path.exists('interpretation'):
        os.makedirs('interpretation')

    # if file exists, append to it
    if os.path.exists(file_path):
        existing_df = pd.read_parquet(file_path)
        df = pd.concat([existing_df, df], ignore_index=True)

    df.to_parquet(file_path, index=False)
    print(f"Saved to {file_path}")

def get_last_processed_index(source, num_examples):
    """Returns the index of the last processed poem, if any."""
    file_name = f"{source}_first_{num_examples}.parquet" if num_examples != 'all' else f"{source}.parquet"
    file_path = os.path.join('interpretation', file_name)

    if os.path.exists(file_path):
        existing_df = pd.read_parquet(file_path)
        return len(existing_df)
    return 0

def get_interpretation_from_openai(source,
                                   num_examples,
                                   model=MODEL,
                                   system_prompt=SYSTEM_PROMPT,
                                   prompt_template=PROMPT_TEMPLATE,
                                   verbose=True):

    if source == 'poets_org':  # 12,666 examples
        df = pd.read_parquet('data/poets_org.parquet')
        title_key = 'title'
        author_key = 'author'
        poem_key = 'poem_text'
    elif source == 'public_domain_poetry':  # 38,499 examples
        df = pd.read_json("hf://datasets/DanFosing/public-domain-poetry/poems.json")
        title_key = 'Title'
        author_key = 'Author'
        poem_key = 'text'
    elif source == 'poetry_foundation':  # 13,854 examples
        df = pd.read_csv("data/PoetryFoundationData.csv")
        title_key = 'Title'
        author_key = 'Poet'
        poem_key = 'Poem'
    else:
        raise ValueError('Source must be "poets_org", "public_domain_poetry", or "poetry_foundation"')

    # determine the number of examples to process
    _num_examples = len(df) if num_examples == "all" else int(num_examples)
    last_processed_index = get_last_processed_index(source, num_examples)

    print(f"{source} corpus: {_num_examples} examples to process...")
    print(f"Resuming from index {last_processed_index}...")

    generated_data = []

    # skip poems already processed, adding tqdm progress bar here
    for i, (_, item) in tqdm(enumerate(df.iloc[last_processed_index:_num_examples].iterrows(), start=last_processed_index), total=_num_examples - last_processed_index):
        title = item[title_key]
        author = item[author_key]
        poem = item[poem_key]

        prompt = prompt_template.format(title=title, author=author, poem=poem)

        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
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

        # save after each iteration (or batch of poems for efficiency)
        if i % 10 == 0 or i == _num_examples - 1:  # save every 10 generations or at the end
            interpretations_df = pd.DataFrame(generated_data)
            save_interpretations_incrementally(interpretations_df, source, num_examples)
            generated_data = []  # reset the buffer to avoid duplication

    # save any remaining data after the loop
    if generated_data:
        interpretations_df = pd.DataFrame(generated_data)
        save_interpretations_incrementally(interpretations_df, source, num_examples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Generate poem interpretations using OpenAI GPT models.")
    parser.add_argument('--source', type=str, required=True, choices=['poets_org', 'public_domain_poetry', 'poetry_foundation'],
                        help="Choose the corpus to generate interpretations from: 'poets_org', 'public_domain_poetry', or 'poetry_foundation'.")
    parser.add_argument('--num_examples', type=str, required=True,
                        help="Number of examples to generate. Pass 'all' to generate for all available samples.")
    parser.add_argument('--verbose', action='store_true',
                        help='Print generated interpretations along the way.')
    args = parser.parse_args()

    client = OpenAI(
        api_key=os.environ.get(OPENAI_API_PROJECT_KEY_, "Set up OPENAI API Key"))

    get_interpretation_from_openai(source=args.source,
                                   num_examples=args.num_examples,
                                   verbose=args.verbose)
