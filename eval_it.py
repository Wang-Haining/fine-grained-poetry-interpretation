"""
This module performs evaluation of LLMs on poem interpretation without fine-tuning.

- Evaluates the model's performance on the test set of the Poem Interpretation Corpus.
- Calculates metrics such as BLEU, ROUGE-L, METEOR, and Perplexity (PPL).

Currently, we support the model(s):

- 'google/gemma-2-2b-it'

"""

__author__ = "hw56@indiana.edu"
__version__ = "0.0.1"
__license__ = "0BSD"

import argparse
import os
from typing import List, Dict

import torch
from datasets import load_from_disk, concatenate_datasets, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import evaluate
from tqdm import tqdm
import pandas as pd

os.environ["TOKENIZERS_PARALLELISM"] = "false"

CKPTS_DIR = 'ckpts'
POEM_INTERPRETATION_CORPUS = 'interpretation/poem_interpretation_corpus_v001.hf'
ALL_POETRY_CORPUS = ''  # todo

# combine the system prompt with the user prompt
PROMPT_INSTRUCTIONS = """You are an assistant that strictly follows user instructions. Provide only the analysis and content requested by the user without any greetings, closing remarks, or unnecessary additions. Do not include any extra text beyond what is required."""

USER_PROMPT_TEMPLATE = """
{instructions}

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

This is the poem: {poem}
"""


def prepare_examples(examples) -> Dict[str, List[str]]:
    """
    Prepares the input prompts and references for the dataset.

    Args:
        examples: A batch of examples from the dataset.

    Returns:
        A dictionary with 'input_text' and 'reference' fields.
    """
    inputs = []
    references = []
    for title, author, poem, interpretation in zip(
            examples['title'], examples['author'], examples['poem'], examples['interpretation']):
        user_prompt = USER_PROMPT_TEMPLATE.format(
            instructions=PROMPT_INSTRUCTIONS,
            title=title,
            author=author,
            poem=poem
        )
        inputs.append(user_prompt.strip())
        references.append(interpretation.strip())
    return {'input_text': inputs, 'reference': references}


def evaluate_model(model, tokenizer, dataset: Dataset, batch_size: int = 2) -> pd.DataFrame:
    """
    Evaluates the model on the given dataset and computes evaluation metrics.

    Args:
        model: The language model to evaluate.
        tokenizer: The tokenizer corresponding to the model.
        dataset: The dataset to evaluate on, must contain 'input_text' and 'reference' fields.
        batch_size: The batch size for evaluation.

    Returns:
        A pandas DataFrame containing input, generated text, reference, and metrics.
    """
    device = torch.device("cuda:0")
    model.to(device)

    from torch.utils.data import DataLoader

    data_loader = DataLoader(dataset, batch_size=batch_size)

    # init metrics
    bleu_metric = evaluate.load("bleu")
    meteor_metric = evaluate.load("meteor")
    rouge_metric = evaluate.load("rouge")

    results = []

    for batch in tqdm(data_loader, desc="Evaluating"):
        input_texts = batch['input_text']
        references = batch['reference']

        # prepare chat messages using the tokenizer's chat template
        messages_list = []
        for input_text in input_texts:
            messages = [
                {"role": "user", "content": input_text}
            ]
            messages_list.append(messages)

        # apply chat template and tokenize
        inputs = tokenizer.apply_chat_template(
            messages_list,
            return_tensors="pt",
            return_dict=True,
            padding=True,
            max_length=4096,  # set max_length to a reasonable value
            truncation=True
        ).to(device)

        # calculate max_new_tokens to not exceed model's max length
        input_lengths = inputs['input_ids'].shape[1]
        max_model_length = 4096  # Use the same value as max_length
        max_new_tokens = max_model_length - input_lengths
        if max_new_tokens <= 0:
            print("Warning: Input length exceeds the model's maximum context length. Truncating input.")
            max_new_tokens = 1  # generate at least one token

        # generate outputs
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

        for i in range(outputs.shape[0]):
            output_ids = outputs[i]
            generated_text = tokenizer.decode(output_ids, skip_special_tokens=True)
            # extract the assistant's response from the generated text
            # assuming the assistant's response follows the <start_of_turn>model token
            # and ends with <end_of_turn>
            start_token = tokenizer.additional_special_tokens_ids[
                tokenizer.additional_special_tokens.index('<start_of_turn>')
            ]  # model's start token
            end_token = tokenizer.additional_special_tokens_ids[
                tokenizer.additional_special_tokens.index('<end_of_turn>')
            ]  # end of turn token

            # find the indices of the assistant's response
            start_indices = (output_ids == start_token).nonzero(as_tuple=True)[0]
            end_indices = (output_ids == end_token).nonzero(as_tuple=True)[0]

            if len(start_indices) > 0:
                start_idx = start_indices[-1] + 1
                if len(end_indices) > 0 and end_indices[-1] > start_idx:
                    end_idx = end_indices[-1]
                else:
                    end_idx = output_ids.size(0)
                assistant_ids = output_ids[start_idx:end_idx]
                assistant_response = tokenizer.decode(assistant_ids, skip_special_tokens=True).strip()
            else:
                # if the tokens are not found, use the entire generated text
                assistant_response = generated_text.strip()

            reference_text = references[i]
            input_text = input_texts[i]

            # Compute per-sample metrics
            bleu_score = bleu_metric.compute(predictions=[assistant_response], references=[[reference_text]])
            meteor_score = meteor_metric.compute(predictions=[assistant_response], references=[reference_text])
            rouge_score = rouge_metric.compute(
                predictions=[assistant_response],
                references=[reference_text],
                rouge_types=["rougeL"]
            )

            # compute perplexity
            with torch.no_grad():
                # prepare the generated text as input for computing perplexity
                encodings = tokenizer(assistant_response, return_tensors='pt').to(device)
                max_length = 4096  # Ensure this matches the model's max length
                # ensure the sequence length does not exceed the model's capacity
                input_ids_ppl = encodings['input_ids'][:, :max_length]
                attention_mask_ppl = encodings['attention_mask'][:, :max_length]

                outputs_ppl = model(input_ids=input_ids_ppl, attention_mask=attention_mask_ppl, labels=input_ids_ppl)
                loss = outputs_ppl.loss
                ppl = torch.exp(loss).item()

            results.append({
                'input_text': input_text,
                'generated_text': assistant_response,
                'reference_text': reference_text.strip(),
                'bleu': bleu_score['bleu'],
                'meteor': meteor_score['meteor'],
                'rougeL': rouge_score['rougeL'],
                'ppl': ppl
            })

    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate LLMs on poem interpretation")
    parser.add_argument("--hf_model", type=str, required=True,
                        help="Name or path of the HuggingFace model")
    parser.add_argument("--corpus", type=str,
                        choices=['poem_interpretation', 'all_poetry'],
                        nargs='+', required=True,
                        help="One or more corpora to use for evaluation")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for evaluation")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output CSV file to save the results")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.hf_model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.hf_model, torch_dtype=torch.bfloat16)

    test_datasets = []
    if 'poem_interpretation' in args.corpus:
        # columns: ['author', 'title', 'poem', 'interpretation', 'source']
        ds = load_from_disk(POEM_INTERPRETATION_CORPUS)
        test_datasets.append(ds['test'])
    if 'all_poetry' in args.corpus and ALL_POETRY_CORPUS:
        ds = load_from_disk(ALL_POETRY_CORPUS)
        test_datasets.append(ds['test'])
    # prepare inputs and references
    test_dataset = concatenate_datasets(test_datasets)
    test_dataset = test_dataset.map(prepare_examples, batched=True)

    # eval
    df = evaluate_model(model, tokenizer, test_dataset, batch_size=args.batch_size)

    # save
    df.to_parquet(args.output_file)

    print(f"Evaluation results saved to {args.output_file}")
