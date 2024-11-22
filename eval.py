"""
This module performs evaluation of LLMs fine-tuned for poem interpretation.

- Evaluates the model's performance on the test set of the Poem Interpretation Corpus or AllPoetry Corpus.
- Calculates metrics such as BLEU, ROUGE-L, METEOR, and Perplexity (PPL).

Currently, we support the models:

- 'allenai/OLMo-1B-0724-hf'
- 'google/gemma-2-2b'

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

PROMPT_TEMPLATE = """
Interpret "{title}" by {author}. This is the poem: {poem}

Interpretation:"""


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
        prompt = PROMPT_TEMPLATE.format(title=title, author=author, poem=poem)
        inputs.append(prompt.strip())
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

        # Tokenize the input texts
        inputs = tokenizer(input_texts, return_tensors="pt", padding=True, truncation=True).to(device)

        # Generate outputs
        outputs = model.generate(
            input_ids=inputs['input_ids'],
            attention_mask=inputs['attention_mask'],
            max_new_tokens=4096,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            num_return_sequences=1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

        for i in range(outputs.shape[0]):
            input_length = inputs['input_ids'].shape[1]
            generated_sequence = outputs[i][input_length:]
            generated_text = tokenizer.decode(generated_sequence, skip_special_tokens=True)
            reference_text = references[i]
            input_text = input_texts[i]

            # compute per-sample metrics
            bleu_score = bleu_metric.compute(predictions=[generated_text], references=[[reference_text]])
            meteor_score = meteor_metric.compute(predictions=[generated_text], references=[reference_text])
            rouge_score = rouge_metric.compute(predictions=[generated_text], references=[reference_text], rouge_types=["rougeL"])

            # compute perplexity
            with torch.no_grad():
                # prepare the generated text as input for computing perplexity
                encodings = tokenizer(generated_text, return_tensors='pt').to(device)
                max_length = model.config.n_positions
                # ensure the sequence length does not exceed the model's capacity
                input_ids = encodings['input_ids'][:, :max_length]
                attention_mask = encodings['attention_mask'][:, :max_length]

                outputs_ppl = model(input_ids=input_ids, attention_mask=attention_mask, labels=input_ids)
                loss = outputs_ppl.loss
                ppl = torch.exp(loss).item()

            results.append({
                'input_text': input_text,
                'generated_text': generated_text.strip(),
                'reference_text': reference_text.strip(),
                'bleu': bleu_score['bleu'],
                'meteor': meteor_score['meteor'],
                'rougeL': rouge_score['rougeL'].mid.fmeasure,
                'ppl': ppl
            })

    df = pd.DataFrame(results)
    return df


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Evaluate LLMs on poem interpretation")
    parser.add_argument("--ckpt_path", type=str, required=True,
                        help="Path to the model checkpoint directory")
    parser.add_argument("--model", type=str,
                        choices=['allenai/OLMo-1B-0724-hf', 'google/gemma-2-2b'],
                        required=True,
                        help="Name of the base model")
    parser.add_argument("--corpus", type=str,
                        choices=['poem_interpretation', 'all_poetry'],
                        nargs='+', required=True,
                        help="One or more corpora to use for evaluation")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size for evaluation")
    parser.add_argument("--output_file", type=str, required=True,
                        help="Output CSV file to save the results")
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(args.ckpt_path, torch_dtype=torch.bfloat16)

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
