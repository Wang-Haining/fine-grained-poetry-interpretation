"""
This module performs supervised fine-tuning of LLMs using:

- GPT-4 generated interpretations: the Fine-grained Poetry Interpretation corpus for
the Poetry Foundation Corpus and Public Domain Poetry Corpus.
- AllPoetry Corpus: filtered users' comments on their feelings and interpretations of
poetry.

Currently, we only support two models:
- 'allenai/OLMo-1B-0724-hf' (with a context length of 4096)
- 'google/gemma-2-2b' (with a context length of 8192)

"""

__author__ = "hw56@indiana.edu"
__version__ = "0.0.1"
__license__ = "0BSD"

import argparse
import os
from typing import List

import torch
import wandb
from datasets import DatasetDict, load_from_disk, concatenate_datasets
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          EarlyStoppingCallback, TrainingArguments)
from trl import SFTTrainer, set_seed, SFTConfig


os.environ["TOKENIZERS_PARALLELISM"] = "false"
PROJECT_NAME = 'Poem_Interpretation'
CKPTS_DIR = 'ckpts'
# todo: you have to have a pre-fixed split
POEM_INTERPRETATION_CORPUS = 'interpretation/poem_interpretation_corpus_v001.hf'
ALL_POETRY_CORPUS = ''

PROMPT_TEMPLATE = """
Interpret "{title}" by {author}. This is the poem: {poem}

Interpretation: {interpretation}
"""


def formatting_func(example: DatasetDict) -> List[str]:
    """
    Formats input examples by concatenating the source text with the target text,
    using the task-specific prefix and response template.

    Args:
        example: A dataset dictionary containing 'source' and 'target' fields.

    Returns:
        A list of formatted strings ready for model training.
    """
    output_texts = []
    for i in range(len(example["poem"])):
        text = PROMPT_TEMPLATE.format(title=example["title"][i],
                                      author=example["author"][i],
                                      poem=example["poem"][i],
                                      interpretation=example["interpretation"][i])
        output_texts.append(text)

    return output_texts


if __name__ == "__main__":

    set_seed(42)
    parser = argparse.ArgumentParser(description="Supervise Fine-tuning OLMo-1B (0724) or Gemma-2-2B")
    parser.add_argument("--model", type=str,
                        choices=['allenai/OLMo-1B-0724-hf', 'google/gemma-2-2b'],
                        help="A huggingface model's name")
    parser.add_argument("--corpus", type=str,
                        choices=['poem_interpretation', 'all_poetry'],
                        nargs='+', help="One or more corpora to use; if multiple are selected, "
                        "they will be mixed proportionally based on their sizes "
                        "(i.e., proportional random sampling)")
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--per_device_train_batch_size",
                        type=int, default=2)
    parser.add_argument("--gradient_checkpointing", action='store_true',
                        help="Whether to use gradient checkpointing")
    parser.add_argument("--deepspeed", action='store_true',
                        help="Whether to use DeepSpeed for training")
    args = parser.parse_args()

    run_name = f'sft_{args.model.split("/")[-1]}'

    training_args = SFTConfig(
        output_dir=f"{CKPTS_DIR}/{run_name}",
        max_seq_length=4096,
        packing=False,
        eval_packing=False,
        overwrite_output_dir=True,
        num_train_epochs=50.0,
        do_train=True,
        do_eval=True,
        eval_strategy="steps",
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_train_batch_size,  # same to training
        gradient_accumulation_steps=4,
        learning_rate=args.learning_rate,
        lr_scheduler_type='constant_with_warmup',
        warmup_steps=50,
        weight_decay=1e-1,
        logging_steps=500,
        eval_steps=500,
        bf16=True,
        report_to="wandb",
        load_best_model_at_end=True,
        save_steps=500,
        save_total_limit=3,
        remove_unused_columns=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={'use_reentrant': False} if args.gradient_checkpointing else None,
        deepspeed='runs/ds_sft_config.json' if args.deepspeed else None,
    )
    wandb.init(project=PROJECT_NAME, name=run_name, config=training_args.to_dict())

    tokenizer = AutoTokenizer.from_pretrained(args.model, padding_side="right")
    # init dataset
    train_datasets = []
    eval_datasets = []
    if 'poem_interpretation' in args.corpus:
        # columns: ['author', 'title', 'poem', 'interpretation', 'source']
        ds = load_from_disk(POEM_INTERPRETATION_CORPUS)
        train_datasets.append(ds['train'])
        eval_datasets.append(ds['validation'])
    if 'all_poetry' in args.corpus:
        ds = load_from_disk(ALL_POETRY_CORPUS)
        train_datasets.append(ds['train'])
        eval_datasets.append(ds['validation'])
    # combine and shuffle
    train_dataset = concatenate_datasets(train_datasets)
    train_dataset = train_dataset.shuffle(seed=42)
    eval_dataset = concatenate_datasets(eval_datasets)

    # init model after trainingArgs init
    model = AutoModelForCausalLM.from_pretrained(args.model,
                                                 torch_dtype=torch.bfloat16)
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.learning_rate)

    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        formatting_func=formatting_func,
        args=training_args,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=5)],
    )

    trainer.train()
