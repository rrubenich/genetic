"""Model + training: a small Llama-style decoder trained with next-base prediction.

Usage: python -m genetic_model.train [config.yaml] [--max-steps N]
"""

import argparse
import math

from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    LlamaConfig,
    LlamaForCausalLM,
    Trainer,
    TrainerCallback,
    TrainingArguments,
)

from .config import DEFAULT_CONFIG, load_config, resolve


def build_model(config: dict, tokenizer) -> LlamaForCausalLM:
    m = config["model"]
    multiple = m.get("vocab_multiple", 1)
    model_config = LlamaConfig(
        # 9 real tokens, padded up so the embedding matrix has a GPU-friendly shape.
        # The extra rows are never produced by the tokenizer and quickly get ~0 probability.
        vocab_size=math.ceil(len(tokenizer) / multiple) * multiple,
        hidden_size=m["hidden_size"],
        num_hidden_layers=m["num_hidden_layers"],
        num_attention_heads=m["num_attention_heads"],
        intermediate_size=m["intermediate_size"],
        max_position_embeddings=config["data"]["window"] + 2,  # + [BOS] and [EOS]
        pad_token_id=tokenizer.pad_token_id,
        bos_token_id=tokenizer.bos_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    return LlamaForCausalLM(model_config)


class BitsPerTokenCallback(TrainerCallback):
    """Print eval loss in bits: 2.0 is a uniform guess over A/C/G/T."""

    def on_evaluate(self, args, state, control, metrics=None, **kwargs):
        if metrics and "eval_loss" in metrics:
            print(f"step {state.global_step}: {metrics['eval_loss'] / math.log(2):.4f} bits/token")


def train(config: dict, max_steps: int | None = None) -> None:
    tokenizer = AutoTokenizer.from_pretrained(resolve(config["tokenizer_dir"]))
    dataset = load_from_disk(resolve(config["data"]["out_dir"]))

    args_dict = dict(config["training"])
    max_eval_samples = args_dict.pop("max_eval_samples", None)
    args_dict["output_dir"] = str(resolve(args_dict["output_dir"]))
    if max_steps is not None:
        args_dict["max_steps"] = max_steps
    args = TrainingArguments(**args_dict, report_to="none")

    validation = dataset["validation"]
    if max_eval_samples:
        # A fixed random subset: evaluating all of chr21 every few hundred steps is too slow
        validation = validation.shuffle(seed=args.seed).select(range(max_eval_samples))

    model = build_model(config, tokenizer)
    print(f"Model parameters: {model.num_parameters() / 1e6:.2f}M")

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=dataset["train"],
        eval_dataset=validation,
        # mlm=False: labels are the inputs themselves; the model shifts them by one internally
        data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
        callbacks=[BitsPerTokenCallback()],
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"Saved model and tokenizer to {args.output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument("--max-steps", type=int, help="override training.max_steps (e.g. for a smoke test)")
    cli = parser.parse_args()
    train(load_config(cli.config), cli.max_steps)
