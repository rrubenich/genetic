"""Hugging Face Hub: write a model card for a trained model and (optionally) push it.

Usage: python -m genetic_model.publish [--model runs/tiny]                  # writes README.md only
       python -m genetic_model.publish --repo-id USER/genetic-dna-tiny --push
Pushing needs a token: run `uv run hf auth login` first.
"""

import argparse
import json
from pathlib import Path

from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DEFAULT_CONFIG, load_config, resolve

CARD = """\
---
license: mit
library_name: transformers
pipeline_tag: text-generation
tags:
  - dna
  - genomics
  - biology
  - causal-lm
---

# {name}

A small DNA language model ({params:.1f}M parameters) trained from scratch on the human
reference genome **GRCh37/hg19**, built to learn how genomic language models work.

> **Not for clinical use.** Scores from this model are not medical advice or a diagnosis.

## Model

- Llama-style decoder (`LlamaForCausalLM`): hidden {hidden_size}, {num_hidden_layers} layers,
  {num_attention_heads} heads, intermediate {intermediate_size}
- Context: {window} bases (+ `[BOS]`/`[EOS]`)
- Objective: next-base prediction (causal language modeling)
- Tokenizer: character level, `[PAD] [UNK] [BOS] [EOS] A C G T N`. One base = one token,
  so a single-base change alters exactly one token. Lowercase (soft-masked) input is
  uppercased and any other letter becomes `N`.

## Training data

- UCSC hg19 chromosomes from `hgdownload.soe.ucsc.edu`
- Train: {train_chroms} (both strands), validation: {validation_chroms} (+ strand)
- Non-overlapping {window} bp windows, dropping windows with more than
  {max_n_percent:g}% unknown bases (`N`)
- {max_steps} steps, batch size {batch_size}, learning rate {learning_rate:g} (cosine)

## Evaluation

Bits per base on {validation_chroms} ({validation_bases:,} bases in windows without `N`),
each base predicted from the bases before it in its window. Uniform guessing = 2.0;
lower is better.

| Method | Bits per base |
|---|---|
{eval_rows}

## Usage

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

tok = AutoTokenizer.from_pretrained("{repo_id}")
model = AutoModelForCausalLM.from_pretrained("{repo_id}")

ids = tok("ACGTTGCA", return_tensors="pt")["input_ids"]
with torch.no_grad():
    logits = model(input_ids=ids).logits
```

Zero-shot variant scoring compares `log P(alt window) - log P(ref window)` on an hg19
window around a single-base change; see `genetic_model.score_variant` in the source repo.

## Limitations

- Trained on {n_train_chroms} chromosome(s) of one reference genome: it has not seen
  human variation, other chromosomes, or other species.
- Tiny model and short context: it captures local sequence statistics, not
  long-range regulatory structure.
- Coordinates and alleles must be GRCh37/hg19, + strand.
"""


def eval_rows(results: dict) -> str:
    rows = [f"| **This model** | **{results['model']:.4f}** |"]
    rows += [f"| Markov order {k} | {v:.4f} |" for k, v in results["markov"].items()]
    rows.append(f"| Uniform | {results['uniform']:.4f} |")
    return "\n".join(rows)


def write_card(config: dict, model_dir: Path, repo_id: str) -> Path:
    results = json.loads((model_dir / "eval_results.json").read_text())
    model = AutoModelForCausalLM.from_pretrained(model_dir)
    data, training = config["data"], config["training"]
    card = CARD.format(
        name=repo_id.split("/")[-1],
        repo_id=repo_id,
        params=model.num_parameters() / 1e6,
        **{k: config["model"][k] for k in
           ("hidden_size", "num_hidden_layers", "num_attention_heads", "intermediate_size")},
        window=data["window"],
        train_chroms=", ".join(data["train_chroms"]),
        validation_chroms=", ".join(data["validation_chroms"]),
        n_train_chroms=len(data["train_chroms"]),
        max_n_percent=data["max_n_fraction"] * 100,
        max_steps=training["max_steps"],
        batch_size=training["per_device_train_batch_size"],
        learning_rate=training["learning_rate"],
        validation_bases=results["validation_bases"],
        eval_rows=eval_rows(results),
    )
    path = model_dir / "README.md"
    path.write_text(card)
    return path


def push(model_dir: Path, repo_id: str, private: bool) -> None:
    for obj in (AutoModelForCausalLM.from_pretrained(model_dir), AutoTokenizer.from_pretrained(model_dir)):
        obj.push_to_hub(repo_id, private=private)
    # Model card and eval results, which push_to_hub does not upload on its own
    from huggingface_hub import HfApi
    api = HfApi()
    for name in ("README.md", "eval_results.json"):
        api.upload_file(path_or_fileobj=model_dir / name, path_in_repo=name, repo_id=repo_id)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="runs/tiny")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument("--repo-id", default="genetic-dna-tiny", help="USER/NAME on the Hub")
    parser.add_argument("--push", action="store_true", help="upload to the Hub (default: only write the card)")
    parser.add_argument("--private", action="store_true")
    cli = parser.parse_args()

    model_dir = resolve(cli.model)
    print(f"Wrote {write_card(load_config(cli.config), model_dir, cli.repo_id)}")
    if cli.push:
        push(model_dir, cli.repo_id, cli.private)
        print(f"Pushed to https://huggingface.co/{cli.repo_id}")
