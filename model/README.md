# genetic / model

A small DNA language model trained from scratch on the human reference genome (GRCh37/hg19) with Hugging Face.

## Roadmap

- [x] Project setup (`uv`, dependencies)
- [x] Download reference data (`scripts/`)
- [x] Character-level tokenizer (`tokenizers`)
- [x] Dataset: FASTA → windows → tokens (`datasets`)
- [x] Llama-style model + training loop (`transformers` `Trainer`)
- [x] Evaluation: bits per base vs. Markov baselines
- [x] Zero-shot variant scoring (log-likelihood ratio)
- [ ] Publish to the Hugging Face Hub

## Pipeline

![Model pipeline: download, tokenizer, dataset, training, evaluation, variant scoring, publish](../docs/model-pipeline.svg)

## Layout

```
model/
├── configs/          # training configs (model size, sequence length, learning rate)
├── scripts/          # data download and helper scripts
├── src/genetic_model/  # package code, one module per phase
├── tests/            # unit tests
├── data/             # downloaded and processed data (git-ignored, see data/README.md)
└── runs/             # tokenizer, checkpoints and trained models (git-ignored)
```

## Setup

```bash
cd model
uv sync
```

## Results

`configs/tiny.yaml`: 6.3M parameters, 4,000 steps (about one pass over chr22, both strands) on an Apple M1 Pro.

Bits per base on chr21, which the model never saw: 35.1M bases in 34,265 windows without `N`, each base predicted from the bases before it in its 1,024 bp window. Lower is better; a uniform guess over A/C/G/T is 2.0.

| Method | Bits per base |
|---|---|
| **Model (`runs/tiny`)** | **1.744** |
| Markov order 10 | 1.831 |
| Markov order 8 | 1.857 |
| Markov order 6 | 1.901 |
| Markov order 4 | 1.926 |
| Markov order 2 | 1.938 |
| Markov order 1 | 1.952 |
| Markov order 0 (base frequencies) | 1.991 |
| Uniform | 2.000 |

The model beats the best Markov chain by 0.087 bits per base. Validation loss was still falling at the end of training (1.749 → 1.747 bits/token over the last 1,000 steps), so a longer run or more chromosomes would likely help.

Reproduce: `uv run python -m genetic_model.evaluate` (about 6 minutes; writes `runs/tiny/eval_results.json`).
