# genetic / model

A small DNA language model trained from scratch on the human reference genome (GRCh37/hg19) with Hugging Face.

## Roadmap

- [x] Project setup (`uv`, dependencies)
- [x] Download reference data (`scripts/`)
- [x] Character-level tokenizer (`tokenizers`)
- [x] Dataset: FASTA → windows → tokens (`datasets`)
- [x] Llama-style model + training loop (`transformers` `Trainer`)
- [x] Evaluation: bits per base vs. Markov baselines
- [ ] Zero-shot variant scoring (log-likelihood ratio)
- [ ] Publish to the Hugging Face Hub

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

_Coming soon._
