# CLAUDE.md

Context for Claude Code when working in this repository.

## Project

**genetic**: upload a genetic test, get insights from open sources.
Monorepo with three parts:

- `model/`: small DNA language model trained from scratch with Hugging Face (current focus)
- `api/`: planned HTTP API (genotype parsing, ClinVar/PharmGKB/GWAS/PGS lookups, model inference)
- `mcp/`: planned MCP server exposing the API as tools for LLMs

Built to learn. Not for clinical use.

## How to collaborate

- **This is a learning project.** The owner wants to write the phase code himself. Explain concepts and show how to do things (snippets, step-by-step); do not implement whole phases unless explicitly asked. Reviewing his code is welcome.
- **Everything in the repo is in English**: code, comments, docs, commit messages.
- Conversation can be in Portuguese (pt-BR).
- Conventional commits, one commit per phase (e.g. `feat(model): character-level DNA tokenizer`).
- Never commit or push without being asked.

## Privacy rules (public repo)

- Personal genetic files (raw data exports, VCFs) must never be inside this repository or referenced by path. `.gitignore` blocks common formats as a safety net only.
- Use fake fixtures for tests (e.g. `api/tests/fixtures/genera_sample.csv`).
- Do not interpret anyone's genetic results in this repo.

## model/: decisions so far

- Python package: `genetic_model` (src layout), managed with `uv`, Python >= 3.14, `transformers` 5.x.
- Data: human reference genome **GRCh37/hg19** from UCSC (same build as consumer genotyping arrays). Downloaded by `scripts/download_hg19.sh` into `data/hg19/` (git-ignored).
- Tokenizer: character-level (`[PAD] [UNK] [BOS] [EOS] A C G T N`) with `tokenizers` WordLevel, wrapped in `PreTrainedTokenizerFast`, saved to `runs/tokenizer/`. Chosen so one SNV changes exactly one token. BPE is a later experiment.
- Dataset: non-overlapping 1024 bp windows, drop windows with > 1% N, add reverse complement to train only. **Split by chromosome**: train chr22, validation chr21. Saved with `datasets.save_to_disk` to `data/processed/`.
- Model: small Llama-style decoder (`LlamaForCausalLM`, ~6M params: hidden 256, 6 layers, 8 heads, intermediate 1024), causal LM objective. Trained locally on a MacBook (MPS) with the `Trainer`.
- Evaluation: bits per base (`loss / ln 2`; uniform = 2.0) compared against k-th order Markov chain baselines.
- Usage: zero-shot variant scoring via log-likelihood ratio `log P(alt) - log P(ref)` on an hg19 window (pyfaidx needs uncompressed FASTA).
- Publish: code on GitHub, model and tokenizer on the Hugging Face Hub (`push_to_hub`).

## Roadmap (model/)

1. [x] Project setup and structure
2. [x] Download script (`scripts/download_hg19.sh`)
3. [ ] Tokenizer: `src/genetic_model/tokenizer.py` + `tests/test_tokenizer.py` ← next
4. [ ] Dataset: `src/genetic_model/data.py`
5. [ ] Model + training: `src/genetic_model/train.py`, `configs/tiny.yaml`
6. [ ] Evaluation vs Markov baselines: `src/genetic_model/evaluate.py`
7. [ ] Variant scoring: `src/genetic_model/score_variant.py`
8. [ ] Hugging Face Hub + model card, results in README

## Input format notes (for api/ later)

Genera raw data export: gzipped CSV, header `RSID,CHROMOSOME,POSITION,RESULT`, ~650k markers, positions in **GRCh37**.

- RESULT: two-letter genotypes (allele order not normalized: `AG` and `GA` both occur), single letters on Y/MT, `II`/`DD`/`ID` for indels, `--` for no-calls.
- ~4.5k IDs are `chrom:pos` instead of rsIDs, and that position does not match the POSITION column (likely GRCh38); use POSITION.
- Some rsIDs appear multiple times (different probes); key by `(rsid, chrom, pos)`.

## Commands (from model/)

```bash
uv sync
./scripts/download_hg19.sh            # chr21, chr22 into data/hg19/
uv run python -m genetic_model.tokenizer
uv run pytest
```
