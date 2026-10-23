# genetic

Upload a genetic test, get insights from open sources.

The project starts with a small DNA language model trained from scratch with Hugging Face, then grows into an API and an MCP server that combine the model with public databases (ClinVar, PharmGKB/CPIC, GWAS Catalog, PGS Catalog).

> **Built to learn. Not for clinical use.** Nothing here is medical advice or a diagnosis.

🤗 **Model:** [rubenich/genetic-dna-tiny](https://huggingface.co/rubenich/genetic-dna-tiny) on the Hugging Face Hub: 6.3M parameters, 1.744 bits per base on held-out chr21 (best Markov baseline: 1.831).

## Structure

| Folder | What it is | Status |
|---|---|---|
| [`model/`](model/) | DNA language model: tokenizer, dataset, training, evaluation, variant scoring | ✅ v1 [on the Hub](https://huggingface.co/rubenich/genetic-dna-tiny) |
| [`api/`](api/) | HTTP API: genotype file parsing, database lookups, model inference | 🔜 planned |
| [`mcp/`](mcp/) | MCP server exposing the API as tools for LLMs | 🔜 planned |
| [`docs/`](docs/) | Notes and learnings from each phase | 📝 ongoing |

## Data and privacy

- Training uses only public data: the human reference genome **GRCh37/hg19** from UCSC.
- Personal genetic files (raw data exports, VCFs) never belong in this repository. `.gitignore` blocks common formats as a safety net, but keep them outside the project folder.

## License

[MIT](LICENSE)
