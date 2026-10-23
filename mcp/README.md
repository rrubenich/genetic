# genetic / mcp

🔜 **Planned.**

[Model Context Protocol](https://modelcontextprotocol.io) server that exposes the [`api/`](../api/) as tools for LLM clients, for example:

| Tool | Purpose |
|---|---|
| `parse_genotype_file` | Read an uploaded genotype file |
| `lookup_variant` | ClinVar and frequency data for a variant |
| `drug_response` | Pharmacogenomic guidance (PharmGKB/CPIC) |
| `score_variant` | Zero-shot score from the DNA language model |

Not for clinical use.
