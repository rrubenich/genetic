# data

Everything in this folder is git-ignored and can be regenerated.

| Path | Contents | Source |
|---|---|---|
| `hg19/` | Raw reference genome FASTA files (e.g. `chr21.fa`, `chr22.fa`) | [UCSC hg19 chromosomes](https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/) (GRCh37) |
| `processed/` | Tokenized train/validation `DatasetDict` (1024 bp windows) | `uv run python -m genetic_model.data` |

## Reference genome source

- **Assembly:** hg19 (UCSC name for GRCh37, Genome Reference Consortium Human Build 37, Feb. 2009)
- **Provider:** UCSC Genome Browser download server, `hgdownload.soe.ucsc.edu`
- **Directory:** <https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/> (one gzipped FASTA per chromosome)
- **Downloaded on:** 2026-10-23
- **Format:** FASTA, soft-masked (repeats in lowercase, `N` for gaps)
- **Terms:** UCSC lists the hg19 sequence as freely usable; see the `README.txt` in the directory above.

SHA-256 of the downloaded archives (before `gunzip`):

| File | SHA-256 |
|---|---|
| `chr20.fa.gz` | `f15c27115a944c1a3cdc6fd2758daa1b12ef293a25f0cea56e899cebce9f6b82` |
| `chr21.fa.gz` | `1953ac4ea94b65885bb619bea88a9a0c5c3ad7a98302b3de71be319995c6f8e7` |
| `chr22.fa.gz` | `4a5a0049b23232b163e575bf506b8af86e7d563fa9990311fb9fa3346d6865ec` |
| `chrX.fa.gz` | `1e9bd71f702f4658b33370c45ace178588ee9ee9623f0ff593b6222dc4d8e643` |

## Downloading

The data is fetched with [`scripts/download_hg19.sh`](../scripts/download_hg19.sh). It downloads each chromosome from the UCSC directory above into `data/hg19/`, verifies the archive against the SHA-256 table above (for chromosomes listed there), then uncompresses it to `<chrom>.fa` so pyfaidx can read random regions. Chromosomes that already have a `.fa` are skipped, and an existing `.fa.gz` is reused instead of downloaded again. Paths resolve relative to `model/`, so it can be run from any directory:

```bash
scripts/download_hg19.sh              # default: chr21 chr22
scripts/download_hg19.sh chr1 chr2    # any other chromosomes
```

To download a chromosome by hand instead (run from `model/`):

```bash
mkdir -p data/hg19
curl -L https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes/chr22.fa.gz -o data/hg19/chr22.fa.gz
shasum -a 256 data/hg19/chr22.fa.gz   # compare against the table above
gunzip data/hg19/chr22.fa.gz
```

Personal genetic files do **not** go here. Keep them outside the repository.
