#!/usr/bin/env bash
# Download human reference genome chromosomes (GRCh37/hg19) from UCSC.
# Usage: scripts/download_hg19.sh [chrom ...]   (default: chr21 chr22)
set -euo pipefail

# Always resolve paths relative to model/, no matter where the script is called from
MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$MODEL_DIR/data/hg19"
BASE_URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes"

# Default chromosomes when no arguments are given
if [ $# -eq 0 ]; then
  set -- chr21 chr22
fi

mkdir -p "$OUT_DIR"

for chrom in "$@"; do
  if [ -f "$OUT_DIR/$chrom.fa" ]; then
    echo "$chrom: already downloaded, skipping"
    continue
  fi

  echo "$chrom: downloading..."
  curl -fL --progress-bar "$BASE_URL/$chrom.fa.gz" -o "$OUT_DIR/$chrom.fa.gz"

  # Uncompress so pyfaidx can read random regions later
  gunzip -f "$OUT_DIR/$chrom.fa.gz"
done

echo "Done. Files in $OUT_DIR:"
ls -lh "$OUT_DIR"