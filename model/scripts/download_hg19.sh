#!/usr/bin/env bash
# Download human reference genome chromosomes (GRCh37/hg19) from UCSC.
# Usage: scripts/download_hg19.sh [chrom ...]   (default: chr21 chr22)
set -euo pipefail

# Always resolve paths relative to model/, no matter where the script is called from
MODEL_DIR="$(cd "$(dirname "$0")/.." && pwd)"
OUT_DIR="$MODEL_DIR/data/hg19"
BASE_URL="https://hgdownload.soe.ucsc.edu/goldenPath/hg19/chromosomes"

# Known SHA-256 of the UCSC .fa.gz archives (see data/README.md)
expected_sha256() {
  case "$1" in
    chr20) echo f15c27115a944c1a3cdc6fd2758daa1b12ef293a25f0cea56e899cebce9f6b82 ;;
    chr21) echo 1953ac4ea94b65885bb619bea88a9a0c5c3ad7a98302b3de71be319995c6f8e7 ;;
    chr22) echo 4a5a0049b23232b163e575bf506b8af86e7d563fa9990311fb9fa3346d6865ec ;;
    chrX)  echo 1e9bd71f702f4658b33370c45ace178588ee9ee9623f0ff593b6222dc4d8e643 ;;
    *) echo "" ;;
  esac
}

sha256() {
  if command -v sha256sum >/dev/null; then
    sha256sum "$1" | cut -d' ' -f1
  else
    shasum -a 256 "$1" | cut -d' ' -f1
  fi
}

# Default chromosomes when no arguments are given
if [ $# -eq 0 ]; then
  set -- chr21 chr22
fi

mkdir -p "$OUT_DIR"

for chrom in "$@"; do
  fa="$OUT_DIR/$chrom.fa"
  gz="$fa.gz"

  if [ -f "$fa" ]; then
    echo "$chrom: already downloaded, skipping"
    continue
  fi

  # Reuse an archive left by an earlier (manual or interrupted) download
  if [ -f "$gz" ]; then
    echo "$chrom: found existing archive, reusing"
  else
    echo "$chrom: downloading..."
    curl -fL --progress-bar "$BASE_URL/$chrom.fa.gz" -o "$gz.part"
    mv "$gz.part" "$gz"
  fi

  # Verify before gunzip, since gunzip removes the archive
  expected="$(expected_sha256 "$chrom")"
  if [ -n "$expected" ]; then
    actual="$(sha256 "$gz")"
    if [ "$actual" != "$expected" ]; then
      echo "$chrom: checksum mismatch (expected $expected, got $actual)" >&2
      echo "$chrom: delete $gz and re-run to download again" >&2
      exit 1
    fi
    echo "$chrom: checksum OK"
  else
    echo "$chrom: no known checksum, skipping verification"
  fi

  # Uncompress so pyfaidx can read random regions later
  gunzip -f "$gz"
done

echo "Done. Files in $OUT_DIR:"
ls -lh "$OUT_DIR"
