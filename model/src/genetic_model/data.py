"""Dataset: FASTA -> fixed-size windows -> token ids, saved with datasets.save_to_disk.

Usage: python -m genetic_model.data [config.yaml]
"""

import sys
from collections.abc import Iterator
from pathlib import Path

from datasets import Dataset, DatasetDict
from pyfaidx import Fasta
from transformers import AutoTokenizer

from .config import DEFAULT_CONFIG, load_config, resolve

_COMPLEMENT = str.maketrans("ACGTN", "TGCAN")


def reverse_complement(seq: str) -> str:
    """The opposite DNA strand, read in its own 5'->3' direction."""
    return seq.translate(_COMPLEMENT)[::-1]


def n_fraction(seq: str) -> float:
    return seq.count("N") / len(seq)


def windows(chrom_seq: str, size: int, max_n_fraction: float) -> Iterator[tuple[int, str]]:
    """Non-overlapping windows of `size` bases, uppercased, skipping gap-heavy ones.

    Yields (0-based start, sequence). The incomplete tail of the chromosome is dropped.
    """
    for start in range(0, len(chrom_seq) - size + 1, size):
        seq = chrom_seq[start:start + size].upper()
        if n_fraction(seq) <= max_n_fraction:
            yield start, seq


def build_split(fasta_dir: Path, chroms: list[str], size: int, max_n_fraction: float,
                reverse_complement_too: bool) -> Dataset:
    rows = {"chrom": [], "start": [], "strand": [], "sequence": []}

    def add(chrom: str, start: int, strand: str, seq: str) -> None:
        rows["chrom"].append(chrom)
        rows["start"].append(start)
        rows["strand"].append(strand)
        rows["sequence"].append(seq)

    for chrom in chroms:
        # Read the whole chromosome once; slicing a Python str is much faster than pyfaidx per window
        chrom_seq = str(Fasta(str(fasta_dir / f"{chrom}.fa"))[chrom])
        for start, seq in windows(chrom_seq, size, max_n_fraction):
            add(chrom, start, "+", seq)
            if reverse_complement_too:
                add(chrom, start, "-", reverse_complement(seq))
    return Dataset.from_dict(rows)


def tokenize(ds: Dataset, tokenizer) -> Dataset:
    return ds.map(
        lambda batch: {"input_ids": tokenizer(batch["sequence"])["input_ids"]},
        batched=True,
        remove_columns=["sequence"],
        desc="Tokenizing",
    )


def build_dataset(config: dict) -> DatasetDict:
    cfg = config["data"]
    tokenizer = AutoTokenizer.from_pretrained(resolve(config["tokenizer_dir"]))
    fasta_dir = resolve(cfg["fasta_dir"])

    splits = {
        # Reverse complement doubles the training data: both strands are real DNA.
        # Validation keeps only the + strand, so its numbers stay comparable.
        "train": build_split(fasta_dir, cfg["train_chroms"], cfg["window"],
                             cfg["max_n_fraction"], cfg["reverse_complement_train"]),
        "validation": build_split(fasta_dir, cfg["validation_chroms"], cfg["window"],
                                  cfg["max_n_fraction"], False),
    }
    return DatasetDict({name: tokenize(ds, tokenizer) for name, ds in splits.items()})


if __name__ == "__main__":
    config = load_config(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_CONFIG)
    dataset = build_dataset(config)
    out_dir = resolve(config["data"]["out_dir"])
    dataset.save_to_disk(out_dir)
    print(dataset)
    print(f"Saved to {out_dir}")
