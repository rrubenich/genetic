"""Zero-shot variant scoring: how much less (or more) likely the model finds a sequence
once a single base is changed.

    LLR = log P(alt window) - log P(ref window)

The window is the hg19 sequence around the variant. A negative LLR means the alternate
allele makes the sequence look less like the genome the model learned from. This is a
learning exercise: a score is not a clinical interpretation of any variant.

Usage: python -m genetic_model.score_variant CHROM POS REF ALT [--model runs/tiny]
       e.g. python -m genetic_model.score_variant chr22 30000000 A G
Positions are 1-based on GRCh37/hg19, alleles on the + strand.
"""

import argparse
from dataclasses import dataclass
from pathlib import Path

import torch
from pyfaidx import Fasta
from transformers import AutoModelForCausalLM, AutoTokenizer

from .config import DEFAULT_CONFIG, load_config, resolve
from .evaluate import default_device


@dataclass
class VariantScore:
    chrom: str
    pos: int
    ref: str
    alt: str
    llr: float          # log P(alt window) - log P(ref window), in nats
    llr_at_site: float  # the same difference, at the variant base only
    window_start: int   # 1-based position of the first base in the window


def normalize_chrom(chrom: str) -> str:
    return chrom if chrom.startswith("chr") else f"chr{chrom}"


def reference_window(fasta_dir: Path, chrom: str, pos: int, size: int) -> tuple[str, int]:
    """`size` bases of hg19 around 1-based `pos`, uppercased. Returns (sequence, 0-based start).

    The variant sits in the middle, or nearer the edge at the ends of a chromosome.
    """
    record = Fasta(str(fasta_dir / f"{chrom}.fa"))[chrom]
    if not 1 <= pos <= len(record):
        raise ValueError(f"{chrom}:{pos} is outside the chromosome (length {len(record)})")
    size = min(size, len(record))
    start = min(max(0, pos - 1 - size // 2), len(record) - size)
    return record[start:start + size].seq.upper(), start


@torch.no_grad()
def per_base_log_probs(model, tokenizer, sequences: list[str]) -> torch.Tensor:
    """log P(base_i | [BOS], base_0..i-1) for every base, shape (n_sequences, length)."""
    ids = tokenizer(sequences, add_special_tokens=False, return_tensors="pt")["input_ids"]
    bos = torch.full((len(ids), 1), tokenizer.bos_token_id)
    inputs = torch.cat([bos, ids], dim=1)[:, :-1].to(model.device)
    logp = torch.log_softmax(model(input_ids=inputs).logits.float(), dim=-1)
    return logp.gather(-1, ids.to(model.device).unsqueeze(-1)).squeeze(-1).cpu()


def score_snv(model, tokenizer, fasta_dir: Path, chrom: str, pos: int, ref: str, alt: str,
              window: int = 1024) -> VariantScore:
    chrom, ref, alt = normalize_chrom(chrom), ref.upper(), alt.upper()
    if len(ref) != 1 or len(alt) != 1 or not {ref, alt} <= set("ACGT") or ref == alt:
        raise ValueError("only single-base substitutions (SNVs) with REF != ALT in A/C/G/T are supported")

    ref_seq, start = reference_window(fasta_dir, chrom, pos, window)
    site = pos - 1 - start
    if ref_seq[site] != ref:
        raise ValueError(
            f"REF {ref} does not match hg19 {chrom}:{pos} ({ref_seq[site]}). "
            "Check the genome build (GRCh37) and that alleles are on the + strand."
        )
    alt_seq = ref_seq[:site] + alt + ref_seq[site + 1:]

    logp = per_base_log_probs(model, tokenizer, [ref_seq, alt_seq])
    return VariantScore(
        chrom=chrom, pos=pos, ref=ref, alt=alt,
        llr=(logp[1].sum() - logp[0].sum()).item(),
        llr_at_site=(logp[1, site] - logp[0, site]).item(),
        window_start=start + 1,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("chrom")
    parser.add_argument("pos", type=int)
    parser.add_argument("ref")
    parser.add_argument("alt")
    parser.add_argument("--model", default="runs/tiny")
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    cli = parser.parse_args()

    config = load_config(cli.config)
    model_dir = resolve(cli.model)
    model = AutoModelForCausalLM.from_pretrained(model_dir).to(default_device()).eval()
    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    score = score_snv(model, tokenizer, resolve(config["data"]["fasta_dir"]),
                      cli.chrom, cli.pos, cli.ref, cli.alt, window=config["data"]["window"])
    print(f"{score.chrom}:{score.pos} {score.ref}>{score.alt}")
    print(f"  LLR (window):  {score.llr:+.4f} nats")
    print(f"  LLR (at site): {score.llr_at_site:+.4f} nats")
