"""Evaluation: bits per base of the model vs. k-th order Markov chain baselines.

Bits per base = average -log2 P(true base). A uniform guess over A/C/G/T scores 2.0;
lower is better. All methods are scored on the same validation bases (chr21 windows
without any N), each base predicted from the bases before it in its window.

Usage: python -m genetic_model.evaluate [config.yaml] [--model runs/tiny] [--max-windows N]
"""

import argparse
import json
import math

import numpy as np
import torch
from datasets import load_from_disk
from transformers import AutoModelForCausalLM

from .config import DEFAULT_CONFIG, load_config, resolve
from .vocabulary import VOCAB

BASE_IDS = np.array([VOCAB[b] for b in "ACGT"])  # token ids of A, C, G, T
MARKOV_ORDERS = [0, 1, 2, 3, 4, 5, 6, 8, 10]


def to_base_windows(input_ids: list[list[int]]) -> np.ndarray:
    """Token-id windows -> (n_windows, window) array of 0..3, dropping windows with any N.

    Strips [BOS]/[EOS]. Markov counts need a closed alphabet, so N windows are skipped
    for every method alike.
    """
    ids = np.asarray(input_ids, dtype=np.int64)[:, 1:-1]
    keep = np.isin(ids, BASE_IDS).all(axis=1)
    lookup = np.full(ids.max() + 1, -1, dtype=np.int8)  # int8 keeps the train split ~70 MB
    lookup[BASE_IDS] = np.arange(4)
    return lookup[ids[keep]]


def _contexts(x: np.ndarray, order: int) -> np.ndarray:
    """Context index of each position i >= order: the previous `order` bases in base 4."""
    idx = np.zeros((x.shape[0], x.shape[1] - order), dtype=np.int64)
    for m in range(1, order + 1):
        idx += x[:, order - m: x.shape[1] - m].astype(np.int64) * 4 ** (m - 1)
    return idx


class MarkovModel:
    """k-th order Markov chain with add-alpha smoothing.

    The first k bases of a window have less than k bases of context, so position i < k
    falls back to the order-i table (the same left-context-only setup as the model).
    """

    def __init__(self, max_order: int, alpha: float = 1.0):
        self.max_order = max_order
        self.alpha = alpha
        self.counts = [np.zeros(4 ** (k + 1), dtype=np.int64) for k in range(max_order + 1)]

    def fit(self, x: np.ndarray, chunk: int = 4096) -> "MarkovModel":
        for start in range(0, len(x), chunk):
            part = x[start:start + chunk]
            for k in range(self.max_order + 1):
                keys = _contexts(part, k) * 4 + part[:, k:]  # int64 + int8 -> int64
                self.counts[k] += np.bincount(keys.ravel(), minlength=4 ** (k + 1))
        return self

    def log2_probs(self, order: int) -> np.ndarray:
        c = self.counts[order].reshape(-1, 4) + self.alpha
        return np.log2(c / c.sum(axis=1, keepdims=True)).ravel()

    def bits_per_base(self, x: np.ndarray, order: int) -> float:
        total = 0.0
        for i in range(min(order, x.shape[1])):  # window start: shorter context
            keys = _contexts(x[:, :i + 1], i)[:, -1] * 4 + x[:, i]
            total -= self.log2_probs(i)[keys].sum()
        keys = _contexts(x, order) * 4 + x[:, order:]
        total -= self.log2_probs(order)[keys].sum()
        return total / x.size


@torch.no_grad()
def model_bits_per_base(model, x: np.ndarray, batch_size: int = 16, device: str = "cpu") -> float:
    """Average -log2 P(base) under the model, predicting each base from [BOS] + previous bases.

    The probability comes from the full softmax, so mass the model puts on N or special
    tokens counts against it. The [EOS] prediction is not scored.
    """
    model.to(device).eval()
    tokens = torch.from_numpy(BASE_IDS[x])
    total = 0.0
    for start in range(0, len(tokens), batch_size):
        batch = tokens[start:start + batch_size].to(device)
        inputs = torch.cat([torch.full((len(batch), 1), VOCAB["[BOS]"], device=device), batch], dim=1)
        logits = model(input_ids=inputs[:, :-1]).logits.float()
        logp = torch.log_softmax(logits, dim=-1).gather(-1, batch.unsqueeze(-1))
        total -= logp.sum().item()
    return total / x.size / math.log(2)


def default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    return "mps" if torch.backends.mps.is_available() else "cpu"


def evaluate(config: dict, model_dir: str, max_windows: int | None = None) -> dict:
    dataset = load_from_disk(resolve(config["data"]["out_dir"]))
    train = to_base_windows(dataset["train"]["input_ids"])
    val = to_base_windows(dataset["validation"]["input_ids"])
    if max_windows:
        val = val[np.random.default_rng(0).permutation(len(val))[:max_windows]]
    print(f"Scoring {val.size:,} validation bases ({len(val):,} windows without N)")

    results = {"validation_bases": int(val.size), "uniform": 2.0, "markov": {}}
    markov = MarkovModel(max(MARKOV_ORDERS)).fit(train)
    for k in MARKOV_ORDERS:
        results["markov"][k] = markov.bits_per_base(val, k)
        print(f"Markov order {k:>2}: {results['markov'][k]:.4f} bits/base")

    model = AutoModelForCausalLM.from_pretrained(resolve(model_dir))
    results["model"] = model_bits_per_base(model, val, device=default_device())
    print(f"Model ({model_dir}): {results['model']:.4f} bits/base")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("config", nargs="?", default=DEFAULT_CONFIG)
    parser.add_argument("--model", default="runs/tiny")
    parser.add_argument("--max-windows", type=int, help="score a random subset of validation windows")
    cli = parser.parse_args()

    results = evaluate(load_config(cli.config), cli.model, cli.max_windows)
    out = resolve(cli.model) / "eval_results.json"
    out.write_text(json.dumps(results, indent=2) + "\n")
    print(f"Saved {out}")
