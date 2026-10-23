import math

import numpy as np
import pytest

from genetic_model.config import load_config
from genetic_model.evaluate import MarkovModel, _contexts, model_bits_per_base, to_base_windows
from genetic_model.tokenizer import build_hf_tokenizer
from genetic_model.train import build_model
from genetic_model.vocabulary import VOCAB


def test_to_base_windows_strips_specials_and_drops_n_windows():
    tok = build_hf_tokenizer()
    ids = tok(["ACGT", "ACNT", "TTGA"])["input_ids"]
    assert to_base_windows(ids).tolist() == [[0, 1, 2, 3], [3, 3, 2, 0]]


def test_contexts_encode_previous_bases():
    x = np.array([[0, 1, 2, 3]])  # A C G T
    # order 2 at position 2: previous bases C (x[1]=1, weight 1) and A (x[0]=0, weight 4)
    assert _contexts(x, 2).tolist() == [[1 * 1 + 0 * 4, 2 * 1 + 1 * 4]]


def test_order_0_on_uniform_data_is_2_bits():
    x = np.tile([0, 1, 2, 3], (10, 25))
    assert MarkovModel(0).fit(x).bits_per_base(x, 0) == pytest.approx(2.0, abs=1e-3)


def test_order_1_learns_a_periodic_sequence():
    x = np.tile([0, 1, 2, 3], (50, 25))  # ACGTACGT...: each base fully predicts the next
    markov = MarkovModel(1).fit(x)
    assert markov.bits_per_base(x, 1) < 0.1
    assert markov.bits_per_base(x, 1) < markov.bits_per_base(x, 0)


def test_counts_cover_every_position():
    x = np.random.default_rng(0).integers(0, 4, size=(3, 20))
    markov = MarkovModel(3).fit(x)
    for k in range(4):
        assert markov.counts[k].sum() == 3 * (20 - k)


def test_model_bits_per_base_of_an_untrained_model():
    model = build_model(load_config(), build_hf_tokenizer())
    x = np.random.default_rng(0).integers(0, 4, size=(2, 32))
    bits = model_bits_per_base(model, x)
    # roughly a uniform guess over 16 vocab rows: log2(16) = 4
    assert 3.0 < bits < 5.0 and math.isfinite(bits)
    assert VOCAB["A"] == 4  # BASE_IDS assumes A, C, G, T are ids 4..7
