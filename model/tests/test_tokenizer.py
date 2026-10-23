import random

import pytest
from tokenizers import Tokenizer

from genetic_model.tokenizer import build_tokenizer
from genetic_model.vocabulary import BASES, SPECIAL_TOKENS, VOCAB


@pytest.fixture(scope="module")
def tok() -> Tokenizer:
    return build_tokenizer()


def ids(tok: Tokenizer, seq: str) -> list[int]:
    return tok.encode(seq, add_special_tokens=False).ids


# Vocabulary: ids are part of the model's contract, they must never shift.

def test_vocab_ids_are_fixed(tok):
    assert tok.get_vocab() == VOCAB
    assert tok.get_vocab_size() == len(SPECIAL_TOKENS) + len(BASES) == 9
    assert VOCAB["[PAD]"] == 0


def test_special_tokens_are_marked_special(tok):
    added = {t.content for t in tok.get_added_tokens_decoder().values() if t.special}
    assert added == set(SPECIAL_TOKENS)


# Normalizer: soft-masked lowercase and unknown letters.

def test_lowercase_equals_uppercase(tok):
    assert ids(tok, "acgtn") == ids(tok, "ACGTN")


@pytest.mark.parametrize("char", list("RYSWKMBDHVX-*.n"))
def test_unknown_characters_become_n(tok, char):
    assert ids(tok, char) == [VOCAB["N"]]


def test_never_produces_unk(tok):
    text = "".join(chr(c) for c in range(33, 127))  # every printable ASCII char
    assert VOCAB["[UNK]"] not in ids(tok, text)


# Pre-tokenizer: one base = one token, so a SNV changes exactly one token.

def test_one_token_per_base(tok):
    seq = "".join(random.Random(0).choices("ACGTN", k=1000))
    assert len(ids(tok, seq)) == 1000


def test_snv_changes_exactly_one_token(tok):
    ref = "ACGTACGTAC"
    alt = ref[:5] + "T" + ref[6:]  # C>T at position 5
    diff = [i for i, (a, b) in enumerate(zip(ids(tok, ref), ids(tok, alt))) if a != b]
    assert diff == [5]


def test_offsets_point_to_each_base(tok):
    enc = tok.encode("ACGT", add_special_tokens=False)
    assert enc.offsets == [(0, 1), (1, 2), (2, 3), (3, 4)]


# Post-processor and decoder.

def test_adds_bos_and_eos(tok):
    assert tok.encode("ACGT").ids == [VOCAB["[BOS]"], 4, 5, 6, 7, VOCAB["[EOS]"]]


def test_round_trip(tok):
    assert tok.decode(tok.encode("ACGTN").ids) == "ACGTN"


def test_round_trip_normalizes(tok):
    assert tok.decode(tok.encode("acgtR").ids) == "ACGTN"


def test_decode_can_keep_special_tokens(tok):
    assert tok.decode(tok.encode("AC").ids, skip_special_tokens=False) == "[BOS]AC[EOS]"


def test_special_token_text_is_not_normalized(tok):
    assert tok.encode("A[PAD]C", add_special_tokens=False).tokens == ["A", "[PAD]", "C"]


# Serialization: what we save must load back identical.

def test_serialization_round_trip(tok):
    reloaded = Tokenizer.from_str(tok.to_str())
    assert reloaded.get_vocab() == tok.get_vocab()
    assert reloaded.encode("acgtR").ids == tok.encode("acgtR").ids
