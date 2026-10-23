import random

import pytest
from tokenizers import Tokenizer
from transformers import AutoTokenizer

from genetic_model.tokenizer import build_hf_tokenizer, build_tokenizer
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


# transformers wrapper: what the dataset and Trainer will actually use.

@pytest.fixture(scope="module")
def hf_tok():
    return build_hf_tokenizer()


def test_hf_special_token_ids(hf_tok):
    assert hf_tok.pad_token_id == VOCAB["[PAD]"] == 0
    assert hf_tok.unk_token_id == VOCAB["[UNK]"]
    assert hf_tok.bos_token_id == VOCAB["[BOS]"]
    assert hf_tok.eos_token_id == VOCAB["[EOS]"]
    assert len(hf_tok) == 9


def test_hf_matches_raw_tokenizer(hf_tok, tok):
    assert hf_tok("acgtR")["input_ids"] == tok.encode("acgtR").ids


def test_hf_padding(hf_tok):
    batch = hf_tok(["ACG", "A"], padding=True)
    assert batch["input_ids"] == [[2, 4, 5, 6, 3], [2, 4, 3, 0, 0]]
    assert batch["attention_mask"] == [[1, 1, 1, 1, 1], [1, 1, 1, 0, 0]]


def test_hf_decode_skips_special_tokens(hf_tok):
    assert hf_tok.decode(hf_tok("ACGT")["input_ids"], skip_special_tokens=True) == "ACGT"


def test_save_and_reload_with_autotokenizer(hf_tok, tmp_path):
    hf_tok.save_pretrained(tmp_path)
    reloaded = AutoTokenizer.from_pretrained(tmp_path)
    assert reloaded.get_vocab() == hf_tok.get_vocab()
    assert reloaded.pad_token_id == 0
    assert reloaded(["acgtR", "A"], padding=True)["input_ids"] == hf_tok(["acgtR", "A"], padding=True)["input_ids"]
