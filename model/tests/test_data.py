import pytest

from genetic_model.data import build_dataset, n_fraction, reverse_complement, windows
from genetic_model.tokenizer import build_hf_tokenizer
from genetic_model.vocabulary import VOCAB


def test_reverse_complement():
    assert reverse_complement("AACGTN") == "NACGTT"
    assert reverse_complement(reverse_complement("ACGGTCAN")) == "ACGGTCAN"


def test_n_fraction():
    assert n_fraction("ACGN") == 0.25
    assert n_fraction("ACGT") == 0.0


def test_windows_are_non_overlapping_and_drop_the_tail():
    assert list(windows("ACGTACGTAC", size=4, max_n_fraction=0.0)) == [(0, "ACGT"), (4, "ACGT")]


def test_windows_uppercase_soft_masked_bases():
    assert list(windows("acgt", size=4, max_n_fraction=0.0)) == [(0, "ACGT")]


def test_windows_drop_gap_heavy_windows():
    seq = "ACGT" + "ANNN" + "ACGN"
    assert [s for s, _ in windows(seq, size=4, max_n_fraction=0.25)] == [0, 8]


@pytest.fixture
def tiny_config(tmp_path):
    fasta_dir = tmp_path / "hg19"
    fasta_dir.mkdir()
    (fasta_dir / "chrA.fa").write_text(">chrA\nACGTacgtNNNNAACC\n")
    (fasta_dir / "chrB.fa").write_text(">chrB\nGGGGTTTT\n")
    tok_dir = tmp_path / "tokenizer"
    build_hf_tokenizer().save_pretrained(tok_dir)
    return {
        "tokenizer_dir": str(tok_dir),
        "data": {
            "fasta_dir": str(fasta_dir),
            "train_chroms": ["chrA"],
            "validation_chroms": ["chrB"],
            "window": 4,
            "max_n_fraction": 0.0,
            "reverse_complement_train": True,
        },
    }


def test_build_dataset(tiny_config):
    ds = build_dataset(tiny_config)
    train, val = ds["train"], ds["validation"]

    # chrA: 4 windows, the NNNN one is dropped, each kept window also gets its reverse complement
    assert list(zip(train["start"], train["strand"])) == [
        (0, "+"), (0, "-"), (4, "+"), (4, "-"), (12, "+"), (12, "-"),
    ]
    # validation is split by chromosome and has no reverse complement
    assert val["chrom"] == ["chrB", "chrB"] and set(val["strand"]) == {"+"}

    bos, eos = VOCAB["[BOS]"], VOCAB["[EOS]"]
    A, C, G, T = (VOCAB[b] for b in "ACGT")
    assert train[0]["input_ids"] == [bos, A, C, G, T, eos]
    assert train[5]["input_ids"] == [bos, G, G, T, T, eos]  # reverse complement of AACC
    assert all(len(ids) == 4 + 2 for ids in train["input_ids"])
