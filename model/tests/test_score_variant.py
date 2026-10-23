import pytest
import torch

from genetic_model.config import load_config
from genetic_model.score_variant import per_base_log_probs, reference_window, score_snv
from genetic_model.tokenizer import build_hf_tokenizer
from genetic_model.train import build_model

# Fake chromosome (no real genome data in tests): 1-based position 5 is "A"
SEQ = "ACGTacgtNNACGTAC"


@pytest.fixture(scope="module")
def fasta_dir(tmp_path_factory):
    d = tmp_path_factory.mktemp("hg19")
    (d / "chrT.fa").write_text(f">chrT\n{SEQ}\n")
    return d


@pytest.fixture(scope="module")
def tok():
    return build_hf_tokenizer()


@pytest.fixture(scope="module")
def model(tok):
    torch.manual_seed(0)
    return build_model(load_config(), tok).eval()


def test_reference_window_is_centered_and_uppercased(fasta_dir):
    seq, start = reference_window(fasta_dir, "chrT", pos=9, size=4)
    assert (seq, start) == ("GTNN", 6)  # 0-based 8 sits at index 2


def test_reference_window_shifts_at_chromosome_edges(fasta_dir):
    assert reference_window(fasta_dir, "chrT", pos=1, size=4) == ("ACGT", 0)
    assert reference_window(fasta_dir, "chrT", pos=16, size=4) == ("GTAC", 12)
    assert reference_window(fasta_dir, "chrT", pos=3, size=100) == (SEQ.upper(), 0)


def test_position_outside_chromosome(fasta_dir):
    with pytest.raises(ValueError, match="outside"):
        reference_window(fasta_dir, "chrT", pos=17, size=4)


def test_ref_must_match_the_genome(model, tok, fasta_dir):
    with pytest.raises(ValueError, match="does not match"):
        score_snv(model, tok, fasta_dir, "chrT", 5, ref="G", alt="T", window=8)


@pytest.mark.parametrize("ref,alt", [("A", "A"), ("A", "AT"), ("A", "R")])
def test_only_snvs(model, tok, fasta_dir, ref, alt):
    with pytest.raises(ValueError, match="SNV"):
        score_snv(model, tok, fasta_dir, "chrT", 5, ref=ref, alt=alt, window=8)


def test_llr_is_the_log_prob_difference(model, tok, fasta_dir):
    score = score_snv(model, tok, fasta_dir, "T", 5, ref="a", alt="g", window=8)  # "T" -> "chrT"
    assert (score.chrom, score.ref, score.alt, score.window_start) == ("chrT", "A", "G", 1)

    ref_seq = SEQ[:8].upper()            # ACGTACGT, variant at index 4
    alt_seq = ref_seq[:4] + "G" + ref_seq[5:]
    logp = per_base_log_probs(model, tok, [ref_seq, alt_seq])
    assert score.llr == pytest.approx((logp[1].sum() - logp[0].sum()).item(), abs=1e-5)
    assert score.llr_at_site == pytest.approx((logp[1, 4] - logp[0, 4]).item(), abs=1e-5)
    # bases before the variant see the same context, so their log probs are identical
    assert torch.allclose(logp[0, :4], logp[1, :4])


def test_per_base_log_probs_shape(model, tok):
    logp = per_base_log_probs(model, tok, ["ACGT", "TTTT"])
    assert logp.shape == (2, 4) and (logp <= 0).all()
