import json

from genetic_model.config import load_config
from genetic_model.publish import write_card
from genetic_model.tokenizer import build_hf_tokenizer
from genetic_model.train import build_model


def test_write_card(tmp_path):
    config = load_config()
    build_model(config, build_hf_tokenizer()).save_pretrained(tmp_path)
    (tmp_path / "eval_results.json").write_text(json.dumps({
        "validation_bases": 1234, "uniform": 2.0, "model": 1.5, "markov": {"0": 1.99, "10": 1.83},
    }))

    card = write_card(config, tmp_path, "someone/genetic-dna-tiny").read_text()

    assert card.startswith("---\nlicense: mit\n")
    assert "# genetic-dna-tiny" in card
    assert "| **This model** | **1.5000** |" in card
    assert "| Markov order 10 | 1.8300 |" in card
    assert 'from_pretrained("someone/genetic-dna-tiny")' in card
    assert "Not for clinical use" in card
    assert "{" not in card.split("```python")[0]  # every placeholder before the code block was filled
