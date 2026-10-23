import torch

from genetic_model.config import load_config
from genetic_model.tokenizer import build_hf_tokenizer
from genetic_model.train import build_model


def test_tiny_model_matches_the_plan():
    config = load_config()
    tokenizer = build_hf_tokenizer()
    model = build_model(config, tokenizer)

    assert model.config.vocab_size == 16  # 9 tokens padded to a multiple of 8
    assert model.config.max_position_embeddings == config["data"]["window"] + 2
    assert model.config.pad_token_id == tokenizer.pad_token_id == 0
    assert 5e6 < model.num_parameters() < 8e6  # ~6M


def test_forward_pass_returns_a_loss():
    config = load_config()
    tokenizer = build_hf_tokenizer()
    model = build_model(config, tokenizer)
    batch = tokenizer(["ACGTACGT", "TTGCA"], padding=True, return_tensors="pt")
    labels = batch["input_ids"].masked_fill(batch["attention_mask"] == 0, -100)

    out = model(**batch, labels=labels)
    assert out.logits.shape == (2, batch["input_ids"].shape[1], 16)
    # An untrained model is close to a uniform guess over 16 rows: ln(16) ~ 2.77
    assert torch.isfinite(out.loss) and 2.0 < out.loss.item() < 3.5
