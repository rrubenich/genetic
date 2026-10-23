from genetic_model.config import MODEL_DIR, model_source


def test_model_source_prefers_an_existing_local_directory():
    assert model_source("configs") == str(MODEL_DIR / "configs")


def test_model_source_falls_back_to_a_hub_id():
    assert model_source("rubenich/genetic-dna-tiny") == "rubenich/genetic-dna-tiny"
