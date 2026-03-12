# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

from vllm.v1.specsteer import initialize_specsteer_models


def test_initialize_specsteer_models_reuses_model_when_base_missing():
    loaded: list[str] = []

    def load_model_fn(name: str):
        loaded.append(name)
        return {"name": name}

    spec_config = SimpleNamespace(model="augmented-slm", base_model=None)

    augmented_model, base_model, base_name = initialize_specsteer_models(
        spec_config, load_model_fn
    )

    assert loaded == ["augmented-slm"]
    assert augmented_model is base_model
    assert base_name == "augmented-slm"


def test_initialize_specsteer_models_reuses_model_when_base_matches_augmented():
    loaded: list[str] = []

    def load_model_fn(name: str):
        loaded.append(name)
        return {"name": name}

    spec_config = SimpleNamespace(
        model="augmented-slm", base_model="augmented-slm"
    )

    augmented_model, base_model, base_name = initialize_specsteer_models(
        spec_config, load_model_fn
    )

    assert loaded == ["augmented-slm"]
    assert augmented_model is base_model
    assert base_name == "augmented-slm"


def test_initialize_specsteer_models_loads_distinct_base_model():
    loaded: list[str] = []

    def load_model_fn(name: str):
        loaded.append(name)
        return {"name": name}

    spec_config = SimpleNamespace(
        model="augmented-slm", base_model="base-verifier"
    )

    augmented_model, base_model, base_name = initialize_specsteer_models(
        spec_config, load_model_fn
    )

    assert loaded == ["augmented-slm", "base-verifier"]
    assert augmented_model is not base_model
    assert base_name == "base-verifier"


def test_initialize_specsteer_models_requires_augmented_model():
    spec_config = SimpleNamespace(model=None, base_model=None)

    try:
        initialize_specsteer_models(spec_config, lambda _: None)
    except ValueError as exc:
        assert "speculative_config.model" in str(exc)
    else:
        raise AssertionError("Expected ValueError when model is missing")
