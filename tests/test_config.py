import pytest

from llm_eval.config import load_config


def test_baseline_config(baseline_config, root):
    assert baseline_config.name == "baseline"
    assert baseline_config.dataset == root / "datasets" / "golden.yaml"
    assert baseline_config.provider.name == "mock"
    assert baseline_config.gate.max["pii_leak_rate"] == 0.0


def test_extends_merges_and_resolves_paths(degraded_config, baseline_config):
    assert degraded_config.name == "candidate-degraded"
    assert degraded_config.top_k == 1
    assert degraded_config.redact_pii is False
    # Inherited from eval.yaml, and resolved relative to that file rather than configs/.
    assert degraded_config.dataset == baseline_config.dataset
    assert degraded_config.provider.options["refusal_threshold"] == 0.4
    assert degraded_config.provider.options["hallucination_rate"] == 0.3
    assert degraded_config.gate.min == baseline_config.gate.min


def test_env_var_overrides_provider_and_drops_mock_options(monkeypatch, root):
    monkeypatch.setenv("LLM_EVAL_PROVIDER", "openai")
    config = load_config(root / "configs" / "candidate-degraded.yaml")
    assert config.provider.name == "openai"
    assert config.provider.model is None
    assert config.provider.options == {}


def test_circular_extends_is_detected(tmp_path):
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n")
    with pytest.raises(ValueError, match="Circular"):
        load_config(tmp_path / "a.yaml")


def test_missing_required_key(tmp_path):
    path = tmp_path / "c.yaml"
    path.write_text("dataset: d.yaml\n")
    with pytest.raises(ValueError, match="corpus"):
        load_config(path)
