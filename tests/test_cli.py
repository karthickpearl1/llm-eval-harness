import json
import runpy
import sys

import pytest

from llm_eval.cli import EXIT_ERROR, EXIT_GATE_FAILED, EXIT_OK, main


@pytest.fixture
def in_repo(root, monkeypatch):
    monkeypatch.chdir(root)
    return root


def test_run_with_gate_passes_for_baseline(in_repo, tmp_path, capsys):
    assert main(["run", "--gate", "-o", str(tmp_path / "run")]) == EXIT_OK
    out = capsys.readouterr().out
    assert "Quality gate: PASSED" in out
    assert (tmp_path / "run" / "report.html").exists()


def test_run_with_gate_fails_for_degraded_candidate(in_repo, tmp_path):
    args = ["run", "-c", "configs/candidate-degraded.yaml", "--gate", "-o", str(tmp_path)]
    assert main(args) == EXIT_GATE_FAILED


def test_run_without_gate_always_succeeds(in_repo, tmp_path):
    args = ["run", "-c", "configs/candidate-degraded.yaml", "-o", str(tmp_path)]
    assert main(args) == EXIT_OK
    assert not (tmp_path / "gate.json").exists()


def test_compare_fails_on_regression_and_writes_summary(in_repo, tmp_path, capsys):
    main(["run", "-c", "configs/candidate-degraded.yaml", "-o", str(tmp_path / "cand")])
    summary = tmp_path / "summary.md"
    code = main(
        [
            "compare",
            "baselines/baseline.json",
            str(tmp_path / "cand" / "results.json"),
            "-o",
            str(tmp_path / "cmp"),
            "--summary-md",
            str(summary),
        ]
    )
    assert code == EXIT_GATE_FAILED
    assert "Quality gate: FAILED" in capsys.readouterr().out
    assert "Quality gate: FAILED" in summary.read_text()
    assert json.loads((tmp_path / "cmp" / "gate.json").read_text())["new_failures"]


def test_compare_passes_for_harmless_change(in_repo, tmp_path):
    main(["run", "-c", "configs/candidate-prompt-tweak.yaml", "-o", str(tmp_path / "cand")])
    code = main(
        [
            "compare",
            "baselines/baseline.json",
            str(tmp_path / "cand" / "results.json"),
            "-o",
            str(tmp_path),
        ]
    )
    assert code == EXIT_OK


def test_baseline_compared_with_itself_passes(in_repo, tmp_path):
    args = ["compare", "baselines/baseline.json", "baselines/baseline.json", "-o", str(tmp_path)]
    assert main(args) == EXIT_OK


def test_report_rerenders(in_repo, tmp_path):
    assert main(["report", "baselines/baseline.json", "-o", str(tmp_path)]) == EXIT_OK
    assert (tmp_path / "report.md").exists()


def test_errors_exit_with_2(in_repo, tmp_path, capsys):
    assert main(["run", "-c", "does-not-exist.yaml"]) == EXIT_ERROR
    bogus = tmp_path / "bogus.json"
    bogus.write_text("{}")
    assert main(["report", str(bogus)]) == EXIT_ERROR
    assert "error:" in capsys.readouterr().err


def test_module_entry_point(in_repo, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["llm-eval", "--version"])
    with pytest.raises(SystemExit) as exc:
        runpy.run_module("llm_eval", run_name="__main__")
    assert exc.value.code == 0
