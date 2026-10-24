PYTHON ?= python3
VENV   ?= .venv
BIN     = $(VENV)/bin

.PHONY: install lint format test eval gate compare demo-regression baseline report clean

install:  ## create a virtualenv and install with dev tools
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install --upgrade pip
	$(BIN)/pip install -e ".[dev]"

lint:
	$(BIN)/ruff check .
	$(BIN)/ruff format --check .

format:
	$(BIN)/ruff format .
	$(BIN)/ruff check --fix .

test:
	$(BIN)/pytest

eval:  ## run the golden dataset with the baseline config
	$(BIN)/llm-eval run -o reports/latest

gate:  ## run and enforce the absolute thresholds
	$(BIN)/llm-eval run --gate -o reports/latest

compare: eval  ## compare the current run against the committed baseline
	$(BIN)/llm-eval compare baselines/baseline.json reports/latest/results.json -o reports/compare

demo-regression:  ## show the gate failing on a deliberately degraded candidate
	$(BIN)/llm-eval run -c configs/candidate-degraded.yaml -o reports/degraded
	-$(BIN)/llm-eval compare baselines/baseline.json reports/degraded/results.json -o reports/compare-degraded

baseline:  ## regenerate the committed baseline (review the diff before committing)
	$(BIN)/llm-eval run -o reports/baseline
	cp reports/baseline/results.json baselines/baseline.json

clean:
	rm -rf reports .pytest_cache .ruff_cache .coverage coverage.xml htmlcov
