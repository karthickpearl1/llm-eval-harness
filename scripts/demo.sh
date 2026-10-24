#!/usr/bin/env bash
# Replays the README demo: a passing baseline run, then the gate rejecting a degraded candidate.
# Used to record docs/demo.cast (asciinema rec -c "bash scripts/demo.sh" docs/demo.cast).
set -u

type_cmd() {
  printf '\033[1;32m$\033[0m '
  local text="$1"
  for ((i = 0; i < ${#text}; i++)); do
    printf '%s' "${text:i:1}"
    sleep 0.02
  done
  printf '\n'
  sleep 0.4
}

run() {
  type_cmd "$1"
  eval "$1"
  sleep 1.5
}

run "llm-eval run --gate -o reports/baseline-run"
run "llm-eval run -c configs/candidate-degraded.yaml -o reports/degraded > /dev/null"
run 'llm-eval compare baselines/baseline.json reports/degraded/results.json -o reports/compare; echo "exit code: $?"'
sleep 1
