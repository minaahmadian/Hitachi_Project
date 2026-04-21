#!/usr/bin/env bash
# Run all three eval scenarios (same email/logs/docx; only CSV changes).
# Requires: VDD_AUDIT=1, GROQ_API_KEY, venv activated. From repo root.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

export VDD_AUDIT=1

run_one() {
  local name="$1"
  local csv="$2"
  local golden="$3"
  local out="$ROOT/output/vdd_last_run_${name}.json"
  export REQUIREMENTS_TRACE_CSV="$csv"
  echo "========== Scenario ${name} =========="
  echo "REQUIREMENTS_TRACE_CSV=${csv}"
  python main.py
  if [[ -f "$ROOT/output/vdd_last_run.json" ]]; then
    cp -f "$ROOT/output/vdd_last_run.json" "$out"
    echo "Saved audit copy to ${out}"
  fi
  python scripts/evaluate_against_golden.py \
    --audit "$ROOT/output/vdd_last_run.json" \
    --golden "$golden"
  echo ""
}

run_one "A" "data/eval/scenarios/scenario_a_hit_fake.csv" "data/eval/golden/scenario_a.json"
run_one "B" "data/eval/scenarios/scenario_b_no_hit_fake.csv" "data/eval/golden/scenario_b.json"
run_one "C" "data/eval/scenarios/scenario_c_extra_conflict_row.csv" "data/eval/golden/scenario_c.json"

echo "All scenarios finished."
