#!/usr/bin/env bash
# scripts/quick_test.sh
# ----------------------
# Smoke-test: 5 communication rounds, hh-rlhf dataset only.
# Useful for CI or a first sanity check on a new machine.

set -euo pipefail

echo "=== Distributed DPO — Quick Smoke Test ==="
echo "Rounds : 5"
echo "Dataset: hh-rlhf"
echo ""

python main.py \
  --rounds 5 \
  --datasets hh-rlhf \
  --output ./ddpo_smoke_test

echo ""
echo "✅ Smoke test passed. Outputs in ./ddpo_smoke_test/"
