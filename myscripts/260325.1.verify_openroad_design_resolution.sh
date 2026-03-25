#!/usr/bin/env bash
set -euo pipefail

# Quick verifier for OpenROAD design alias resolution.
# Usage:
#   bash myscripts/000001.verify_openroad_design_resolution.sh

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

echo "[INFO] Verifying alias -> resolved design mapping (no make run)"
for d in ariane133 ariane136 swerv_wrapper bp bp_be bp_fe bp_be_top black_parrot; do
  echo "\n=== design=${d} ==="
  python "${ROOT_DIR}/src/utils/openroad_evaluator.py" \
    --def_path "${ROOT_DIR}/README.md" \
    --design "${d}" \
    --platform nangate45 \
    --variant eval_xp \
    --resolve_only || true
done

echo "\n[DONE] Resolution verification finished"
