#!/usr/bin/env bash
# Reproduce all quantitative results and figures of the paper (same as `etf reproduce`).
#
#   bash scripts/reproduce_all.sh [etf reproduce options, e.g. --outroot DIR --saliency]
#
# The environment switches of version 1.x still work: OUTROOT=DIR, RUN_SALIENCY=1,
# SALIENCY_MODEL=<model>, RUN_CONTINUOUS_POWER=0, RUN_CLIPLEN_SENSITIVITY=0,
# RUN_STAGE_ERROR_BINS=0, RUN_ANCHOR_SENSITIVITY=0 and PYTHON_BIN (also read from .env).
set -euo pipefail
cd "$(dirname "$0")/.."

args=()
[ -n "${OUTROOT:-}" ] && args+=(--outroot "$OUTROOT")
skip=()
[ "${RUN_CONTINUOUS_POWER:-1}" = 0 ] && skip+=(power-continuous)
[ "${RUN_CLIPLEN_SENSITIVITY:-1}" = 0 ] && skip+=(cliplen)
[ "${RUN_STAGE_ERROR_BINS:-1}" = 0 ] && skip+=(stage-error)
[ "${RUN_ANCHOR_SENSITIVITY:-1}" = 0 ] && skip+=(anchor)
[ ${#skip[@]} -gt 0 ] && args+=(--skip "$(IFS=,; echo "${skip[*]}")")
[ "${RUN_SALIENCY:-0}" = 1 ] && args+=(--saliency --saliency_model "${SALIENCY_MODEL:-full}")
[ -n "${CLIPLEN_OUTROOT:-}" ] && echo "[WARN] CLIPLEN_OUTROOT is ignored; results go to <outroot>/cliplen_sensitivity" >&2

PY="${PYTHON_BIN:-}"
if [ -z "$PY" ] && [ -f .env ]; then
  PY="$(sed -n 's/^[[:space:]]*PYTHON_BIN[[:space:]]*=[[:space:]]*//p' .env | tail -n 1 | tr -d "\"'")"
fi
PY="${PY:-$(command -v python3 || command -v python)}"

# Always run the code of this checkout (with or without `pip install -e .`).
PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}" exec "$PY" -m etf reproduce ${args[@]+"${args[@]}"} "$@"
