#!/usr/bin/env bash
set -euo pipefail
LEG="${1:-main}"
CKPT="${2:-runs/${LEG}.pt}"
python -m chronorisk.bridge evaluate --leg "${LEG}" --checkpoint "${CKPT}"
