#!/usr/bin/env bash
set -euo pipefail
LEG="${1:-main}"
python -m chronorisk.bridge train --leg "${LEG}" --out "runs/${LEG}.pt"
