#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "[dev-reset] uninstalling previously installed n2n-redactor if present..."
pip uninstall -y n2n-redactor >/dev/null 2>&1 || true

echo "[dev-reset] installing editable checkout..."
pip install -e . >/dev/null

echo "[dev-reset] verifying import path..."
python -c "import n2n, pathlib; print(pathlib.Path(n2n.__file__).resolve())"
