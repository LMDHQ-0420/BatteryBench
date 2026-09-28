#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
PYTHON="${BATTERYBENCH_PYTHON:-python}"
exec "${PYTHON}" -u scripts/run_pipeline.py --domains calb "$@"
