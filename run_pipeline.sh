#!/usr/bin/env bash
# 按预定阶段使用 GPU 0/1/2/3 训练并测评；BatLiNet 最后运行且不使用 GPU 0。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"
CONDA_ENV="zw@BatteryBench"
if [[ -n "${BATTERYBENCH_PYTHON:-}" ]]; then
    PIPELINE_PYTHON="${BATTERYBENCH_PYTHON}"
else
    CONDA_BASE="$(conda info --base 2>/dev/null || true)"
    CONDA_BASE="${CONDA_BASE:-${HOME}/anaconda3}"
    PIPELINE_PYTHON="${CONDA_BASE}/envs/${CONDA_ENV}/bin/python"
fi
if [[ ! -x "${PIPELINE_PYTHON}" ]]; then
    echo "找不到 ${CONDA_ENV} 的 Python：${PIPELINE_PYTHON}" >&2
    exit 1
fi
exec "${PIPELINE_PYTHON}" -u scripts/run_pipeline.py "$@"
