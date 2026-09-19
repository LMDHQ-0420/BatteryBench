#!/usr/bin/env bash
# 一次执行：创建/更新 zw@BatteryBench 环境，然后启动调度。
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

# 预演不安装依赖、不创建环境、不启动实验。
for argument in "$@"; do
    if [[ "${argument}" == "--dry-run" ]]; then
        exec bash run_pipeline.sh "$@"
    fi
done

mkdir -p pipeline_state
exec 9>pipeline_state/pipeline.lock
if ! flock -n 9; then
    echo "已有调度或环境安装正在运行，请勿同时修改环境。" >&2
    exit 1
fi
CONDA_EXE_PATH="${CONDA_EXE:-$(command -v conda || true)}"
if [[ -z "${CONDA_EXE_PATH}" ]]; then
    echo "未找到 conda，请先安装 conda。" >&2
    exit 1
fi
CONDA_BASE_PATH="$("${CONDA_EXE_PATH}" info --base 2>/dev/null)"
ENV_PREFIX="${CONDA_BASE_PATH}/envs/zw@BatteryBench"
PYTHON_VERSION="$(sed -n 's/^# Python: //p' requirements.txt)"
if [[ -z "${PYTHON_VERSION}" ]]; then
    echo "requirements.txt 缺少 # Python: 版本声明。" >&2
    exit 1
fi
if [[ ! -x "${ENV_PREFIX}/bin/python" ]]; then
    CONDA_NO_PLUGINS=true CONDA_SOLVER=classic "${CONDA_EXE_PATH}" create \
        --prefix "${ENV_PREFIX}" "python=${PYTHON_VERSION}" pip --yes --solver classic
fi
"${ENV_PREFIX}/bin/python" -c \
    "import sys; assert '.'.join(map(str, sys.version_info[:2])) == '${PYTHON_VERSION}', '环境 Python 版本与 requirements.txt 不一致'"
"${ENV_PREFIX}/bin/python" -m pip install -r requirements.txt
"${ENV_PREFIX}/bin/python" -m pip check
flock -u 9
exec 9>&-
export BATTERYBENCH_PYTHON="${ENV_PREFIX}/bin/python"
exec bash run_pipeline.sh "$@"
