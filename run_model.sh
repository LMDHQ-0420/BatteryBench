#!/usr/bin/env bash
# run_model.sh — 并发训练指定 model 下所有 (domain, task) 组合
# 用法: bash run_model.sh <model> [gpu0 gpu1 ...]
#   例: bash run_model.sh dlinear
#       bash run_model.sh patchtst 0 1
# model 需已在 src/models/registry.py 中注册

set -euo pipefail

MODEL="${1:?用法: bash run_model.sh <model> [gpu ...]}"
shift || true
if [[ $# -gt 0 ]]; then
    GPUS=("$@")
else
    GPUS=(0 1 2 3)
fi

CONDA_ENV="zw@RUL-Predict"
CONDA_BASE=$(conda info --base 2>/dev/null || echo "$HOME/miniconda3")
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate "${CONDA_ENV}"
PYTHON=$(which python)
echo "[$(date '+%H:%M:%S')] Python: ${PYTHON}"

TIMESTAMP=$(date +"%y-%m-%d_%H-%M-%S")
LOG_DIR="log/${TIMESTAMP}_${MODEL}"
mkdir -p "${LOG_DIR}"

SLOTS_PER_GPU=3
TOTAL_SLOTS=$(( ${#GPUS[@]} * SLOTS_PER_GPU ))
JOB_THREADS=3

DOMAINS=(li_ion calb na_ion zn_ion three_level)

JOBS=()
for domain in "${DOMAINS[@]}"; do
    for task in rul soh_point soh_traj; do
        JOBS+=("${domain}|${task}")
    done
done

TOTAL=${#JOBS[@]}
echo "[$(date '+%H:%M:%S')] Model: ${MODEL}  Total jobs: ${TOTAL}"
echo "[$(date '+%H:%M:%S')] Log dir: ${LOG_DIR}"
echo "[$(date '+%H:%M:%S')] GPUs: ${GPUS[*]} (${SLOTS_PER_GPU} slots each = ${TOTAL_SLOTS} concurrent)"

declare -a SLOT_PIDS
declare -a SLOT_GPU
for (( s=0; s<TOTAL_SLOTS; s++ )); do
    SLOT_PIDS[$s]=0
    SLOT_GPU[$s]=${GPUS[$((s / SLOTS_PER_GPU))]}
done

find_free_slot() {
    for (( s=0; s<TOTAL_SLOTS; s++ )); do
        pid=${SLOT_PIDS[$s]}
        if [[ $pid -eq 0 ]]; then echo $s; return; fi
        if ! kill -0 "$pid" 2>/dev/null; then
            SLOT_PIDS[$s]=0
            echo $s; return
        fi
    done
    echo -1
}

wait_for_free_slot() {
    while true; do
        slot=$(find_free_slot)
        if [[ $slot -ge 0 ]]; then echo $slot; return; fi
        sleep 3
    done
}

for job in "${JOBS[@]}"; do
    IFS='|' read -r domain task <<< "$job"

    slot=$(wait_for_free_slot)
    gpu=${SLOT_GPU[$slot]}

    log_file="${LOG_DIR}/${domain}_${task}.log"
    echo "[$(date '+%H:%M:%S')] START  gpu=${gpu} slot=${slot}  ${domain}/${task}/${MODEL}"

    nohup bash -c "
        export OMP_NUM_THREADS=${JOB_THREADS} MKL_NUM_THREADS=${JOB_THREADS}
        ${PYTHON} scripts/train.py \
            --domain '${domain}' --task '${task}' --model '${MODEL}' --gpu '${gpu}' --seed 42 && \
        ${PYTHON} scripts/evaluate.py \
            --domain '${domain}' --task '${task}' --model '${MODEL}' --gpu '${gpu}'
    " > "${log_file}" 2>&1 &

    SLOT_PIDS[$slot]=$!
done

echo ""
echo "[$(date '+%H:%M:%S')] All ${TOTAL} jobs launched. Waiting for completion..."
for (( s=0; s<TOTAL_SLOTS; s++ )); do
    pid=${SLOT_PIDS[$s]}
    if [[ $pid -ne 0 ]]; then
        wait "$pid" 2>/dev/null || true
    fi
done

echo "[$(date '+%H:%M:%S')] All done. Logs: ${LOG_DIR}/"
