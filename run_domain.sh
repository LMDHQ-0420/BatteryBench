#!/usr/bin/env bash
# run_domain.sh — 并发训练指定 domain 下所有 (task, model) 组合
# 用法: bash run_domain.sh <domain> [gpu0 gpu1 ...]
#   例: bash run_domain.sh li_ion
#       bash run_domain.sh three_level 0 1
# domain 可选: li_ion | calb | na_ion | zn_ion | three_level

set -euo pipefail

DOMAIN="${1:?用法: bash run_domain.sh <domain> [gpu ...]}"
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
LOG_DIR="log/${TIMESTAMP}_${DOMAIN}"
mkdir -p "${LOG_DIR}"

SLOTS_PER_GPU=3
TOTAL_SLOTS=$(( ${#GPUS[@]} * SLOTS_PER_GPU ))
JOB_THREADS=3

MODELS="autoformer batlinet bigru bilstm cnn dlinear gru ic2ml itransformer lstm micn mlp patchtst severson timemixer transformer"

JOBS=()
for task in rul soh_point soh_traj; do
    for model in ${MODELS}; do
        JOBS+=("${task}|${model}")
    done
done

TOTAL=${#JOBS[@]}
echo "[$(date '+%H:%M:%S')] Domain: ${DOMAIN}  Total jobs: ${TOTAL}"
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
    IFS='|' read -r task model <<< "$job"

    slot=$(wait_for_free_slot)
    gpu=${SLOT_GPU[$slot]}

    log_file="${LOG_DIR}/${task}_${model}.log"
    echo "[$(date '+%H:%M:%S')] START  gpu=${gpu} slot=${slot}  ${DOMAIN}/${task}/${model}"

    nohup bash -c "
        export OMP_NUM_THREADS=${JOB_THREADS} MKL_NUM_THREADS=${JOB_THREADS}
        ${PYTHON} scripts/train.py \
            --domain '${DOMAIN}' --task '${task}' --model '${model}' --gpu '${gpu}' --seed 42 && \
        ${PYTHON} scripts/evaluate.py \
            --domain '${DOMAIN}' --task '${task}' --model '${model}' --gpu '${gpu}'
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
