#!/usr/bin/env bash
# run_all.sh — 并发训练所有 (domain, task, model) 组合
# 用法: bash run_all.sh
# GPU 分配: 4 × RTX 3090，每张最多 2 个并发进程（共 8 槽）

set -euo pipefail

TIMESTAMP=$(date +"%y-%m-%d_%H-%M-%S")
LOG_DIR="log/${TIMESTAMP}"
mkdir -p "${LOG_DIR}"

CONDA_ENV="zw@RUL-Predict"
GPUS=(0 1 2 3)
SLOTS_PER_GPU=2
TOTAL_SLOTS=$(( ${#GPUS[@]} * SLOTS_PER_GPU ))

# ── 任务列表 ──────────────────────────────────────────────────────────────────

DOMAINS=(li_ion calb na_ion zn_ion three_level)

declare -A TASK_MODELS
TASK_MODELS[rul]="autoformer batlinet batterymformer bigru bilstm cnn dlinear gru ic2ml itransformer lstm micn mlp patchtst severson timemixer transformer"
TASK_MODELS[soh_point]="autoformer bigru bilstm cnn dlinear gru ic2ml itransformer lstm micn mlp patchtst timemixer transformer"
TASK_MODELS[soh_traj]="autoformer bigru bilstm cnn dlinear gru ic2ml itransformer lstm micn mlp patchtst timemixer transformer"

# 构造所有 (domain, task, model) 三元组
JOBS=()
for domain in "${DOMAINS[@]}"; do
    for task in rul soh_point soh_traj; do
        for model in ${TASK_MODELS[$task]}; do
            JOBS+=("${domain}|${task}|${model}")
        done
    done
done

TOTAL=${#JOBS[@]}
echo "[$(date '+%H:%M:%S')] Total jobs: ${TOTAL}"
echo "[$(date '+%H:%M:%S')] Log dir: ${LOG_DIR}"
echo "[$(date '+%H:%M:%S')] GPUs: ${GPUS[*]} (${SLOTS_PER_GPU} slots each = ${TOTAL_SLOTS} concurrent)"
echo ""

# ── 调度器 ────────────────────────────────────────────────────────────────────
# slot_pids[slot] = PID of running job (0 = free)
declare -a SLOT_PIDS
declare -a SLOT_GPU
for (( s=0; s<TOTAL_SLOTS; s++ )); do
    SLOT_PIDS[$s]=0
    SLOT_GPU[$s]=${GPUS[$((s / SLOTS_PER_GPU))]}
done

find_free_slot() {
    for (( s=0; s<TOTAL_SLOTS; s++ )); do
        pid=${SLOT_PIDS[$s]}
        if [[ $pid -eq 0 ]]; then
            echo $s; return
        fi
        # check if process finished
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

DONE=0
for job in "${JOBS[@]}"; do
    IFS='|' read -r domain task model <<< "$job"

    slot=$(wait_for_free_slot)
    gpu=${SLOT_GPU[$slot]}

    log_file="${LOG_DIR}/${domain}_${task}_${model}.log"

    echo "[$(date '+%H:%M:%S')] START  gpu=${gpu} slot=${slot}  ${domain}/${task}/${model}"

    nohup conda run -n "${CONDA_ENV}" python scripts/train.py \
        --domain "${domain}" \
        --task   "${task}"   \
        --model  "${model}"  \
        --gpu    "${gpu}"    \
        > "${log_file}" 2>&1 &

    SLOT_PIDS[$slot]=$!
    DONE=$(( DONE + 1 ))
done

# 等待全部完成
echo ""
echo "[$(date '+%H:%M:%S')] All ${TOTAL} jobs launched. Waiting for completion..."
for (( s=0; s<TOTAL_SLOTS; s++ )); do
    pid=${SLOT_PIDS[$s]}
    if [[ $pid -ne 0 ]]; then
        wait "$pid" 2>/dev/null || true
    fi
done

echo "[$(date '+%H:%M:%S')] All done. Logs: ${LOG_DIR}/"
