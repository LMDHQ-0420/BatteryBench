<p align="center">
  <img src="asset/logo.svg" alt="BatteryBench logo" width="520">
</p>

<h1 align="center">BatteryBench</h1>

<p align="center">
  A unified benchmark for battery lifetime and state-of-health prediction across chemistries and generalization levels.
</p>

<p align="center">
  <a href="README-ZH.md">中文</a>
</p>

## News

- 🔒 **[2026/09/19]** Reworked all task inputs to use complete charge curves and removed capacity leakage from SOH point estimation.
- 🧪 **[2026/09/19]** Started retraining all baselines with the updated 400-point input protocol.
- 🧭 **[2026/07/15]** Introduced the Four-Level protocol for cross-batch, cross-dataset, cross-cathode, and cross-ion evaluation.
- 🚀 **[2026/06/26]** Released the BatteryBench benchmark and training pipeline.

## About

BatteryBench provides a shared training and evaluation interface for battery lifetime and state-of-health prediction. It covers Li-ion, CALB, Na-ion, and Zn-ion experiments, together with a Four-Level protocol for same-dataset cross-batch (L1), cross-dataset (L2), cross-cathode (L3), and cross-ion (L4) generalization.

All tasks now use only the complete positive-current charge segment. Every curve is linearly resampled to 400 points. Voltage is divided by the maximum voltage of that charge segment; current and charge capacity are divided by nominal cell capacity. Discharge capacity is used only to construct SOH, EOL, and RUL labels and is never exposed as a model input.

## Input Protocol

| Task | Observed input | Tensor shape | Channels | Target |
|---|---|---|---|---|
| SOH Point | Complete charge curve of the current cycle | <code>[B, 1, 2, 400]</code> | V, I | Current-cycle SOH |
| SOH Trajectory | Complete charge curves from cycle 1 through cycle u | <code>[B, S, 3, 400]</code> | V, I, Q<sub>charge</sub> | SOH from cycle u+1 to EOL |
| RUL | Complete charge curves from cycle 1 through cycle u | <code>[B, S, 3, 400]</code> | V, I, Q<sub>charge</sub> | Battery lifetime/EOL |

Unobserved history positions are zero-filled and excluded through <code>curve_attn_mask</code>. SOH trajectory loss and metrics are computed only after the last observed cycle. No task uses discharge curves, and SOH Point does not expose charge or discharge capacity to any baseline, including IC2ML, BatLiNet, and Severson.

## Models

| Family | Models |
|---|---|
| MLP / linear | MLP, DLinear |
| Recurrent | GRU, BiGRU, LSTM, BiLSTM |
| Convolutional | CNN, MICN |
| Transformer / mixer | Transformer, PatchTST, Autoformer, iTransformer, TimeMixer |
| Battery-specific | IC2ML, BatLiNet, Severson |

## Installation

    conda create -n BatteryBench python=3.11
    conda activate BatteryBench
    pip install -r requirements.txt

Place raw datasets under <code>data/raw/</code>. Dataset paths and experiment settings are defined in <code>configs/</code>.

## Quick Start

Train and evaluate one experiment:

    python scripts/train.py --domain li_ion --model gru --task rul --seed 1 --gpu 0
    python scripts/evaluate.py --domain li_ion --model gru --task rul --seed 1 --gpu 0

Resume the complete benchmark pipeline:

    bash run_pipeline.sh

The pipeline first runs Four-Level seed 1 in the order SOH Trajectory, SOH Point, and RUL; it then runs Four-Level seeds 2–5, all standard-domain experiments, and finally every BatLiNet experiment. Ordinary models use two slots on GPU 0 and three slots on GPUs 1–3. BatLiNet uses one slot on GPUs 1–3 and never runs on GPU 0. Recovery is based on complete checkpoints and evaluation artifacts.

## Results

All previous results used the old input definition and have been removed. Every model is being retrained from scratch. Standard domains use one fixed train/validation/test split shared by seeds 1–5; completed summaries report each metric as **mean ± standard deviation** across those seeds.

| Task | MAE | RMSE | MAPE | ACC15 |
|---|---:|---:|---:|---:|
| RUL |  |  |  |  |
| SOH Point |  |  |  | — |
| SOH Trajectory |  |  |  | — |

Result files follow this layout:

    results/<domain>/<task>/<model>/
    ├── seed<seed>/
    │   ├── best.pt
    │   ├── results.json
    │   └── test/
    │       ├── predictions.csv
    │       ├── trajectories.npz
    │       └── <level_dataset>/    # Four-Level only
    └── summary.json

Git publishes JSON summaries and CSV predictions. Checkpoints, fitted models, caches, and trajectory NPZ files remain local.

## Project Structure

    BatteryBench/
    ├── asset/                  Project assets
    ├── configs/                Default and domain configurations
    ├── data/                   Local raw, processed, and cached data
    ├── results/                Evaluation metrics and predictions
    ├── scripts/                Training, evaluation, and pipeline entry points
    ├── src/
    │   ├── data/               Datasets
    │   ├── evaluate/           Metrics and result writers
    │   ├── models/             Model registry and implementations
    │   ├── preprocess/         Dataset preprocessing
    │   └── train/              Training loops
    ├── tests/                  Regression tests
    ├── requirements.txt
    └── run_pipeline.sh

## TODO

- [ ] Finish retraining and evaluation under the charge-only input protocol.
- [ ] Publish the complete five-seed benchmark tables.
