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

- 📊 **[2026/09/26]** Published complete five-seed Four-Level results for 15 standard baselines; BatLiNet and the remaining standard-domain runs are still in progress.
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

## Data Splits

All splits are made at the **battery level**, so cycles from one battery never appear in more than one split. The split is fixed with <code>split_seed: 1</code> and shared by training seeds 1–5; those seeds change model initialization and optimization, not the train/validation/test batteries.

| Domain | Training | Validation | Test | Split rule |
|---|---:|---:|---:|---|
| Li-ion | 70% | 10% | 20% | Fixed battery-level random split |
| CALB | 60% | 20% | 20% | Fixed battery-level random split |
| Na-ion | 60% | 20% | 20% | Fixed battery-level random split |
| Zn-ion | 60% | 20% | 20% | Fixed battery-level random split |

The **Four-Level** split measures different forms of generalization as the test distribution moves away from the training pool. The levels describe distinct distribution shifts rather than a guarantee that every metric increases monotonically from L1 to L4. Its training pool contains HUST batches 1–7 and 10, MATR batches 1–3, RWTH, SDU, Stanford, Tongji, ISU-ILCC, MICH, CALB, and XJTU. Eight percent of each cathode group is held out for validation. All test batteries below are fixed and excluded from training and validation.

| Level | Generalization setting | Fixed test set |
|---|---|---|
| L1 | Same chemistry and dataset, unseen batches | HUST batches 8–9 |
| L2 | Same chemistry under an unseen batch/protocol distribution | MATR batch 4 |
| L3 | Unseen Li-ion cathode and dataset distributions | CALCE and HNEI |
| L4 | Unseen ion chemistry | Na-ion and Zn-ion |

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

All previous results used the old input definition and have been removed. The table below reports the lowest mean RMSE at each Four-Level setting among the 15 standard baselines that have completed all five seeds. Values are **mean ± population standard deviation** across seeds 1–5. SOH MAE and RMSE use the SOH ratio; RUL errors use cycles; MAPE is shown as a percentage. RUL L1 is empty because that test set has no valid samples under the current RUL construction.

| Task | Level | Model | MAE | RMSE | MAPE |
|---|---|---|---:|---:|---:|
| SOH Trajectory | L1 | TimeMixer | 0.0183 ± 0.0011 | 0.0277 ± 0.0010 | 1.97 ± 0.11% |
| SOH Trajectory | L2 | CNN | 0.0185 ± 0.0017 | 0.0261 ± 0.0010 | 2.04 ± 0.18% |
| SOH Trajectory | L3 | IC2ML | 0.0384 ± 0.0123 | 0.0435 ± 0.0112 | 4.43 ± 1.49% |
| SOH Trajectory | L4 | iTransformer | 0.0620 ± 0.0024 | 0.0702 ± 0.0028 | 114.81 ± 0.74% |
| SOH Point | L1 | BiGRU | 0.0316 ± 0.0044 | 0.0438 ± 0.0013 | 3.41 ± 0.42% |
| SOH Point | L2 | CNN | 0.0446 ± 0.0056 | 0.0828 ± 0.0060 | 6.70 ± 0.73% |
| SOH Point | L3 | MLP | 0.1219 ± 0.0209 | 0.1601 ± 0.0192 | 22.26 ± 3.16% |
| SOH Point | L4 | BiGRU | 0.1362 ± 0.0112 | 0.1692 ± 0.0063 | 165.38 ± 5.27% |
| RUL | L1 |  |  |  |  |
| RUL | L2 | PatchTST | 208.1265 ± 43.6308 | 247.4517 ± 45.4220 | 25.81 ± 4.82% |
| RUL | L3 | MLP | 180.5774 ± 11.6868 | 202.9692 ± 10.7287 | 64.86 ± 4.66% |
| RUL | L4 | MLP | 372.4896 ± 35.6333 | 404.8696 ± 21.0490 | 201.36 ± 29.14% |

Complete model-by-level and model-by-test-set tables are available in [the Four-Level result summary](results/four_level/summary_levels.md) and [the test-set summary](results/four_level/summary_test_sets.md). The repository currently contains 284 complete experiments: all five seeds for 15 standard Four-Level baselines, plus 59 Li-ion SOH Trajectory experiments. BatLiNet and the remaining standard-domain experiments are pending.

Result files follow this layout:

    results/<domain>/<task>/<model>/
    ├── seed<seed>/
    │   ├── results.json
    │   └── test/
    │       ├── predictions.csv
    │       └── <level_dataset>/    # Four-Level only
    └── summary.json

Git publishes JSON summaries, CSV predictions, and Markdown tables. All `.pt`, `.pkl`, and `.npz` files remain local.

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
