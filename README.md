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

- 🧪 **[2026/09/19]** Started the full five-seed benchmark refresh with the new reproducible result layout; completed outputs are published incrementally.
- 📊 **[2026/08/11]** Released full-lifecycle SOH point evaluation across all standard domain-model combinations.
- 🔧 **[2026/07/19]** Re-evaluated SOH trajectory experiments after the BatLiNet speedup and Four-Level EOL correction.
- 🧭 **[2026/07/15]** Introduced the Four-Level protocol for cross-batch, cross-dataset, cross-cathode, and cross-ion evaluation.
- 🚀 **[2026/06/26]** Released the first complete BatteryBench benchmark results and training pipeline.

## About

BatteryBench provides one training and evaluation interface for three battery degradation tasks:

| Task | Goal | Output |
|---|---|---|
| `rul` | Predict remaining useful life until end of life | scalar cycle count |
| `soh_point` | Estimate SOH at an observed cycle | scalar SOH |
| `soh_traj` | Forecast the future SOH degradation trajectory | SOH sequence |

The benchmark includes Li-ion, CALB, Na-ion, and Zn-ion domain experiments plus a Four-Level protocol that measures progressively harder generalization: same-dataset cross-batch (L1), cross-dataset (L2), cross-cathode (L3), and cross-ion (L4). Sixteen sequence, time-series, and battery-specific baselines share the same data and reporting interfaces.

## Models

| Family | Models |
|---|---|
| MLP / linear | MLP, DLinear |
| Recurrent | GRU, BiGRU, LSTM, BiLSTM |
| Convolutional | CNN, MICN |
| Transformer / mixer | Transformer, PatchTST, Autoformer, iTransformer, TimeMixer |
| Battery-specific | IC2ML, BatLiNet, Severson |

## Installation

```bash
conda create -n BatteryBench python=3.11
conda activate BatteryBench
pip install -r requirements.txt
```

Raw datasets should be placed under `data/raw/`. Dataset-specific paths and experiment settings are defined in `configs/`.

## Quick Start

Train and evaluate one experiment:

```bash
python scripts/train.py --domain li_ion --model gru --task rul --seed 42 --gpu 0
python scripts/evaluate.py --domain li_ion --model gru --task rul --seed 42 --gpu 0
```

Resume the complete benchmark pipeline:

```bash
bash run_pipeline.sh
```

The pipeline skips experiments whose complete new-format evaluation artifacts already exist. Standard domains use three fixed data splits per model seed; reported summaries aggregate completed model seeds. Four-Level uses one fixed generalization split per seed.

## Result Layout

```text
results/<domain>/<task>/<model>/
├── seed<seed>/
│   ├── results.json
│   └── test/<split-or-level_dataset>/
│       ├── predictions.csv
│       └── trajectories.npz       # SOH trajectory models
└── summary.json
```

Model checkpoints and fitted scalers remain local and are not published in Git. `summary.json`, per-seed metrics, predictions, and trajectory outputs follow the layout above.

## Latest Results

Values are **mean ± standard deviation** over the completed model seeds recorded in each current `summary.json`. MAPE and ACC15 use the `[0, 1]` scale. `—` or an empty cell means that a complete evaluation is not yet available. Training is still in progress, so seed counts can differ across tasks.

### Standard Domains — RUL

| Model | Li-ion | CALB | Na-ion | Zn-ion |
|---|---|---|---|---|
| Autoformer | MAPE 0.5240 ± 0.0210<br>ACC15 0.3262 ± 0.0111 | MAPE 0.2352 ± 0.0065<br>ACC15 0.3107 ± 0.0142 | MAPE 0.1927 ± 0.0140<br>ACC15 0.3703 ± 0.0497 | MAPE 0.6451 ± 0.0138<br>ACC15 0.1799 ± 0.0153 |
| BatLiNet | — | — | — | — |
| BiGRU | MAPE 0.2509 ± 0.0091<br>ACC15 0.5456 ± 0.0182 | MAPE 0.1920 ± 0.0061<br>ACC15 0.4939 ± 0.0341 | MAPE 0.1879 ± 0.0068<br>ACC15 0.3717 ± 0.0428 | MAPE 0.8877 ± 0.0603<br>ACC15 0.0464 ± 0.0074 |
| BiLSTM | MAPE 0.2701 ± 0.0157<br>ACC15 0.5449 ± 0.0228 | MAPE 0.1945 ± 0.0113<br>ACC15 0.4505 ± 0.0499 | MAPE 0.1849 ± 0.0151<br>ACC15 0.3828 ± 0.0511 | MAPE 0.9937 ± 0.0260<br>ACC15 0.0270 ± 0.0129 |
| CNN | MAPE 0.2895 ± 0.0113<br>ACC15 0.5059 ± 0.0136 | MAPE 0.1965 ± 0.0042<br>ACC15 0.4776 ± 0.0296 | MAPE 0.1864 ± 0.0054<br>ACC15 0.3662 ± 0.0456 | MAPE 0.5168 ± 0.0686<br>ACC15 0.2681 ± 0.0426 |
| DLinear | MAPE 0.9013 ± 0.0089<br>ACC15 0.1908 ± 0.0074 | MAPE 0.2153 ± 0.0046<br>ACC15 0.4318 ± 0.0176 | MAPE 0.1974 ± 0.0003<br>ACC15 0.4431 ± 0.0027 | MAPE 0.9785 ± 0.0151<br>ACC15 0.0414 ± 0.0042 |
| GRU | MAPE 0.2810 ± 0.0060<br>ACC15 0.5236 ± 0.0205 | MAPE 0.1967 ± 0.0103<br>ACC15 0.4519 ± 0.0497 | MAPE 0.1968 ± 0.0113<br>ACC15 0.2964 ± 0.0464 | MAPE 0.9774 ± 0.0470<br>ACC15 0.0498 ± 0.0130 |
| IC2ML | MAPE 0.3730 ± 0.0216<br>ACC15 0.3801 ± 0.0324 | MAPE 0.1409 ± 0.0117<br>ACC15 0.7433 ± 0.0529 | MAPE 0.1756 ± 0.0203<br>ACC15 0.5190 ± 0.0613 | MAPE 0.8165 ± 0.0794<br>ACC15 0.0766 ± 0.0213 |
| iTransformer | MAPE 0.3880 ± 0.0177<br>ACC15 0.4241 ± 0.0215 | MAPE 0.1865 ± 0.0063<br>ACC15 0.4818 ± 0.0495 | MAPE 0.1963 ± 0.0069<br>ACC15 0.3104 ± 0.0478 | MAPE 0.9900 ± 0.0368<br>ACC15 0.0375 ± 0.0231 |
| LSTM | MAPE 0.2691 ± 0.0267<br>ACC15 0.5152 ± 0.0385 | MAPE 0.1890 ± 0.0091<br>ACC15 0.4901 ± 0.0521 | MAPE 0.1939 ± 0.0125<br>ACC15 0.3377 ± 0.0386 | MAPE 1.0160 ± 0.0354<br>ACC15 0.0309 ± 0.0071 |
| MICN | MAPE 0.3303 ± 0.0239<br>ACC15 0.4591 ± 0.0239 | MAPE 0.1964 ± 0.0211<br>ACC15 0.4715 ± 0.0894 | MAPE 0.1984 ± 0.0064<br>ACC15 0.2920 ± 0.0548 | MAPE 0.5165 ± 0.0153<br>ACC15 0.2280 ± 0.0097 |
| MLP | MAPE 0.5236 ± 0.0243<br>ACC15 0.3380 ± 0.0122 | MAPE 0.2113 ± 0.0073<br>ACC15 0.3629 ± 0.0464 | MAPE 0.1939 ± 0.0122<br>ACC15 0.3389 ± 0.0352 | MAPE 0.9643 ± 0.0283<br>ACC15 0.0499 ± 0.0117 |
| PatchTST | MAPE 0.3753 ± 0.0184<br>ACC15 0.4211 ± 0.0285 | MAPE 0.2570 ± 0.0134<br>ACC15 0.3933 ± 0.0323 | MAPE 0.2011 ± 0.0044<br>ACC15 0.4322 ± 0.0095 | MAPE 0.6502 ± 0.2142<br>ACC15 0.1864 ± 0.0831 |
| Severson | MAPE 0.9817 ± 0.0000<br>ACC15 0.1685 ± 0.0000 | MAPE 0.4199 ± 0.0000<br>ACC15 0.0524 ± 0.0000 | MAPE 0.1911 ± 0.0000<br>ACC15 0.4456 ± 0.0000 | MAPE 1.0329 ± 0.0000<br>ACC15 0.0856 ± 0.0000 |
| TimeMixer | MAPE 0.3187 ± 0.0190<br>ACC15 0.5066 ± 0.0119 | MAPE 0.2037 ± 0.0075<br>ACC15 0.4412 ± 0.0500 | MAPE 0.2012 ± 0.0093<br>ACC15 0.2766 ± 0.0199 | MAPE 0.8584 ± 0.0830<br>ACC15 0.0665 ± 0.0194 |
| Transformer | MAPE 0.2506 ± 0.0218<br>ACC15 0.5466 ± 0.0167 | MAPE 0.1755 ± 0.0213<br>ACC15 0.5609 ± 0.0731 | MAPE 0.1903 ± 0.0231<br>ACC15 0.3796 ± 0.1320 | MAPE 0.5402 ± 0.0515<br>ACC15 0.2125 ± 0.0558 |

### Standard Domains — SOH Point

| Model | Li-ion | CALB | Na-ion | Zn-ion |
|---|---|---|---|---|
| Autoformer | MAE 0.3271 ± 0.0005<br>RMSE 0.3642 ± 0.0002<br>MAPE 3.7143 ± 0.0171 | MAE 0.0206 ± 0.0012<br>RMSE 0.0207 ± 0.0012<br>MAPE 0.0206 ± 0.0012 | MAE 0.0399 ± 0.0001<br>RMSE 0.0515 ± 0.0002<br>MAPE 0.0492 ± 0.0001 | MAE 0.1454 ± 0.0003<br>RMSE 0.1834 ± 0.0002<br>MAPE 0.2401 ± 0.0002 |
| BatLiNet | — | — | — | — |
| BiGRU | MAE 0.0205 ± 0.0003<br>RMSE 0.0325 ± 0.0010<br>MAPE 0.0946 ± 0.0113 | MAE 0.0037 ± 0.0002<br>RMSE 0.0053 ± 0.0004<br>MAPE 0.0037 ± 0.0002 | MAE 0.0020 ± 0.0004<br>RMSE 0.0043 ± 0.0007<br>MAPE 0.0025 ± 0.0004 | MAE 0.0037 ± 0.0003<br>RMSE 0.0056 ± 0.0004<br>MAPE 0.0058 ± 0.0005 |
| BiLSTM | MAE 0.0183 ± 0.0007<br>RMSE 0.0295 ± 0.0017<br>MAPE 0.0932 ± 0.0183 | MAE 0.0040 ± 0.0004<br>RMSE 0.0060 ± 0.0006<br>MAPE 0.0040 ± 0.0004 | MAE 0.0047 ± 0.0007<br>RMSE 0.0076 ± 0.0007<br>MAPE 0.0057 ± 0.0008 | MAE 0.0033 ± 0.0002<br>RMSE 0.0052 ± 0.0002<br>MAPE 0.0051 ± 0.0001 |
| CNN | MAE 0.0178 ± 0.0005<br>RMSE 0.0276 ± 0.0009<br>MAPE 0.1150 ± 0.0139 | MAE 0.0054 ± 0.0006<br>RMSE 0.0085 ± 0.0014<br>MAPE 0.0054 ± 0.0006 | MAE 0.0016 ± 0.0001<br>RMSE 0.0049 ± 0.0003<br>MAPE 0.0020 ± 0.0001 | MAE 0.0046 ± 0.0003<br>RMSE 0.0074 ± 0.0004<br>MAPE 0.0063 ± 0.0005 |
| DLinear | MAE 0.1230 ± 0.0003<br>RMSE 0.1607 ± 0.0004<br>MAPE 1.0689 ± 0.0157 | MAE 0.1537 ± 0.0043<br>RMSE 0.3185 ± 0.0113<br>MAPE 0.1539 ± 0.0043 | MAE 0.0204 ± 0.0014<br>RMSE 0.0275 ± 0.0016<br>MAPE 0.0248 ± 0.0018 | MAE 0.0549 ± 0.0033<br>RMSE 0.0704 ± 0.0020<br>MAPE 0.0809 ± 0.0021 |
| GRU | MAE 0.0186 ± 0.0007<br>RMSE 0.0287 ± 0.0012<br>MAPE 0.0787 ± 0.0081 | MAE 0.0036 ± 0.0004<br>RMSE 0.0043 ± 0.0004<br>MAPE 0.0036 ± 0.0004 | MAE 0.0019 ± 0.0002<br>RMSE 0.0043 ± 0.0007<br>MAPE 0.0024 ± 0.0002 | MAE 0.0042 ± 0.0002<br>RMSE 0.0062 ± 0.0003<br>MAPE 0.0063 ± 0.0004 |
| IC2ML | MAE 0.0362 ± 0.0009<br>RMSE 0.0658 ± 0.0006<br>MAPE 0.1886 ± 0.0515 | MAE 0.0118 ± 0.0005<br>RMSE 0.0119 ± 0.0005<br>MAPE 0.0118 ± 0.0005 | MAE 0.0021 ± 0.0003<br>RMSE 0.0039 ± 0.0004<br>MAPE 0.0025 ± 0.0003 | MAE 0.0023 ± 0.0002<br>RMSE 0.0032 ± 0.0003<br>MAPE 0.0031 ± 0.0003 |
| iTransformer | MAE 0.0246 ± 0.0009<br>RMSE 0.0368 ± 0.0012<br>MAPE 0.1294 ± 0.0215 | MAE 0.0013 ± 0.0003<br>RMSE 0.0014 ± 0.0003<br>MAPE 0.0013 ± 0.0003 | MAE 0.0109 ± 0.0046<br>RMSE 0.0154 ± 0.0042<br>MAPE 0.0132 ± 0.0055 | MAE 0.0117 ± 0.0012<br>RMSE 0.0149 ± 0.0016<br>MAPE 0.0169 ± 0.0023 |
| LSTM | MAE 0.0193 ± 0.0006<br>RMSE 0.0314 ± 0.0016<br>MAPE 0.1062 ± 0.0152 | MAE 0.0034 ± 0.0003<br>RMSE 0.0043 ± 0.0003<br>MAPE 0.0034 ± 0.0003 | MAE 0.0044 ± 0.0006<br>RMSE 0.0076 ± 0.0008<br>MAPE 0.0053 ± 0.0007 | MAE 0.0033 ± 0.0002<br>RMSE 0.0053 ± 0.0004<br>MAPE 0.0051 ± 0.0004 |
| MICN | MAE 0.0487 ± 0.0000<br>RMSE 0.0663 ± 0.0000<br>MAPE 0.2364 ± 0.0000 | MAE 0.0041 ± 0.0002<br>RMSE 0.0053 ± 0.0005<br>MAPE 0.0041 ± 0.0002 | MAE 0.0026 ± 0.0002<br>RMSE 0.0073 ± 0.0002<br>MAPE 0.0034 ± 0.0002 | MAE 0.0033 ± 0.0000<br>RMSE 0.0071 ± 0.0002<br>MAPE 0.0053 ± 0.0001 |
| MLP | MAE 0.0379 ± 0.0019<br>RMSE 0.0491 ± 0.0025<br>MAPE 0.2538 ± 0.0482 | MAE 0.0278 ± 0.0070<br>RMSE 0.0326 ± 0.0073<br>MAPE 0.0278 ± 0.0070 | MAE 0.0088 ± 0.0008<br>RMSE 0.0123 ± 0.0020<br>MAPE 0.0106 ± 0.0010 | MAE 0.0183 ± 0.0014<br>RMSE 0.0252 ± 0.0023<br>MAPE 0.0311 ± 0.0023 |
| PatchTST | MAE 0.0364 ± 0.0001<br>RMSE 0.0543 ± 0.0011<br>MAPE 0.2934 ± 0.0044 | MAE 0.0169 ± 0.0052<br>RMSE 0.0355 ± 0.0120<br>MAPE 0.0169 ± 0.0052 | MAE 0.0320 ± 0.0016<br>RMSE 0.0393 ± 0.0025<br>MAPE 0.0387 ± 0.0020 | MAE 0.0374 ± 0.0009<br>RMSE 0.0552 ± 0.0008<br>MAPE 0.0540 ± 0.0009 |
| Severson | MAE 0.2154 ± 0.0000<br>RMSE 0.2638 ± 0.0000<br>MAPE 2.3211 ± 0.0000 | MAE 0.0250 ± 0.0000<br>RMSE 0.0250 ± 0.0000<br>MAPE 0.0250 ± 0.0000 | MAE 0.0010 ± 0.0000<br>RMSE 0.0035 ± 0.0000<br>MAPE 0.0012 ± 0.0000 | MAE 0.0223 ± 0.0000<br>RMSE 0.0298 ± 0.0000<br>MAPE 0.0313 ± 0.0000 |
| TimeMixer | MAE 0.0222 ± 0.0002<br>RMSE 0.0305 ± 0.0004<br>MAPE 0.1578 ± 0.0188 | MAE 0.0161 ± 0.0042<br>RMSE 0.0193 ± 0.0046<br>MAPE 0.0161 ± 0.0042 | MAE 0.0111 ± 0.0036<br>RMSE 0.0136 ± 0.0040<br>MAPE 0.0129 ± 0.0042 | MAE 0.0164 ± 0.0033<br>RMSE 0.0212 ± 0.0037<br>MAPE 0.0231 ± 0.0044 |
| Transformer | MAE 0.0187 ± 0.0000<br>RMSE 0.0272 ± 0.0000<br>MAPE 0.2558 ± 0.0000 | MAE 0.0017 ± 0.0005<br>RMSE 0.0021 ± 0.0007<br>MAPE 0.0017 ± 0.0005 | MAE 0.0025 ± 0.0004<br>RMSE 0.0070 ± 0.0007<br>MAPE 0.0031 ± 0.0005 | MAE 0.0043 ± 0.0012<br>RMSE 0.0067 ± 0.0011<br>MAPE 0.0059 ± 0.0014 |

### Standard Domains — SOH Trajectory

| Model | Li-ion | CALB | Na-ion | Zn-ion |
|---|---|---|---|---|
| Autoformer | MAE 0.0279 ± 0.0016<br>RMSE 0.0366 ± 0.0019<br>MAPE 0.0310 ± 0.0019 | MAE 0.0068 ± 0.0004<br>RMSE 0.0113 ± 0.0009<br>MAPE 0.0071 ± 0.0004 | MAE 0.0193 ± 0.0015<br>RMSE 0.0251 ± 0.0017<br>MAPE 0.0226 ± 0.0017 | MAE 0.0413 ± 0.0006<br>RMSE 0.0519 ± 0.0005<br>MAPE 0.0446 ± 0.0007 |
| BatLiNet | — | — | — | — |
| BiGRU | MAE 0.0198 ± 0.0007<br>RMSE 0.0287 ± 0.0011<br>MAPE 0.0220 ± 0.0008 | MAE 0.0063 ± 0.0004<br>RMSE 0.0116 ± 0.0010<br>MAPE 0.0066 ± 0.0004 | MAE 0.0197 ± 0.0003<br>RMSE 0.0255 ± 0.0004<br>MAPE 0.0229 ± 0.0004 | MAE 0.0388 ± 0.0007<br>RMSE 0.0512 ± 0.0008<br>MAPE 0.0423 ± 0.0007 |
| BiLSTM | MAE 0.0200 ± 0.0005<br>RMSE 0.0287 ± 0.0005<br>MAPE 0.0222 ± 0.0006 | MAE 0.0059 ± 0.0007<br>RMSE 0.0102 ± 0.0018<br>MAPE 0.0062 ± 0.0007 | MAE 0.0192 ± 0.0006<br>RMSE 0.0246 ± 0.0007<br>MAPE 0.0224 ± 0.0007 | MAE 0.0405 ± 0.0005<br>RMSE 0.0518 ± 0.0001<br>MAPE 0.0441 ± 0.0005 |
| CNN | MAE 0.0201 ± 0.0002<br>RMSE 0.0284 ± 0.0002<br>MAPE 0.0222 ± 0.0002 | MAE 0.0065 ± 0.0008<br>RMSE 0.0114 ± 0.0024<br>MAPE 0.0068 ± 0.0008 | MAE 0.0132 ± 0.0006<br>RMSE 0.0185 ± 0.0011<br>MAPE 0.0155 ± 0.0006 | MAE 0.0351 ± 0.0008<br>RMSE 0.0479 ± 0.0006<br>MAPE 0.0381 ± 0.0008 |
| DLinear | MAE 0.0470 ± 0.0001<br>RMSE 0.0567 ± 0.0001<br>MAPE 0.0519 ± 0.0001 | MAE 0.0055 ± 0.0001<br>RMSE 0.0086 ± 0.0001<br>MAPE 0.0057 ± 0.0001 | MAE 0.0264 ± 0.0001<br>RMSE 0.0316 ± 0.0001<br>MAPE 0.0311 ± 0.0002 | MAE 0.0430 ± 0.0004<br>RMSE 0.0527 ± 0.0004<br>MAPE 0.0462 ± 0.0003 |
| GRU | MAE 0.0199 ± 0.0005<br>RMSE 0.0284 ± 0.0007<br>MAPE 0.0220 ± 0.0005 | MAE 0.0059 ± 0.0004<br>RMSE 0.0100 ± 0.0011<br>MAPE 0.0061 ± 0.0004 | MAE 0.0177 ± 0.0013<br>RMSE 0.0231 ± 0.0018<br>MAPE 0.0208 ± 0.0016 | MAE 0.0403 ± 0.0003<br>RMSE 0.0515 ± 0.0004<br>MAPE 0.0438 ± 0.0003 |
| IC2ML | MAE 0.0210 ± 0.0004<br>RMSE 0.0300 ± 0.0005<br>MAPE 0.0234 ± 0.0004 | MAE 0.0063 ± 0.0004<br>RMSE 0.0097 ± 0.0007<br>MAPE 0.0066 ± 0.0004 | MAE 0.0103 ± 0.0006<br>RMSE 0.0140 ± 0.0005<br>MAPE 0.0121 ± 0.0007 | MAE 0.0365 ± 0.0013<br>RMSE 0.0487 ± 0.0015<br>MAPE 0.0395 ± 0.0013 |
| iTransformer | MAE 0.0231 ± 0.0005<br>RMSE 0.0320 ± 0.0007<br>MAPE 0.0256 ± 0.0006 | MAE 0.0052 ± 0.0001<br>RMSE 0.0083 ± 0.0001<br>MAPE 0.0055 ± 0.0001 | MAE 0.0251 ± 0.0003<br>RMSE 0.0302 ± 0.0005<br>MAPE 0.0294 ± 0.0004 | MAE 0.0400 ± 0.0005<br>RMSE 0.0511 ± 0.0004<br>MAPE 0.0435 ± 0.0005 |
| LSTM | MAE 0.0202 ± 0.0002<br>RMSE 0.0289 ± 0.0004<br>MAPE 0.0225 ± 0.0002 | MAE 0.0052 ± 0.0003<br>RMSE 0.0084 ± 0.0006<br>MAPE 0.0055 ± 0.0003 | MAE 0.0141 ± 0.0022<br>RMSE 0.0185 ± 0.0024<br>MAPE 0.0166 ± 0.0026 | MAE 0.0406 ± 0.0004<br>RMSE 0.0514 ± 0.0003<br>MAPE 0.0440 ± 0.0004 |
| MICN | MAE 0.0221 ± 0.0003<br>RMSE 0.0311 ± 0.0003<br>MAPE 0.0245 ± 0.0003 | MAE 0.0071 ± 0.0002<br>RMSE 0.0124 ± 0.0006<br>MAPE 0.0073 ± 0.0002 | MAE 0.0190 ± 0.0004<br>RMSE 0.0251 ± 0.0015<br>MAPE 0.0222 ± 0.0005 | MAE 0.0403 ± 0.0016<br>RMSE 0.0529 ± 0.0014<br>MAPE 0.0435 ± 0.0016 |
| MLP | MAE 0.0314 ± 0.0007<br>RMSE 0.0398 ± 0.0007<br>MAPE 0.0348 ± 0.0007 | MAE 0.0059 ± 0.0001<br>RMSE 0.0090 ± 0.0001<br>MAPE 0.0062 ± 0.0001 | MAE 0.0253 ± 0.0006<br>RMSE 0.0301 ± 0.0006<br>MAPE 0.0297 ± 0.0007 | MAE 0.0431 ± 0.0008<br>RMSE 0.0531 ± 0.0006<br>MAPE 0.0464 ± 0.0008 |
| PatchTST | MAE 0.0394 ± 0.0003<br>RMSE 0.0513 ± 0.0004<br>MAPE 0.0421 ± 0.0003 | MAE 0.0232 ± 0.0012<br>RMSE 0.0331 ± 0.0021<br>MAPE 0.0238 ± 0.0012 | MAE 0.0191 ± 0.0021<br>RMSE 0.0259 ± 0.0020<br>MAPE 0.0223 ± 0.0025 | MAE 0.0466 ± 0.0015<br>RMSE 0.0581 ± 0.0016<br>MAPE 0.0492 ± 0.0015 |
| Severson | MAE 0.0366 ± 0.0000<br>RMSE 0.0421 ± 0.0000<br>MAPE 0.0407 ± 0.0000 | MAE 0.0274 ± 0.0000<br>RMSE 0.0303 ± 0.0000<br>MAPE 0.0296 ± 0.0000 | MAE 0.0212 ± 0.0000<br>RMSE 0.0266 ± 0.0000<br>MAPE 0.0251 ± 0.0000 | MAE 0.0379 ± 0.0000<br>RMSE 0.0945 ± 0.0000<br>MAPE 0.0411 ± 0.0000 |
| TimeMixer | MAE 0.0197 ± 0.0003<br>RMSE 0.0278 ± 0.0004<br>MAPE 0.0218 ± 0.0002 | MAE 0.0073 ± 0.0002<br>RMSE 0.0147 ± 0.0005<br>MAPE 0.0076 ± 0.0003 | MAE 0.0162 ± 0.0019<br>RMSE 0.0239 ± 0.0025<br>MAPE 0.0190 ± 0.0022 | MAE 0.0373 ± 0.0009<br>RMSE 0.0493 ± 0.0009<br>MAPE 0.0404 ± 0.0010 |
| Transformer | MAE 0.0198 ± 0.0006<br>RMSE 0.0287 ± 0.0011<br>MAPE 0.0219 ± 0.0006 | MAE 0.0060 ± 0.0004<br>RMSE 0.0109 ± 0.0008<br>MAPE 0.0063 ± 0.0004 | MAE 0.0147 ± 0.0020<br>RMSE 0.0227 ± 0.0022<br>MAPE 0.0172 ± 0.0023 | MAE 0.0342 ± 0.0006<br>RMSE 0.0475 ± 0.0005<br>MAPE 0.0371 ± 0.0006 |

### Four-Level — SOH Point

| Model | Seeds | L1 | L2 | L3 | L4 |
|---|---:|---|---|---|---|
| Autoformer | 1 | MAE 0.4013 ± 0.0000<br>RMSE 0.4047 ± 0.0000<br>MAPE 0.4125 ± 0.0000 | MAE 0.3369 ± 0.0000<br>RMSE 0.3473 ± 0.0000<br>MAPE 0.3703 ± 0.0000 | MAE 0.1970 ± 0.0000<br>RMSE 0.2237 ± 0.0000<br>MAPE 0.2781 ± 0.0000 | MAE 0.2926 ± 0.0000<br>RMSE 0.3200 ± 0.0000<br>MAPE 1.1387 ± 0.0000 |
| BatLiNet |  |  |  |  |  |
| BiGRU | 1 | MAE 0.0129 ± 0.0000<br>RMSE 0.0160 ± 0.0000<br>MAPE 0.0132 ± 0.0000 | MAE 0.0090 ± 0.0000<br>RMSE 0.0104 ± 0.0000<br>MAPE 0.0103 ± 0.0000 | MAE 0.0220 ± 0.0000<br>RMSE 0.0301 ± 0.0000<br>MAPE 0.0365 ± 0.0000 | MAE 0.0436 ± 0.0000<br>RMSE 0.0489 ± 0.0000<br>MAPE 0.0715 ± 0.0000 |
| BiLSTM | 1 | MAE 0.0181 ± 0.0000<br>RMSE 0.0221 ± 0.0000<br>MAPE 0.0189 ± 0.0000 | MAE 0.0146 ± 0.0000<br>RMSE 0.0356 ± 0.0000<br>MAPE 0.0239 ± 0.0000 | MAE 0.0248 ± 0.0000<br>RMSE 0.0282 ± 0.0000<br>MAPE 0.0380 ± 0.0000 | MAE 0.1581 ± 0.0000<br>RMSE 0.1781 ± 0.0000<br>MAPE 0.2110 ± 0.0000 |
| CNN | 1 | MAE 0.0221 ± 0.0000<br>RMSE 0.0290 ± 0.0000<br>MAPE 0.0237 ± 0.0000 | MAE 0.0159 ± 0.0000<br>RMSE 0.0365 ± 0.0000<br>MAPE 0.0245 ± 0.0000 | MAE 0.0517 ± 0.0000<br>RMSE 0.0578 ± 0.0000<br>MAPE 0.0686 ± 0.0000 | MAE 0.4133 ± 0.0000<br>RMSE 0.4322 ± 0.0000<br>MAPE 0.7133 ± 0.0000 |
| DLinear | 1 | MAE 0.1442 ± 0.0000<br>RMSE 0.1918 ± 0.0000<br>MAPE 0.1526 ± 0.0000 | MAE 0.1648 ± 0.0000<br>RMSE 0.2554 ± 0.0000<br>MAPE 0.2015 ± 0.0000 | MAE 0.3582 ± 0.0000<br>RMSE 0.3851 ± 0.0000<br>MAPE 0.4806 ± 0.0000 | MAE 0.7828 ± 0.0000<br>RMSE 0.8402 ± 0.0000<br>MAPE 2.4334 ± 0.0000 |
| GRU | 1 | MAE 0.0204 ± 0.0000<br>RMSE 0.0322 ± 0.0000<br>MAPE 0.0222 ± 0.0000 | MAE 0.0032 ± 0.0000<br>RMSE 0.0062 ± 0.0000<br>MAPE 0.0048 ± 0.0000 | MAE 0.0161 ± 0.0000<br>RMSE 0.0223 ± 0.0000<br>MAPE 0.0249 ± 0.0000 | MAE 0.1048 ± 0.0000<br>RMSE 0.1113 ± 0.0000<br>MAPE 0.1940 ± 0.0000 |
| IC2ML | 1 | MAE 0.0093 ± 0.0000<br>RMSE 0.0119 ± 0.0000<br>MAPE 0.0095 ± 0.0000 | MAE 0.0150 ± 0.0000<br>RMSE 0.0379 ± 0.0000<br>MAPE 0.0232 ± 0.0000 | MAE 0.0641 ± 0.0000<br>RMSE 0.1094 ± 0.0000<br>MAPE 0.1269 ± 0.0000 | MAE 0.0286 ± 0.0000<br>RMSE 0.0631 ± 0.0000<br>MAPE 0.1374 ± 0.0000 |
| iTransformer | 1 | MAE 0.0233 ± 0.0000<br>RMSE 0.0314 ± 0.0000<br>MAPE 0.0253 ± 0.0000 | MAE 0.0404 ± 0.0000<br>RMSE 0.0665 ± 0.0000<br>MAPE 0.0590 ± 0.0000 | MAE 0.0337 ± 0.0000<br>RMSE 0.0420 ± 0.0000<br>MAPE 0.0586 ± 0.0000 | MAE 0.1070 ± 0.0000<br>RMSE 0.1590 ± 0.0000<br>MAPE 1.4924 ± 0.0000 |
| LSTM | 1 | MAE 0.0121 ± 0.0000<br>RMSE 0.0158 ± 0.0000<br>MAPE 0.0126 ± 0.0000 | MAE 0.0045 ± 0.0000<br>RMSE 0.0061 ± 0.0000<br>MAPE 0.0056 ± 0.0000 | MAE 0.0306 ± 0.0000<br>RMSE 0.0363 ± 0.0000<br>MAPE 0.0503 ± 0.0000 | MAE 0.1338 ± 0.0000<br>RMSE 0.1575 ± 0.0000<br>MAPE 0.3048 ± 0.0000 |
| MICN | 1 | MAE 0.0222 ± 0.0000<br>RMSE 0.0271 ± 0.0000<br>MAPE 0.0231 ± 0.0000 | MAE 0.0260 ± 0.0000<br>RMSE 0.0378 ± 0.0000<br>MAPE 0.0329 ± 0.0000 | MAE 0.0634 ± 0.0000<br>RMSE 0.0727 ± 0.0000<br>MAPE 0.0884 ± 0.0000 | MAE 0.2700 ± 0.0000<br>RMSE 0.2774 ± 0.0000<br>MAPE 0.4150 ± 0.0000 |
| MLP | 1 | MAE 0.0244 ± 0.0000<br>RMSE 0.0394 ± 0.0000<br>MAPE 0.0269 ± 0.0000 | MAE 0.0332 ± 0.0000<br>RMSE 0.0575 ± 0.0000<br>MAPE 0.0474 ± 0.0000 | MAE 0.0755 ± 0.0000<br>RMSE 0.0786 ± 0.0000<br>MAPE 0.1139 ± 0.0000 | MAE 0.4721 ± 0.0000<br>RMSE 0.5214 ± 0.0000<br>MAPE 1.8344 ± 0.0000 |
| PatchTST | 1 | MAE 0.0214 ± 0.0000<br>RMSE 0.0290 ± 0.0000<br>MAPE 0.0227 ± 0.0000 | MAE 0.0319 ± 0.0000<br>RMSE 0.0870 ± 0.0000<br>MAPE 0.0537 ± 0.0000 | MAE 0.1024 ± 0.0000<br>RMSE 0.1159 ± 0.0000<br>MAPE 0.1754 ± 0.0000 | MAE 0.1246 ± 0.0000<br>RMSE 0.1416 ± 0.0000<br>MAPE 0.4349 ± 0.0000 |
| Severson | 1 | MAE 0.0525 ± 0.0000<br>RMSE 0.0612 ± 0.0000<br>MAPE 0.0548 ± 0.0000 | MAE 0.1379 ± 0.0000<br>RMSE 0.4317 ± 0.0000<br>MAPE 0.1551 ± 0.0000 | MAE 0.1378 ± 0.0000<br>RMSE 0.1495 ± 0.0000<br>MAPE 0.2380 ± 0.0000 | MAE 0.1260 ± 0.0000<br>RMSE 0.1348 ± 0.0000<br>MAPE 0.6774 ± 0.0000 |
| TimeMixer | 1 | MAE 0.0157 ± 0.0000<br>RMSE 0.0208 ± 0.0000<br>MAPE 0.0167 ± 0.0000 | MAE 0.0179 ± 0.0000<br>RMSE 0.0264 ± 0.0000<br>MAPE 0.0233 ± 0.0000 | MAE 0.0299 ± 0.0000<br>RMSE 0.0359 ± 0.0000<br>MAPE 0.0464 ± 0.0000 | MAE 0.1717 ± 0.0000<br>RMSE 0.2312 ± 0.0000<br>MAPE 0.6766 ± 0.0000 |
| Transformer | 1 | MAE 0.0246 ± 0.0000<br>RMSE 0.0316 ± 0.0000<br>MAPE 0.0260 ± 0.0000 | MAE 0.0159 ± 0.0000<br>RMSE 0.0288 ± 0.0000<br>MAPE 0.0227 ± 0.0000 | MAE 0.0413 ± 0.0000<br>RMSE 0.0532 ± 0.0000<br>MAPE 0.0730 ± 0.0000 | MAE 0.2033 ± 0.0000<br>RMSE 0.2105 ± 0.0000<br>MAPE 0.3207 ± 0.0000 |

### Four-Level — SOH Trajectory

| Model | Seeds | L1 | L2 | L3 | L4 |
|---|---:|---|---|---|---|
| Autoformer | 5 | MAE 0.0368 ± 0.0029<br>RMSE 0.0436 ± 0.0027<br>MAPE 0.0383 ± 0.0028 | MAE 0.0245 ± 0.0019<br>RMSE 0.0319 ± 0.0020<br>MAPE 0.0268 ± 0.0020 | MAE 0.0523 ± 0.0027<br>RMSE 0.0580 ± 0.0022<br>MAPE 0.0611 ± 0.0034 | MAE 0.0643 ± 0.0008<br>RMSE 0.0734 ± 0.0010<br>MAPE 1.2522 ± 0.0078 |
| BatLiNet |  |  |  |  |  |
| BiGRU | 5 | MAE 0.0204 ± 0.0016<br>RMSE 0.0308 ± 0.0018<br>MAPE 0.0217 ± 0.0016 | MAE 0.0226 ± 0.0023<br>RMSE 0.0304 ± 0.0022<br>MAPE 0.0248 ± 0.0024 | MAE 0.0599 ± 0.0081<br>RMSE 0.0647 ± 0.0073<br>MAPE 0.0694 ± 0.0095 | MAE 0.0706 ± 0.0056<br>RMSE 0.0816 ± 0.0082<br>MAPE 1.1700 ± 0.0155 |
| BiLSTM | 5 | MAE 0.0208 ± 0.0018<br>RMSE 0.0314 ± 0.0016<br>MAPE 0.0222 ± 0.0017 | MAE 0.0220 ± 0.0014<br>RMSE 0.0286 ± 0.0015<br>MAPE 0.0240 ± 0.0014 | MAE 0.0551 ± 0.0036<br>RMSE 0.0602 ± 0.0037<br>MAPE 0.0639 ± 0.0042 | MAE 0.0790 ± 0.0072<br>RMSE 0.0949 ± 0.0104<br>MAPE 1.2293 ± 0.0316 |
| CNN | 5 | MAE 0.0212 ± 0.0016<br>RMSE 0.0294 ± 0.0014<br>MAPE 0.0227 ± 0.0016 | MAE 0.0187 ± 0.0027<br>RMSE 0.0272 ± 0.0018<br>MAPE 0.0207 ± 0.0028 | MAE 0.0430 ± 0.0057<br>RMSE 0.0473 ± 0.0058<br>MAPE 0.0501 ± 0.0070 | MAE 0.1171 ± 0.0454<br>RMSE 0.1411 ± 0.0538<br>MAPE 1.3217 ± 0.0775 |
| DLinear | 5 | MAE 0.0623 ± 0.0016<br>RMSE 0.0669 ± 0.0018<br>MAPE 0.0639 ± 0.0016 | MAE 0.0231 ± 0.0012<br>RMSE 0.0311 ± 0.0005<br>MAPE 0.0255 ± 0.0012 | MAE 0.0669 ± 0.0016<br>RMSE 0.0733 ± 0.0017<br>MAPE 0.0788 ± 0.0018 | MAE 0.0649 ± 0.0002<br>RMSE 0.0724 ± 0.0002<br>MAPE 1.2425 ± 0.0026 |
| GRU | 5 | MAE 0.0191 ± 0.0012<br>RMSE 0.0294 ± 0.0014<br>MAPE 0.0205 ± 0.0013 | MAE 0.0248 ± 0.0025<br>RMSE 0.0327 ± 0.0020<br>MAPE 0.0271 ± 0.0025 | MAE 0.0500 ± 0.0121<br>RMSE 0.0555 ± 0.0118<br>MAPE 0.0579 ± 0.0140 | MAE 0.0728 ± 0.0026<br>RMSE 0.0838 ± 0.0041<br>MAPE 1.2214 ± 0.0336 |
| IC2ML | 5 | MAE 0.0196 ± 0.0012<br>RMSE 0.0290 ± 0.0016<br>MAPE 0.0211 ± 0.0012 | MAE 0.0195 ± 0.0022<br>RMSE 0.0278 ± 0.0027<br>MAPE 0.0216 ± 0.0025 | MAE 0.0237 ± 0.0019<br>RMSE 0.0307 ± 0.0030<br>MAPE 0.0274 ± 0.0022 | MAE 0.0442 ± 0.0011<br>RMSE 0.0607 ± 0.0009<br>MAPE 1.2047 ± 0.0154 |
| iTransformer | 5 | MAE 0.0199 ± 0.0014<br>RMSE 0.0288 ± 0.0017<br>MAPE 0.0214 ± 0.0014 | MAE 0.0231 ± 0.0031<br>RMSE 0.0311 ± 0.0026<br>MAPE 0.0255 ± 0.0032 | MAE 0.0682 ± 0.0066<br>RMSE 0.0731 ± 0.0063<br>MAPE 0.0799 ± 0.0078 | MAE 0.0690 ± 0.0035<br>RMSE 0.0820 ± 0.0046<br>MAPE 1.2897 ± 0.0169 |
| LSTM | 5 | MAE 0.0194 ± 0.0017<br>RMSE 0.0290 ± 0.0027<br>MAPE 0.0209 ± 0.0017 | MAE 0.0214 ± 0.0014<br>RMSE 0.0285 ± 0.0023<br>MAPE 0.0234 ± 0.0015 | MAE 0.0526 ± 0.0125<br>RMSE 0.0584 ± 0.0115<br>MAPE 0.0608 ± 0.0146 | MAE 0.0696 ± 0.0061<br>RMSE 0.0818 ± 0.0085<br>MAPE 1.2398 ± 0.0242 |
| MICN | 5 | MAE 0.0193 ± 0.0006<br>RMSE 0.0291 ± 0.0014<br>MAPE 0.0207 ± 0.0006 | MAE 0.0254 ± 0.0027<br>RMSE 0.0318 ± 0.0025<br>MAPE 0.0277 ± 0.0029 | MAE 0.0520 ± 0.0060<br>RMSE 0.0557 ± 0.0054<br>MAPE 0.0602 ± 0.0070 | MAE 0.2492 ± 0.0225<br>RMSE 0.2747 ± 0.0249<br>MAPE 1.4035 ± 0.0332 |
| MLP | 5 | MAE 0.0310 ± 0.0045<br>RMSE 0.0381 ± 0.0043<br>MAPE 0.0323 ± 0.0044 | MAE 0.0280 ± 0.0033<br>RMSE 0.0372 ± 0.0039<br>MAPE 0.0308 ± 0.0036 | MAE 0.0377 ± 0.0039<br>RMSE 0.0438 ± 0.0039<br>MAPE 0.0442 ± 0.0045 | MAE 0.1328 ± 0.0443<br>RMSE 0.1536 ± 0.0492<br>MAPE 1.2784 ± 0.0447 |
| PatchTST | 5 | MAE 0.0561 ± 0.0045<br>RMSE 0.0717 ± 0.0049<br>MAPE 0.0575 ± 0.0045 | MAE 0.0398 ± 0.0033<br>RMSE 0.0499 ± 0.0036<br>MAPE 0.0429 ± 0.0035 | MAE 0.0590 ± 0.0020<br>RMSE 0.0687 ± 0.0015<br>MAPE 0.0686 ± 0.0023 | MAE 0.0813 ± 0.0028<br>RMSE 0.0928 ± 0.0030<br>MAPE 1.1713 ± 0.0109 |
| Severson | 5 | MAE 0.0480 ± 0.0000<br>RMSE 0.0491 ± 0.0000<br>MAPE 0.0496 ± 0.0000 | MAE 0.0238 ± 0.0000<br>RMSE 0.0550 ± 0.0000<br>MAPE 0.0255 ± 0.0000 | MAE 0.0467 ± 0.0000<br>RMSE 0.0492 ± 0.0000<br>MAPE 0.0542 ± 0.0000 | MAE 0.0424 ± 0.0000<br>RMSE 0.0503 ± 0.0000<br>MAPE 6.3073 ± 0.0000 |
| TimeMixer | 5 | MAE 0.0187 ± 0.0012<br>RMSE 0.0273 ± 0.0010<br>MAPE 0.0199 ± 0.0013 | MAE 0.0224 ± 0.0020<br>RMSE 0.0306 ± 0.0028<br>MAPE 0.0246 ± 0.0022 | MAE 0.0674 ± 0.0065<br>RMSE 0.0723 ± 0.0065<br>MAPE 0.0782 ± 0.0077 | MAE 0.1438 ± 0.0338<br>RMSE 0.1654 ± 0.0348<br>MAPE 1.3417 ± 0.0644 |
| Transformer | 5 | MAE 0.0211 ± 0.0024<br>RMSE 0.0320 ± 0.0020<br>MAPE 0.0226 ± 0.0024 | MAE 0.0224 ± 0.0020<br>RMSE 0.0307 ± 0.0017<br>MAPE 0.0247 ± 0.0021 | MAE 0.0731 ± 0.0061<br>RMSE 0.0766 ± 0.0063<br>MAPE 0.0845 ± 0.0073 | MAE 0.0825 ± 0.0040<br>RMSE 0.0998 ± 0.0051<br>MAPE 1.3057 ± 0.0176 |

## Project Structure

```text
BatteryBench/
├── asset/                  project artwork
├── configs/                default and domain configurations
├── data/                   local raw and processed datasets
├── results/                metrics and evaluation outputs
├── scripts/                train, evaluate, and pipeline entry points
├── src/
│   ├── data/               datasets and preprocessing
│   ├── evaluate/           task-specific evaluators and writers
│   ├── models/             model registry and implementations
│   └── train/              task-specific training loops
├── tests/                  pipeline and output regression tests
├── requirements.txt
└── run_pipeline.sh
```

## TODO

- [ ] Finish the remaining model training and evaluation jobs.
- [ ] Publish the final five-seed summaries after every experiment has completed.
