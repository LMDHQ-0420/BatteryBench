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

The tables report every registered model at each Four-Level setting. Each cell is **MAE / RMSE / MAPE**, shown as **mean ± population standard deviation** across seeds 1–5. SOH MAE and RMSE use the SOH ratio; RUL errors use cycles; MAPE is a percentage. BatLiNet is blank because its runs are pending. RUL L1 is blank because that test set has no valid samples under the current RUL construction.

### SOH Trajectory

| Model | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| Autoformer | 0.0416 ± 0.0019 / 0.0480 ± 0.0020 / 4.32 ± 0.18% | 0.0286 ± 0.0038 / 0.0369 ± 0.0038 / 3.12 ± 0.40% | 0.0711 ± 0.0051 / 0.0777 ± 0.0054 / 8.35 ± 0.60% | 0.0727 ± 0.0029 / 0.0850 ± 0.0038 / 128.07 ± 1.59% |
| BatLiNet |  |  |  |  |
| BiGRU | 0.0214 ± 0.0007 / 0.0307 ± 0.0005 / 2.27 ± 0.07% | 0.0260 ± 0.0020 / 0.0343 ± 0.0023 / 2.84 ± 0.22% | 0.0481 ± 0.0085 / 0.0540 ± 0.0089 / 5.47 ± 1.02% | 0.0777 ± 0.0050 / 0.0927 ± 0.0080 / 130.09 ± 1.32% |
| BiLSTM | 0.0204 ± 0.0015 / 0.0301 ± 0.0017 / 2.17 ± 0.14% | 0.0233 ± 0.0020 / 0.0320 ± 0.0017 / 2.56 ± 0.21% | 0.0493 ± 0.0059 / 0.0548 ± 0.0054 / 5.62 ± 0.72% | 0.0672 ± 0.0036 / 0.0794 ± 0.0060 / 127.69 ± 3.17% |
| CNN | 0.0211 ± 0.0021 / 0.0292 ± 0.0016 / 2.25 ± 0.21% | 0.0185 ± 0.0017 / 0.0261 ± 0.0010 / 2.04 ± 0.18% | 0.0478 ± 0.0045 / 0.0526 ± 0.0045 / 5.50 ± 0.51% | 0.1500 ± 0.0768 / 0.1814 ± 0.0918 / 143.71 ± 9.76% |
| DLinear | 0.0413 ± 0.0028 / 0.0471 ± 0.0025 / 4.30 ± 0.29% | 0.0376 ± 0.0025 / 0.0492 ± 0.0023 / 4.16 ± 0.27% | 0.0601 ± 0.0017 / 0.0669 ± 0.0017 / 7.09 ± 0.20% | 0.1661 ± 0.0064 / 0.1837 ± 0.0066 / 137.85 ± 0.82% |
| GRU | 0.0187 ± 0.0018 / 0.0286 ± 0.0015 / 2.02 ± 0.17% | 0.0273 ± 0.0044 / 0.0361 ± 0.0053 / 2.98 ± 0.47% | 0.0588 ± 0.0058 / 0.0674 ± 0.0060 / 6.73 ± 0.67% | 0.0727 ± 0.0070 / 0.0867 ± 0.0080 / 129.18 ± 0.97% |
| IC2ML | 0.0186 ± 0.0016 / 0.0278 ± 0.0016 / 2.00 ± 0.17% | 0.0197 ± 0.0025 / 0.0288 ± 0.0019 / 2.19 ± 0.27% | 0.0384 ± 0.0123 / 0.0435 ± 0.0112 / 4.43 ± 1.49% | 0.0626 ± 0.0108 / 0.0728 ± 0.0114 / 122.09 ± 1.42% |
| iTransformer | 0.0194 ± 0.0013 / 0.0285 ± 0.0017 / 2.08 ± 0.12% | 0.0267 ± 0.0040 / 0.0353 ± 0.0041 / 2.95 ± 0.44% | 0.0803 ± 0.0068 / 0.0857 ± 0.0064 / 9.25 ± 0.81% | 0.0620 ± 0.0024 / 0.0702 ± 0.0028 / 114.81 ± 0.74% |
| LSTM | 0.0212 ± 0.0014 / 0.0298 ± 0.0009 / 2.27 ± 0.14% | 0.0256 ± 0.0015 / 0.0338 ± 0.0019 / 2.81 ± 0.17% | 0.0605 ± 0.0123 / 0.0654 ± 0.0120 / 6.92 ± 1.45% | 0.0701 ± 0.0116 / 0.0847 ± 0.0128 / 129.63 ± 1.71% |
| MICN | 0.0194 ± 0.0016 / 0.0280 ± 0.0015 / 2.07 ± 0.16% | 0.0267 ± 0.0013 / 0.0337 ± 0.0011 / 2.91 ± 0.14% | 0.0514 ± 0.0029 / 0.0560 ± 0.0029 / 5.91 ± 0.34% | 0.4691 ± 0.0451 / 0.5033 ± 0.0573 / 176.16 ± 7.61% |
| MLP | 0.0258 ± 0.0009 / 0.0329 ± 0.0010 / 2.73 ± 0.10% | 0.0288 ± 0.0042 / 0.0359 ± 0.0046 / 3.14 ± 0.45% | 0.0458 ± 0.0033 / 0.0512 ± 0.0033 / 5.36 ± 0.40% | 0.0989 ± 0.0402 / 0.1155 ± 0.0464 / 126.95 ± 6.97% |
| PatchTST | 0.0473 ± 0.0038 / 0.0612 ± 0.0041 / 4.87 ± 0.39% | 0.0334 ± 0.0049 / 0.0421 ± 0.0062 / 3.61 ± 0.51% | 0.0974 ± 0.0083 / 0.1084 ± 0.0088 / 11.38 ± 0.98% | 0.0650 ± 0.0059 / 0.0782 ± 0.0074 / 123.52 ± 2.72% |
| Severson | 0.0432 ± 0.0000 / 0.0571 ± 0.0000 / 4.53 ± 0.00% | 0.0428 ± 0.0000 / 0.0585 ± 0.0000 / 4.62 ± 0.00% | 0.1184 ± 0.0000 / 0.1372 ± 0.0000 / 13.85 ± 0.00% | 1.8763 ± 0.0000 / 2.1315 ± 0.0000 / 279.89 ± 0.00% |
| TimeMixer | 0.0183 ± 0.0011 / 0.0277 ± 0.0010 / 1.97 ± 0.11% | 0.0231 ± 0.0031 / 0.0321 ± 0.0035 / 2.54 ± 0.32% | 0.0406 ± 0.0082 / 0.0466 ± 0.0071 / 4.66 ± 0.99% | 0.3342 ± 0.1190 / 0.3753 ± 0.1287 / 159.20 ± 13.33% |
| Transformer | 0.0184 ± 0.0020 / 0.0286 ± 0.0022 / 1.98 ± 0.21% | 0.0210 ± 0.0012 / 0.0290 ± 0.0013 / 2.31 ± 0.13% | 0.0463 ± 0.0037 / 0.0513 ± 0.0040 / 5.32 ± 0.44% | 0.0680 ± 0.0049 / 0.0784 ± 0.0070 / 126.13 ± 1.44% |

### SOH Point

| Model | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| Autoformer | 0.3857 ± 0.0019 / 0.3893 ± 0.0019 / 39.64 ± 0.19% | 0.3221 ± 0.0018 / 0.3325 ± 0.0018 / 35.44 ± 0.19% | 0.1894 ± 0.0009 / 0.2149 ± 0.0010 / 27.24 ± 0.06% | 0.2801 ± 0.0015 / 0.3068 ± 0.0016 / 114.78 ± 0.11% |
| BatLiNet |  |  |  |  |
| BiGRU | 0.0316 ± 0.0044 / 0.0438 ± 0.0013 / 3.41 ± 0.42% | 0.0368 ± 0.0017 / 0.0848 ± 0.0064 / 6.01 ± 0.33% | 0.1512 ± 0.0271 / 0.1867 ± 0.0248 / 27.21 ± 4.49% | 0.1362 ± 0.0112 / 0.1692 ± 0.0063 / 165.38 ± 5.27% |
| BiLSTM | 0.0331 ± 0.0054 / 0.0479 ± 0.0046 / 3.62 ± 0.55% | 0.0517 ± 0.0125 / 0.1012 ± 0.0161 / 7.91 ± 1.63% | 0.1403 ± 0.0233 / 0.1754 ± 0.0241 / 25.34 ± 4.46% | 0.1437 ± 0.0123 / 0.2018 ± 0.0161 / 160.88 ± 3.94% |
| CNN | 0.0296 ± 0.0008 / 0.0447 ± 0.0015 / 3.24 ± 0.07% | 0.0446 ± 0.0056 / 0.0828 ± 0.0060 / 6.70 ± 0.73% | 0.1254 ± 0.0118 / 0.1640 ± 0.0145 / 23.71 ± 2.25% | 0.2553 ± 0.0890 / 0.2938 ± 0.0965 / 175.03 ± 11.05% |
| DLinear | 0.0887 ± 0.0062 / 0.1467 ± 0.0104 / 9.59 ± 0.67% | 0.2720 ± 0.0143 / 0.3742 ± 0.0174 / 31.74 ± 1.60% | 0.5336 ± 0.0429 / 0.5832 ± 0.0447 / 69.60 ± 5.36% | 4.1385 ± 0.7526 / 4.1949 ± 0.7657 / 613.01 ± 125.15% |
| GRU | 0.3841 ± 0.0012 / 0.3876 ± 0.0012 / 39.46 ± 0.13% | 0.3205 ± 0.0012 / 0.3310 ± 0.0012 / 35.27 ± 0.13% | 0.1886 ± 0.0006 / 0.2140 ± 0.0007 / 27.18 ± 0.04% | 0.2788 ± 0.0010 / 0.3054 ± 0.0010 / 114.88 ± 0.08% |
| IC2ML | 0.0312 ± 0.0066 / 0.0451 ± 0.0040 / 3.39 ± 0.65% | 0.0498 ± 0.0105 / 0.0907 ± 0.0077 / 7.48 ± 1.26% | 0.1403 ± 0.0560 / 0.1727 ± 0.0515 / 24.01 ± 7.92% | 0.1303 ± 0.0155 / 0.1744 ± 0.0255 / 140.44 ± 17.42% |
| iTransformer | 0.0326 ± 0.0066 / 0.0481 ± 0.0018 / 3.55 ± 0.62% | 0.0496 ± 0.0063 / 0.0976 ± 0.0056 / 7.56 ± 0.74% | 0.1381 ± 0.0137 / 0.1827 ± 0.0154 / 26.30 ± 2.58% | 0.1305 ± 0.0054 / 0.1803 ± 0.0139 / 152.82 ± 5.04% |
| LSTM | 0.3847 ± 0.0015 / 0.3882 ± 0.0015 / 39.53 ± 0.16% | 0.3211 ± 0.0015 / 0.3316 ± 0.0015 / 35.34 ± 0.16% | 0.1889 ± 0.0007 / 0.2143 ± 0.0008 / 27.20 ± 0.05% | 0.2793 ± 0.0012 / 0.3059 ± 0.0013 / 114.85 ± 0.09% |
| MICN | 0.0354 ± 0.0064 / 0.0471 ± 0.0025 / 3.80 ± 0.59% | 0.0549 ± 0.0056 / 0.1066 ± 0.0103 / 8.37 ± 0.79% | 0.1318 ± 0.0049 / 0.1710 ± 0.0043 / 24.83 ± 0.86% | 0.1472 ± 0.0254 / 0.1869 ± 0.0204 / 160.30 ± 1.58% |
| MLP | 0.0360 ± 0.0086 / 0.0498 ± 0.0035 / 3.89 ± 0.82% | 0.0593 ± 0.0073 / 0.1130 ± 0.0084 / 8.98 ± 0.90% | 0.1219 ± 0.0209 / 0.1601 ± 0.0192 / 22.26 ± 3.16% | 0.6155 ± 0.3078 / 0.6440 ± 0.3080 / 145.01 ± 48.09% |
| PatchTST | 0.0316 ± 0.0022 / 0.0455 ± 0.0042 / 3.44 ± 0.26% | 0.0683 ± 0.0189 / 0.1046 ± 0.0153 / 9.26 ± 2.00% | 0.1624 ± 0.0215 / 0.1986 ± 0.0207 / 29.49 ± 3.57% | 0.1588 ± 0.0338 / 0.1939 ± 0.0286 / 135.72 ± 9.31% |
| Severson | 0.1491 ± 0.0000 / 0.1570 ± 0.0000 / 15.16 ± 0.00% | 0.1721 ± 0.0000 / 0.1906 ± 0.0000 / 21.49 ± 0.00% | 0.1488 ± 0.0000 / 0.1687 ± 0.0000 / 21.03 ± 0.00% | 0.6589 ± 0.0000 / 0.6932 ± 0.0000 / 214.22 ± 0.00% |
| TimeMixer | 0.0353 ± 0.0073 / 0.0463 ± 0.0031 / 3.78 ± 0.70% | 0.0456 ± 0.0025 / 0.0947 ± 0.0052 / 7.10 ± 0.35% | 0.1555 ± 0.0236 / 0.1828 ± 0.0222 / 25.68 ± 2.90% | 1.0268 ± 0.7108 / 1.0872 ± 0.6996 / 220.77 ± 56.96% |
| Transformer | 0.0321 ± 0.0030 / 0.0508 ± 0.0025 / 3.54 ± 0.28% | 0.0463 ± 0.0063 / 0.0950 ± 0.0149 / 7.13 ± 1.03% | 0.1547 ± 0.0267 / 0.1937 ± 0.0263 / 28.48 ± 4.73% | 0.1475 ± 0.0360 / 0.1905 ± 0.0281 / 143.49 ± 12.85% |

### RUL

| Model | L1 | L2 | L3 | L4 |
|---|---|---|---|---|
| Autoformer |  | 263.1750 ± 33.6629 / 303.6639 ± 33.2659 / 32.45 ± 4.34% | 223.9837 ± 20.5496 / 277.2419 ± 20.5152 / 76.84 ± 6.63% | 470.0591 ± 25.1792 / 497.2503 ± 25.7991 / 274.29 ± 18.45% |
| BatLiNet |  |  |  |  |
| BiGRU |  | 260.8443 ± 20.5333 / 298.9698 ± 23.3529 / 32.42 ± 2.87% | 575.5553 ± 215.4128 / 695.6169 ± 317.0543 / 210.99 ± 88.39% | 549.5131 ± 147.8781 / 576.1844 ± 152.1226 / 324.89 ± 94.77% |
| BiLSTM |  | 302.7785 ± 21.7836 / 339.9262 ± 14.6521 / 37.28 ± 3.13% | 423.3654 ± 110.8046 / 468.7000 ± 87.2975 / 150.79 ± 46.97% | 468.7247 ± 21.3394 / 493.1609 ± 17.7687 / 274.53 ± 14.97% |
| CNN |  | 270.8816 ± 17.6545 / 300.0389 ± 12.3066 / 33.98 ± 2.78% | 363.8583 ± 35.0047 / 385.4259 ± 34.2367 / 132.51 ± 11.81% | 911.1057 ± 838.0793 / 1046.0253 ± 909.7172 / 511.69 ± 479.22% |
| DLinear |  | 469.6271 ± 53.6518 / 496.7614 ± 51.2077 / 60.08 ± 7.43% | 417.6275 ± 9.5408 / 427.8703 ± 9.9304 / 145.36 ± 3.36% | 1120.5731 ± 179.3002 / 1209.4298 ± 175.9599 / 528.37 ± 95.90% |
| GRU |  | 298.3029 ± 24.7871 / 333.7321 ± 21.0722 / 36.86 ± 3.34% | 341.9509 ± 50.5913 / 368.6274 ± 41.3275 / 117.50 ± 19.46% | 551.4545 ± 88.0949 / 584.8020 ± 105.3485 / 332.88 ± 64.68% |
| IC2ML |  | 332.5235 ± 39.1318 / 364.7868 ± 33.9505 / 41.31 ± 6.16% | 184.7533 ± 52.5924 / 230.6151 ± 44.5424 / 56.93 ± 21.27% | 358.3383 ± 60.7985 / 413.2107 ± 47.2969 / 186.66 ± 48.34% |
| iTransformer |  | 252.5808 ± 27.2628 / 295.0410 ± 33.0197 / 30.77 ± 2.92% | 330.2482 ± 50.3869 / 348.4696 ± 45.4669 / 109.01 ± 20.50% | 1426.7870 ± 880.3153 / 1502.6243 ± 846.1682 / 811.17 ± 492.71% |
| LSTM |  | 268.6224 ± 36.7905 / 308.4121 ± 37.2353 / 32.83 ± 4.76% | 391.1326 ± 62.4872 / 409.7756 ± 63.6946 / 141.98 ± 19.21% | 612.9893 ± 297.3166 / 639.9581 ± 308.6743 / 362.49 ± 181.17% |
| MICN |  | 298.2004 ± 50.3530 / 328.0254 ± 43.4196 / 37.40 ± 7.32% | 316.1951 ± 33.7296 / 344.7130 ± 39.7386 / 101.66 ± 3.91% | 10676.7444 ± 8942.3455 / 10913.2146 ± 9094.2005 / 5816.67 ± 4806.62% |
| MLP |  | 256.7387 ± 34.7389 / 293.2709 ± 35.5950 / 31.08 ± 4.27% | 180.5774 ± 11.6868 / 202.9692 ± 10.7287 / 64.86 ± 4.66% | 372.4896 ± 35.6333 / 404.8696 ± 21.0490 / 201.36 ± 29.14% |
| PatchTST |  | 208.1265 ± 43.6308 / 247.4517 ± 45.4220 / 25.81 ± 4.82% | 466.4516 ± 115.7790 / 526.9574 ± 165.5197 / 159.55 ± 24.54% | 393.5300 ± 44.9322 / 443.4222 ± 53.5076 / 211.68 ± 30.94% |
| Severson |  | 403.6044 ± 0.0000 / 434.8972 ± 0.0000 / 50.58 ± 0.00% | 626.0155 ± 0.0000 / 679.5745 ± 0.0000 / 204.86 ± 0.00% | 709.5113 ± 0.0000 / 798.7553 ± 0.0000 / 300.49 ± 0.00% |
| TimeMixer |  | 269.3463 ± 24.9043 / 307.8619 ± 23.2150 / 32.56 ± 3.31% | 239.1558 ± 17.8233 / 274.3424 ± 21.3195 / 80.51 ± 7.75% | 654.1992 ± 350.1715 / 778.2451 ± 476.9883 / 374.01 ± 206.82% |
| Transformer |  | 258.9072 ± 25.4522 / 297.8803 ± 26.8746 / 31.41 ± 2.81% | 217.0306 ± 36.6630 / 251.7011 ± 28.0436 / 64.32 ± 11.24% | 498.0100 ± 36.1977 / 520.2197 ± 38.5899 / 295.37 ± 26.16% |

Complete model-by-test-set results are available in [the test-set summary](results/four_level/summary_test_sets.md). The repository currently contains 284 complete experiments: all five seeds for 15 standard Four-Level baselines, plus 59 Li-ion SOH Trajectory experiments. BatLiNet and the remaining standard-domain experiments are pending.

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
