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

- 📊 **[2026/09/28]** Published complete five-seed results for 15 baselines across Four-Level, Li-ion, CALB, Na-ion, and Zn-ion benchmarks.
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

Unobserved history positions are zero-filled and excluded through <code>curve_attn_mask</code>. SOH trajectory loss and metrics are computed only after the last observed cycle. No task uses discharge curves, and SOH Point does not expose charge or discharge capacity to any baseline, including battery-specific methods.

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
| Battery-specific | IC2ML, Severson |

## Installation

    conda create -n BatteryBench python=3.11
    conda activate BatteryBench
    pip install -r requirements.txt

Place raw datasets under <code>data/raw/</code>. Dataset paths and experiment settings are defined in <code>configs/</code>.

## Quick Start

Train and evaluate one experiment:

    python scripts/train.py --domain li_ion --model gru --task rul --seed 1 --gpu 0
    python scripts/evaluate.py --domain li_ion --model gru --task rul --seed 1 --gpu 0

Run or resume a complete data split:

    bash run_li_ion.sh
    bash run_calb.sh
    bash run_na_ion.sh
    bash run_zn_ion.sh
    bash run_four_level.sh

Each script covers all tasks, models, and seeds 1–5 for its split. Ordinary models use two slots on GPU 0 and three slots on GPUs 1–3. Recovery checks complete checkpoints and evaluation artifacts, so finished experiments are skipped.

## Results

The complete summary covers five domains, three tasks, 15 models, and seeds 1–5: 1,125 complete experiments. Values are **mean ± population standard deviation** over five seeds. Underlining marks the best mean in each metric column; lower is better for RMSE, MAE, and MAPE. MAPE is shown as a percentage.

### Four-Level

Four-Level tables use a two-row header: the first row gives the generalization level and the second gives RMSE, MAE, and MAPE. RUL L1 has no valid test samples and is left empty.

#### SOH Trajectory

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="3">L1</th>
<th colspan="3">L2</th>
<th colspan="3">L3</th>
<th colspan="3">L4</th>
</tr>
<tr>
<th>RMSE</th><th>MAE</th><th>MAPE</th>
<th>RMSE</th><th>MAE</th><th>MAPE</th>
<th>RMSE</th><th>MAE</th><th>MAPE</th>
<th>RMSE</th><th>MAE</th><th>MAPE</th>
</tr>
</thead>
<tbody>
<tr>
<td>Autoformer</td>
<td>0.0480 ± 0.0020</td>
<td>0.0416 ± 0.0019</td>
<td>4.32 ± 0.18%</td>
<td>0.0369 ± 0.0038</td>
<td>0.0286 ± 0.0038</td>
<td>3.12 ± 0.40%</td>
<td>0.0777 ± 0.0054</td>
<td>0.0711 ± 0.0051</td>
<td>8.35 ± 0.60%</td>
<td>0.0850 ± 0.0038</td>
<td>0.0727 ± 0.0029</td>
<td>128.07 ± 1.59%</td>
</tr>
<tr>
<td>BiGRU</td>
<td>0.0307 ± 0.0005</td>
<td>0.0214 ± 0.0007</td>
<td>2.27 ± 0.07%</td>
<td>0.0343 ± 0.0023</td>
<td>0.0260 ± 0.0020</td>
<td>2.84 ± 0.22%</td>
<td>0.0540 ± 0.0089</td>
<td>0.0481 ± 0.0085</td>
<td>5.47 ± 1.02%</td>
<td>0.0927 ± 0.0080</td>
<td>0.0777 ± 0.0050</td>
<td>130.09 ± 1.32%</td>
</tr>
<tr>
<td>BiLSTM</td>
<td>0.0301 ± 0.0017</td>
<td>0.0204 ± 0.0015</td>
<td>2.17 ± 0.14%</td>
<td>0.0320 ± 0.0017</td>
<td>0.0233 ± 0.0020</td>
<td>2.56 ± 0.21%</td>
<td>0.0548 ± 0.0054</td>
<td>0.0493 ± 0.0059</td>
<td>5.62 ± 0.72%</td>
<td>0.0794 ± 0.0060</td>
<td>0.0672 ± 0.0036</td>
<td>127.69 ± 3.17%</td>
</tr>
<tr>
<td>CNN</td>
<td>0.0292 ± 0.0016</td>
<td>0.0211 ± 0.0021</td>
<td>2.25 ± 0.21%</td>
<td><u>0.0261 ± 0.0010</u></td>
<td><u>0.0185 ± 0.0017</u></td>
<td><u>2.04 ± 0.18%</u></td>
<td>0.0526 ± 0.0045</td>
<td>0.0478 ± 0.0045</td>
<td>5.50 ± 0.51%</td>
<td>0.1814 ± 0.0918</td>
<td>0.1500 ± 0.0768</td>
<td>143.71 ± 9.76%</td>
</tr>
<tr>
<td>DLinear</td>
<td>0.0471 ± 0.0025</td>
<td>0.0413 ± 0.0028</td>
<td>4.30 ± 0.29%</td>
<td>0.0492 ± 0.0023</td>
<td>0.0376 ± 0.0025</td>
<td>4.16 ± 0.27%</td>
<td>0.0669 ± 0.0017</td>
<td>0.0601 ± 0.0017</td>
<td>7.09 ± 0.20%</td>
<td>0.1837 ± 0.0066</td>
<td>0.1661 ± 0.0064</td>
<td>137.85 ± 0.82%</td>
</tr>
<tr>
<td>GRU</td>
<td>0.0286 ± 0.0015</td>
<td>0.0187 ± 0.0018</td>
<td>2.02 ± 0.17%</td>
<td>0.0361 ± 0.0053</td>
<td>0.0273 ± 0.0044</td>
<td>2.98 ± 0.47%</td>
<td>0.0674 ± 0.0060</td>
<td>0.0588 ± 0.0058</td>
<td>6.73 ± 0.67%</td>
<td>0.0867 ± 0.0080</td>
<td>0.0727 ± 0.0070</td>
<td>129.18 ± 0.97%</td>
</tr>
<tr>
<td>IC2ML</td>
<td>0.0278 ± 0.0016</td>
<td>0.0186 ± 0.0016</td>
<td>2.00 ± 0.17%</td>
<td>0.0288 ± 0.0019</td>
<td>0.0197 ± 0.0025</td>
<td>2.19 ± 0.27%</td>
<td><u>0.0435 ± 0.0112</u></td>
<td><u>0.0384 ± 0.0123</u></td>
<td><u>4.43 ± 1.49%</u></td>
<td>0.0728 ± 0.0114</td>
<td>0.0626 ± 0.0108</td>
<td>122.09 ± 1.42%</td>
</tr>
<tr>
<td>iTransformer</td>
<td>0.0285 ± 0.0017</td>
<td>0.0194 ± 0.0013</td>
<td>2.08 ± 0.12%</td>
<td>0.0353 ± 0.0041</td>
<td>0.0267 ± 0.0040</td>
<td>2.95 ± 0.44%</td>
<td>0.0857 ± 0.0064</td>
<td>0.0803 ± 0.0068</td>
<td>9.25 ± 0.81%</td>
<td><u>0.0702 ± 0.0028</u></td>
<td><u>0.0620 ± 0.0024</u></td>
<td><u>114.81 ± 0.74%</u></td>
</tr>
<tr>
<td>LSTM</td>
<td>0.0298 ± 0.0009</td>
<td>0.0212 ± 0.0014</td>
<td>2.27 ± 0.14%</td>
<td>0.0338 ± 0.0019</td>
<td>0.0256 ± 0.0015</td>
<td>2.81 ± 0.17%</td>
<td>0.0654 ± 0.0120</td>
<td>0.0605 ± 0.0123</td>
<td>6.92 ± 1.45%</td>
<td>0.0847 ± 0.0128</td>
<td>0.0701 ± 0.0116</td>
<td>129.63 ± 1.71%</td>
</tr>
<tr>
<td>MICN</td>
<td>0.0280 ± 0.0015</td>
<td>0.0194 ± 0.0016</td>
<td>2.07 ± 0.16%</td>
<td>0.0337 ± 0.0011</td>
<td>0.0267 ± 0.0013</td>
<td>2.91 ± 0.14%</td>
<td>0.0560 ± 0.0029</td>
<td>0.0514 ± 0.0029</td>
<td>5.91 ± 0.34%</td>
<td>0.5033 ± 0.0573</td>
<td>0.4691 ± 0.0451</td>
<td>176.16 ± 7.61%</td>
</tr>
<tr>
<td>MLP</td>
<td>0.0329 ± 0.0010</td>
<td>0.0258 ± 0.0009</td>
<td>2.73 ± 0.10%</td>
<td>0.0359 ± 0.0046</td>
<td>0.0288 ± 0.0042</td>
<td>3.14 ± 0.45%</td>
<td>0.0512 ± 0.0033</td>
<td>0.0458 ± 0.0033</td>
<td>5.36 ± 0.40%</td>
<td>0.1155 ± 0.0464</td>
<td>0.0989 ± 0.0402</td>
<td>126.95 ± 6.97%</td>
</tr>
<tr>
<td>PatchTST</td>
<td>0.0612 ± 0.0041</td>
<td>0.0473 ± 0.0038</td>
<td>4.87 ± 0.39%</td>
<td>0.0421 ± 0.0062</td>
<td>0.0334 ± 0.0049</td>
<td>3.61 ± 0.51%</td>
<td>0.1084 ± 0.0088</td>
<td>0.0974 ± 0.0083</td>
<td>11.38 ± 0.98%</td>
<td>0.0782 ± 0.0074</td>
<td>0.0650 ± 0.0059</td>
<td>123.52 ± 2.72%</td>
</tr>
<tr>
<td>Severson</td>
<td>0.0571 ± 0.0000</td>
<td>0.0432 ± 0.0000</td>
<td>4.53 ± 0.00%</td>
<td>0.0585 ± 0.0000</td>
<td>0.0428 ± 0.0000</td>
<td>4.62 ± 0.00%</td>
<td>0.1372 ± 0.0000</td>
<td>0.1184 ± 0.0000</td>
<td>13.85 ± 0.00%</td>
<td>2.1315 ± 0.0000</td>
<td>1.8763 ± 0.0000</td>
<td>279.89 ± 0.00%</td>
</tr>
<tr>
<td>TimeMixer</td>
<td><u>0.0277 ± 0.0010</u></td>
<td><u>0.0183 ± 0.0011</u></td>
<td><u>1.97 ± 0.11%</u></td>
<td>0.0321 ± 0.0035</td>
<td>0.0231 ± 0.0031</td>
<td>2.54 ± 0.32%</td>
<td>0.0466 ± 0.0071</td>
<td>0.0406 ± 0.0082</td>
<td>4.66 ± 0.99%</td>
<td>0.3753 ± 0.1287</td>
<td>0.3342 ± 0.1190</td>
<td>159.20 ± 13.33%</td>
</tr>
<tr>
<td>Transformer</td>
<td>0.0286 ± 0.0022</td>
<td>0.0184 ± 0.0020</td>
<td>1.98 ± 0.21%</td>
<td>0.0290 ± 0.0013</td>
<td>0.0210 ± 0.0012</td>
<td>2.31 ± 0.13%</td>
<td>0.0513 ± 0.0040</td>
<td>0.0463 ± 0.0037</td>
<td>5.32 ± 0.44%</td>
<td>0.0784 ± 0.0070</td>
<td>0.0680 ± 0.0049</td>
<td>126.13 ± 1.44%</td>
</tr>
</tbody>
</table>

#### SOH Point

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="3">L1</th>
<th colspan="3">L2</th>
<th colspan="3">L3</th>
<th colspan="3">L4</th>
</tr>
<tr>
<th>RMSE</th><th>MAE</th><th>MAPE</th>
<th>RMSE</th><th>MAE</th><th>MAPE</th>
<th>RMSE</th><th>MAE</th><th>MAPE</th>
<th>RMSE</th><th>MAE</th><th>MAPE</th>
</tr>
</thead>
<tbody>
<tr>
<td>Autoformer</td>
<td>0.3893 ± 0.0019</td>
<td>0.3857 ± 0.0019</td>
<td>39.64 ± 0.19%</td>
<td>0.3325 ± 0.0018</td>
<td>0.3221 ± 0.0018</td>
<td>35.44 ± 0.19%</td>
<td>0.2149 ± 0.0010</td>
<td>0.1894 ± 0.0009</td>
<td>27.24 ± 0.06%</td>
<td>0.3068 ± 0.0016</td>
<td>0.2801 ± 0.0015</td>
<td><u>114.78 ± 0.11%</u></td>
</tr>
<tr>
<td>BiGRU</td>
<td><u>0.0438 ± 0.0013</u></td>
<td>0.0316 ± 0.0044</td>
<td>3.41 ± 0.42%</td>
<td>0.0848 ± 0.0064</td>
<td><u>0.0368 ± 0.0017</u></td>
<td><u>6.01 ± 0.33%</u></td>
<td>0.1867 ± 0.0248</td>
<td>0.1512 ± 0.0271</td>
<td>27.21 ± 4.49%</td>
<td><u>0.1692 ± 0.0063</u></td>
<td>0.1362 ± 0.0112</td>
<td>165.38 ± 5.27%</td>
</tr>
<tr>
<td>BiLSTM</td>
<td>0.0479 ± 0.0046</td>
<td>0.0331 ± 0.0054</td>
<td>3.62 ± 0.55%</td>
<td>0.1012 ± 0.0161</td>
<td>0.0517 ± 0.0125</td>
<td>7.91 ± 1.63%</td>
<td>0.1754 ± 0.0241</td>
<td>0.1403 ± 0.0233</td>
<td>25.34 ± 4.46%</td>
<td>0.2018 ± 0.0161</td>
<td>0.1437 ± 0.0123</td>
<td>160.88 ± 3.94%</td>
</tr>
<tr>
<td>CNN</td>
<td>0.0447 ± 0.0015</td>
<td><u>0.0296 ± 0.0008</u></td>
<td><u>3.24 ± 0.07%</u></td>
<td><u>0.0828 ± 0.0060</u></td>
<td>0.0446 ± 0.0056</td>
<td>6.70 ± 0.73%</td>
<td>0.1640 ± 0.0145</td>
<td>0.1254 ± 0.0118</td>
<td>23.71 ± 2.25%</td>
<td>0.2938 ± 0.0965</td>
<td>0.2553 ± 0.0890</td>
<td>175.03 ± 11.05%</td>
</tr>
<tr>
<td>DLinear</td>
<td>0.1467 ± 0.0104</td>
<td>0.0887 ± 0.0062</td>
<td>9.59 ± 0.67%</td>
<td>0.3742 ± 0.0174</td>
<td>0.2720 ± 0.0143</td>
<td>31.74 ± 1.60%</td>
<td>0.5832 ± 0.0447</td>
<td>0.5336 ± 0.0429</td>
<td>69.60 ± 5.36%</td>
<td>4.1949 ± 0.7657</td>
<td>4.1385 ± 0.7526</td>
<td>613.01 ± 125.15%</td>
</tr>
<tr>
<td>GRU</td>
<td>0.3876 ± 0.0012</td>
<td>0.3841 ± 0.0012</td>
<td>39.46 ± 0.13%</td>
<td>0.3310 ± 0.0012</td>
<td>0.3205 ± 0.0012</td>
<td>35.27 ± 0.13%</td>
<td>0.2140 ± 0.0007</td>
<td>0.1886 ± 0.0006</td>
<td>27.18 ± 0.04%</td>
<td>0.3054 ± 0.0010</td>
<td>0.2788 ± 0.0010</td>
<td>114.88 ± 0.08%</td>
</tr>
<tr>
<td>IC2ML</td>
<td>0.0451 ± 0.0040</td>
<td>0.0312 ± 0.0066</td>
<td>3.39 ± 0.65%</td>
<td>0.0907 ± 0.0077</td>
<td>0.0498 ± 0.0105</td>
<td>7.48 ± 1.26%</td>
<td>0.1727 ± 0.0515</td>
<td>0.1403 ± 0.0560</td>
<td>24.01 ± 7.92%</td>
<td>0.1744 ± 0.0255</td>
<td><u>0.1303 ± 0.0155</u></td>
<td>140.44 ± 17.42%</td>
</tr>
<tr>
<td>iTransformer</td>
<td>0.0481 ± 0.0018</td>
<td>0.0326 ± 0.0066</td>
<td>3.55 ± 0.62%</td>
<td>0.0976 ± 0.0056</td>
<td>0.0496 ± 0.0063</td>
<td>7.56 ± 0.74%</td>
<td>0.1827 ± 0.0154</td>
<td>0.1381 ± 0.0137</td>
<td>26.30 ± 2.58%</td>
<td>0.1803 ± 0.0139</td>
<td>0.1305 ± 0.0054</td>
<td>152.82 ± 5.04%</td>
</tr>
<tr>
<td>LSTM</td>
<td>0.3882 ± 0.0015</td>
<td>0.3847 ± 0.0015</td>
<td>39.53 ± 0.16%</td>
<td>0.3316 ± 0.0015</td>
<td>0.3211 ± 0.0015</td>
<td>35.34 ± 0.16%</td>
<td>0.2143 ± 0.0008</td>
<td>0.1889 ± 0.0007</td>
<td>27.20 ± 0.05%</td>
<td>0.3059 ± 0.0013</td>
<td>0.2793 ± 0.0012</td>
<td>114.85 ± 0.09%</td>
</tr>
<tr>
<td>MICN</td>
<td>0.0471 ± 0.0025</td>
<td>0.0354 ± 0.0064</td>
<td>3.80 ± 0.59%</td>
<td>0.1066 ± 0.0103</td>
<td>0.0549 ± 0.0056</td>
<td>8.37 ± 0.79%</td>
<td>0.1710 ± 0.0043</td>
<td>0.1318 ± 0.0049</td>
<td>24.83 ± 0.86%</td>
<td>0.1869 ± 0.0204</td>
<td>0.1472 ± 0.0254</td>
<td>160.30 ± 1.58%</td>
</tr>
<tr>
<td>MLP</td>
<td>0.0498 ± 0.0035</td>
<td>0.0360 ± 0.0086</td>
<td>3.89 ± 0.82%</td>
<td>0.1130 ± 0.0084</td>
<td>0.0593 ± 0.0073</td>
<td>8.98 ± 0.90%</td>
<td><u>0.1601 ± 0.0192</u></td>
<td><u>0.1219 ± 0.0209</u></td>
<td>22.26 ± 3.16%</td>
<td>0.6440 ± 0.3080</td>
<td>0.6155 ± 0.3078</td>
<td>145.01 ± 48.09%</td>
</tr>
<tr>
<td>PatchTST</td>
<td>0.0455 ± 0.0042</td>
<td>0.0316 ± 0.0022</td>
<td>3.44 ± 0.26%</td>
<td>0.1046 ± 0.0153</td>
<td>0.0683 ± 0.0189</td>
<td>9.26 ± 2.00%</td>
<td>0.1986 ± 0.0207</td>
<td>0.1624 ± 0.0215</td>
<td>29.49 ± 3.57%</td>
<td>0.1939 ± 0.0286</td>
<td>0.1588 ± 0.0338</td>
<td>135.72 ± 9.31%</td>
</tr>
<tr>
<td>Severson</td>
<td>0.1570 ± 0.0000</td>
<td>0.1491 ± 0.0000</td>
<td>15.16 ± 0.00%</td>
<td>0.1906 ± 0.0000</td>
<td>0.1721 ± 0.0000</td>
<td>21.49 ± 0.00%</td>
<td>0.1687 ± 0.0000</td>
<td>0.1488 ± 0.0000</td>
<td><u>21.03 ± 0.00%</u></td>
<td>0.6932 ± 0.0000</td>
<td>0.6589 ± 0.0000</td>
<td>214.22 ± 0.00%</td>
</tr>
<tr>
<td>TimeMixer</td>
<td>0.0463 ± 0.0031</td>
<td>0.0353 ± 0.0073</td>
<td>3.78 ± 0.70%</td>
<td>0.0947 ± 0.0052</td>
<td>0.0456 ± 0.0025</td>
<td>7.10 ± 0.35%</td>
<td>0.1828 ± 0.0222</td>
<td>0.1555 ± 0.0236</td>
<td>25.68 ± 2.90%</td>
<td>1.0872 ± 0.6996</td>
<td>1.0268 ± 0.7108</td>
<td>220.77 ± 56.96%</td>
</tr>
<tr>
<td>Transformer</td>
<td>0.0508 ± 0.0025</td>
<td>0.0321 ± 0.0030</td>
<td>3.54 ± 0.28%</td>
<td>0.0950 ± 0.0149</td>
<td>0.0463 ± 0.0063</td>
<td>7.13 ± 1.03%</td>
<td>0.1937 ± 0.0263</td>
<td>0.1547 ± 0.0267</td>
<td>28.48 ± 4.73%</td>
<td>0.1905 ± 0.0281</td>
<td>0.1475 ± 0.0360</td>
<td>143.49 ± 12.85%</td>
</tr>
</tbody>
</table>

#### RUL

<table>
<thead>
<tr>
<th rowspan="2">Model</th>
<th colspan="3">L1</th>
<th colspan="3">L2</th>
<th colspan="3">L3</th>
<th colspan="3">L4</th>
</tr>
<tr>
<th>RMSE</th><th>MAE</th><th>MAPE</th>
<th>RMSE</th><th>MAE</th><th>MAPE</th>
<th>RMSE</th><th>MAE</th><th>MAPE</th>
<th>RMSE</th><th>MAE</th><th>MAPE</th>
</tr>
</thead>
<tbody>
<tr>
<td>Autoformer</td>
<td></td>
<td></td>
<td></td>
<td>303.6639 ± 33.2659</td>
<td>263.1750 ± 33.6629</td>
<td>32.45 ± 4.34%</td>
<td>277.2419 ± 20.5152</td>
<td>223.9837 ± 20.5496</td>
<td>76.84 ± 6.63%</td>
<td>497.2503 ± 25.7991</td>
<td>470.0591 ± 25.1792</td>
<td>274.29 ± 18.45%</td>
</tr>
<tr>
<td>BiGRU</td>
<td></td>
<td></td>
<td></td>
<td>298.9698 ± 23.3529</td>
<td>260.8443 ± 20.5333</td>
<td>32.42 ± 2.87%</td>
<td>695.6169 ± 317.0543</td>
<td>575.5553 ± 215.4128</td>
<td>210.99 ± 88.39%</td>
<td>576.1844 ± 152.1226</td>
<td>549.5131 ± 147.8781</td>
<td>324.89 ± 94.77%</td>
</tr>
<tr>
<td>BiLSTM</td>
<td></td>
<td></td>
<td></td>
<td>339.9262 ± 14.6521</td>
<td>302.7785 ± 21.7836</td>
<td>37.28 ± 3.13%</td>
<td>468.7000 ± 87.2975</td>
<td>423.3654 ± 110.8046</td>
<td>150.79 ± 46.97%</td>
<td>493.1609 ± 17.7687</td>
<td>468.7247 ± 21.3394</td>
<td>274.53 ± 14.97%</td>
</tr>
<tr>
<td>CNN</td>
<td></td>
<td></td>
<td></td>
<td>300.0389 ± 12.3066</td>
<td>270.8816 ± 17.6545</td>
<td>33.98 ± 2.78%</td>
<td>385.4259 ± 34.2367</td>
<td>363.8583 ± 35.0047</td>
<td>132.51 ± 11.81%</td>
<td>1046.0253 ± 909.7172</td>
<td>911.1057 ± 838.0793</td>
<td>511.69 ± 479.22%</td>
</tr>
<tr>
<td>DLinear</td>
<td></td>
<td></td>
<td></td>
<td>496.7614 ± 51.2077</td>
<td>469.6271 ± 53.6518</td>
<td>60.08 ± 7.43%</td>
<td>427.8703 ± 9.9304</td>
<td>417.6275 ± 9.5408</td>
<td>145.36 ± 3.36%</td>
<td>1209.4298 ± 175.9599</td>
<td>1120.5731 ± 179.3002</td>
<td>528.37 ± 95.90%</td>
</tr>
<tr>
<td>GRU</td>
<td></td>
<td></td>
<td></td>
<td>333.7321 ± 21.0722</td>
<td>298.3029 ± 24.7871</td>
<td>36.86 ± 3.34%</td>
<td>368.6274 ± 41.3275</td>
<td>341.9509 ± 50.5913</td>
<td>117.50 ± 19.46%</td>
<td>584.8020 ± 105.3485</td>
<td>551.4545 ± 88.0949</td>
<td>332.88 ± 64.68%</td>
</tr>
<tr>
<td>IC2ML</td>
<td></td>
<td></td>
<td></td>
<td>364.7868 ± 33.9505</td>
<td>332.5235 ± 39.1318</td>
<td>41.31 ± 6.16%</td>
<td>230.6151 ± 44.5424</td>
<td>184.7533 ± 52.5924</td>
<td><u>56.93 ± 21.27%</u></td>
<td>413.2107 ± 47.2969</td>
<td><u>358.3383 ± 60.7985</u></td>
<td><u>186.66 ± 48.34%</u></td>
</tr>
<tr>
<td>iTransformer</td>
<td></td>
<td></td>
<td></td>
<td>295.0410 ± 33.0197</td>
<td>252.5808 ± 27.2628</td>
<td>30.77 ± 2.92%</td>
<td>348.4696 ± 45.4669</td>
<td>330.2482 ± 50.3869</td>
<td>109.01 ± 20.50%</td>
<td>1502.6243 ± 846.1682</td>
<td>1426.7870 ± 880.3153</td>
<td>811.17 ± 492.71%</td>
</tr>
<tr>
<td>LSTM</td>
<td></td>
<td></td>
<td></td>
<td>308.4121 ± 37.2353</td>
<td>268.6224 ± 36.7905</td>
<td>32.83 ± 4.76%</td>
<td>409.7756 ± 63.6946</td>
<td>391.1326 ± 62.4872</td>
<td>141.98 ± 19.21%</td>
<td>639.9581 ± 308.6743</td>
<td>612.9893 ± 297.3166</td>
<td>362.49 ± 181.17%</td>
</tr>
<tr>
<td>MICN</td>
<td></td>
<td></td>
<td></td>
<td>328.0254 ± 43.4196</td>
<td>298.2004 ± 50.3530</td>
<td>37.40 ± 7.32%</td>
<td>344.7130 ± 39.7386</td>
<td>316.1951 ± 33.7296</td>
<td>101.66 ± 3.91%</td>
<td>10913.2146 ± 9094.2005</td>
<td>10676.7444 ± 8942.3455</td>
<td>5816.67 ± 4806.62%</td>
</tr>
<tr>
<td>MLP</td>
<td></td>
<td></td>
<td></td>
<td>293.2709 ± 35.5950</td>
<td>256.7387 ± 34.7389</td>
<td>31.08 ± 4.27%</td>
<td><u>202.9692 ± 10.7287</u></td>
<td><u>180.5774 ± 11.6868</u></td>
<td>64.86 ± 4.66%</td>
<td><u>404.8696 ± 21.0490</u></td>
<td>372.4896 ± 35.6333</td>
<td>201.36 ± 29.14%</td>
</tr>
<tr>
<td>PatchTST</td>
<td></td>
<td></td>
<td></td>
<td><u>247.4517 ± 45.4220</u></td>
<td><u>208.1265 ± 43.6308</u></td>
<td><u>25.81 ± 4.82%</u></td>
<td>526.9574 ± 165.5197</td>
<td>466.4516 ± 115.7790</td>
<td>159.55 ± 24.54%</td>
<td>443.4222 ± 53.5076</td>
<td>393.5300 ± 44.9322</td>
<td>211.68 ± 30.94%</td>
</tr>
<tr>
<td>Severson</td>
<td></td>
<td></td>
<td></td>
<td>434.8972 ± 0.0000</td>
<td>403.6044 ± 0.0000</td>
<td>50.58 ± 0.00%</td>
<td>679.5745 ± 0.0000</td>
<td>626.0155 ± 0.0000</td>
<td>204.86 ± 0.00%</td>
<td>798.7553 ± 0.0000</td>
<td>709.5113 ± 0.0000</td>
<td>300.49 ± 0.00%</td>
</tr>
<tr>
<td>TimeMixer</td>
<td></td>
<td></td>
<td></td>
<td>307.8619 ± 23.2150</td>
<td>269.3463 ± 24.9043</td>
<td>32.56 ± 3.31%</td>
<td>274.3424 ± 21.3195</td>
<td>239.1558 ± 17.8233</td>
<td>80.51 ± 7.75%</td>
<td>778.2451 ± 476.9883</td>
<td>654.1992 ± 350.1715</td>
<td>374.01 ± 206.82%</td>
</tr>
<tr>
<td>Transformer</td>
<td></td>
<td></td>
<td></td>
<td>297.8803 ± 26.8746</td>
<td>258.9072 ± 25.4522</td>
<td>31.41 ± 2.81%</td>
<td>251.7011 ± 28.0436</td>
<td>217.0306 ± 36.6630</td>
<td>64.32 ± 11.24%</td>
<td>520.2197 ± 38.5899</td>
<td>498.0100 ± 36.1977</td>
<td>295.37 ± 26.16%</td>
</tr>
</tbody>
</table>

### Standard Domains

#### Li-ion

##### SOH Trajectory

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Autoformer | 0.0436 ± 0.0016 | 0.0349 ± 0.0010 | 3.86 ± 0.12% |
| BiGRU | 0.0283 ± 0.0007 | 0.0200 ± 0.0005 | 2.22 ± 0.06% |
| BiLSTM | 0.0283 ± 0.0005 | 0.0202 ± 0.0002 | 2.24 ± 0.02% |
| CNN | 0.0308 ± 0.0004 | 0.0226 ± 0.0003 | 2.48 ± 0.03% |
| DLinear | 0.0539 ± 0.0001 | 0.0443 ± 0.0002 | 4.87 ± 0.01% |
| GRU | 0.0281 ± 0.0008 | <u>0.0199 ± 0.0007</u> | <u>2.20 ± 0.08%</u> |
| IC2ML | 0.0284 ± 0.0006 | 0.0202 ± 0.0005 | 2.23 ± 0.05% |
| iTransformer | 0.0331 ± 0.0014 | 0.0245 ± 0.0013 | 2.71 ± 0.14% |
| LSTM | 0.0304 ± 0.0022 | 0.0220 ± 0.0013 | 2.43 ± 0.14% |
| MICN | 0.0326 ± 0.0021 | 0.0237 ± 0.0012 | 2.61 ± 0.13% |
| MLP | 0.0410 ± 0.0013 | 0.0328 ± 0.0014 | 3.63 ± 0.14% |
| PatchTST | 0.0459 ± 0.0020 | 0.0350 ± 0.0016 | 3.78 ± 0.16% |
| Severson | 0.0887 ± 0.0000 | 0.0590 ± 0.0000 | 6.47 ± 0.00% |
| TimeMixer | <u>0.0281 ± 0.0006</u> | 0.0202 ± 0.0006 | 2.23 ± 0.06% |
| Transformer | 0.0303 ± 0.0009 | 0.0211 ± 0.0007 | 2.33 ± 0.08% |

##### SOH Point

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Autoformer | 0.3713 ± 0.0002 | 0.3331 ± 0.0003 | 352.28 ± 1.38% |
| BiGRU | <u>0.0729 ± 0.0057</u> | <u>0.0481 ± 0.0031</u> | 17.00 ± 1.59% |
| BiLSTM | 0.0757 ± 0.0079 | 0.0484 ± 0.0031 | <u>16.71 ± 1.51%</u> |
| CNN | 0.0781 ± 0.0025 | 0.0513 ± 0.0018 | 20.59 ± 1.92% |
| DLinear | 0.3024 ± 0.0012 | 0.2565 ± 0.0006 | 267.27 ± 4.86% |
| GRU | 0.0846 ± 0.0109 | 0.0558 ± 0.0086 | 17.86 ± 2.80% |
| IC2ML | 0.0754 ± 0.0051 | 0.0515 ± 0.0035 | 19.74 ± 2.94% |
| iTransformer | 0.0932 ± 0.0026 | 0.0628 ± 0.0023 | 21.85 ± 2.32% |
| LSTM | 0.0903 ± 0.0171 | 0.0595 ± 0.0112 | 19.55 ± 4.49% |
| MICN | 0.1534 ± 0.0010 | 0.1122 ± 0.0010 | 75.58 ± 2.79% |
| MLP | 0.0983 ± 0.0051 | 0.0684 ± 0.0024 | 28.38 ± 2.52% |
| PatchTST | 0.0998 ± 0.0033 | 0.0672 ± 0.0028 | 31.25 ± 3.28% |
| Severson | 0.2434 ± 0.0000 | 0.2027 ± 0.0000 | 181.16 ± 0.00% |
| TimeMixer | 0.0750 ± 0.0045 | 0.0519 ± 0.0022 | 20.50 ± 2.05% |
| Transformer | 0.0904 ± 0.0115 | 0.0628 ± 0.0084 | 24.45 ± 2.82% |

##### RUL

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Autoformer | 494.1630 ± 34.3603 | 283.3882 ± 14.6077 | 63.75 ± 5.26% |
| BiGRU | <u>348.7108 ± 16.5302</u> | 173.5165 ± 6.5658 | 35.48 ± 2.67% |
| BiLSTM | 363.8498 ± 19.4175 | 180.3630 ± 8.3527 | 37.74 ± 2.58% |
| CNN | 363.2264 ± 21.7098 | <u>172.1728 ± 4.5653</u> | 36.50 ± 2.38% |
| DLinear | 657.1648 ± 1.1415 | 421.1658 ± 0.3785 | 91.90 ± 0.89% |
| GRU | 365.4432 ± 25.3909 | 182.6584 ± 10.3783 | 37.45 ± 3.05% |
| IC2ML | 387.1932 ± 13.3224 | 177.6986 ± 7.7100 | <u>33.74 ± 0.96%</u> |
| iTransformer | 442.6612 ± 43.0281 | 214.3215 ± 15.4555 | 49.28 ± 7.15% |
| LSTM | 379.0283 ± 19.3657 | 192.8145 ± 8.2462 | 41.91 ± 2.25% |
| MICN | 405.4890 ± 20.1809 | 194.5459 ± 11.6446 | 45.00 ± 4.55% |
| MLP | 410.5599 ± 9.3442 | 245.1838 ± 7.7044 | 60.88 ± 4.37% |
| PatchTST | 474.4975 ± 18.4636 | 249.8008 ± 17.9458 | 51.70 ± 6.52% |
| Severson | 426.5483 ± 0.0000 | 288.8097 ± 0.0000 | 64.85 ± 0.00% |
| TimeMixer | 380.8769 ± 20.4289 | 184.7122 ± 8.9964 | 43.42 ± 2.01% |
| Transformer | 397.6769 ± 27.9441 | 179.5731 ± 9.1095 | 34.63 ± 1.00% |

#### CALB

##### SOH Trajectory

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Autoformer | 0.0062 ± 0.0003 | 0.0045 ± 0.0002 | 0.47 ± 0.02% |
| BiGRU | <u>0.0051 ± 0.0004</u> | 0.0037 ± 0.0002 | 0.39 ± 0.02% |
| BiLSTM | 0.0057 ± 0.0003 | 0.0041 ± 0.0002 | 0.43 ± 0.02% |
| CNN | 0.0055 ± 0.0004 | 0.0042 ± 0.0003 | 0.43 ± 0.03% |
| DLinear | 0.0086 ± 0.0005 | 0.0060 ± 0.0004 | 0.62 ± 0.04% |
| GRU | 0.0058 ± 0.0005 | 0.0040 ± 0.0003 | 0.41 ± 0.03% |
| IC2ML | 0.0052 ± 0.0004 | <u>0.0036 ± 0.0002</u> | <u>0.38 ± 0.03%</u> |
| iTransformer | 0.0052 ± 0.0002 | 0.0038 ± 0.0001 | 0.40 ± 0.01% |
| LSTM | 0.0056 ± 0.0001 | 0.0041 ± 0.0002 | 0.42 ± 0.03% |
| MICN | 0.0056 ± 0.0002 | 0.0040 ± 0.0001 | 0.42 ± 0.01% |
| MLP | 0.0060 ± 0.0004 | 0.0046 ± 0.0003 | 0.47 ± 0.04% |
| PatchTST | 0.0224 ± 0.0016 | 0.0158 ± 0.0014 | 1.62 ± 0.14% |
| Severson | 0.0079 ± 0.0000 | 0.0051 ± 0.0000 | 0.53 ± 0.00% |
| TimeMixer | 0.0052 ± 0.0002 | 0.0037 ± 0.0001 | 0.39 ± 0.01% |
| Transformer | 0.0053 ± 0.0003 | 0.0038 ± 0.0001 | 0.39 ± 0.01% |

##### SOH Point

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Autoformer | 0.0273 ± 0.0007 | 0.0273 ± 0.0007 | 2.73 ± 0.07% |
| BiGRU | 0.0016 ± 0.0002 | 0.0013 ± 0.0001 | 0.13 ± 0.01% |
| BiLSTM | 0.0015 ± 0.0002 | 0.0012 ± 0.0002 | 0.12 ± 0.02% |
| CNN | 0.0018 ± 0.0003 | 0.0015 ± 0.0002 | 0.15 ± 0.02% |
| DLinear | 0.0038 ± 0.0006 | 0.0030 ± 0.0005 | 0.30 ± 0.05% |
| GRU | 0.0255 ± 0.0038 | 0.0255 ± 0.0038 | 2.55 ± 0.38% |
| IC2ML | 0.0019 ± 0.0006 | 0.0017 ± 0.0007 | 0.17 ± 0.07% |
| iTransformer | 0.0025 ± 0.0009 | 0.0021 ± 0.0008 | 0.21 ± 0.08% |
| LSTM | 0.0255 ± 0.0043 | 0.0255 ± 0.0043 | 2.55 ± 0.43% |
| MICN | <u>0.0013 ± 0.0005</u> | <u>0.0010 ± 0.0005</u> | <u>0.10 ± 0.05%</u> |
| MLP | 0.0163 ± 0.0139 | 0.0159 ± 0.0141 | 1.59 ± 1.41% |
| PatchTST | 0.0022 ± 0.0005 | 0.0018 ± 0.0004 | 0.18 ± 0.04% |
| Severson | 0.0020 ± 0.0000 | 0.0018 ± 0.0000 | 0.18 ± 0.00% |
| TimeMixer | 0.0307 ± 0.0175 | 0.0305 ± 0.0176 | 3.06 ± 1.76% |
| Transformer | 0.0026 ± 0.0005 | 0.0022 ± 0.0004 | 0.22 ± 0.04% |

##### RUL

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Autoformer | 183.2930 ± 5.0500 | 164.2061 ± 3.2385 | 14.04 ± 0.27% |
| BiGRU | 160.1677 ± 10.0058 | 135.5630 ± 4.5983 | 12.58 ± 0.58% |
| BiLSTM | 166.7667 ± 11.5042 | 140.5816 ± 3.4163 | 13.05 ± 0.14% |
| CNN | 168.2221 ± 11.3398 | 148.2543 ± 7.6671 | 13.28 ± 0.47% |
| DLinear | 361.5422 ± 10.5955 | 324.0481 ± 13.1305 | 26.34 ± 1.14% |
| GRU | 164.2689 ± 8.4353 | 137.9854 ± 5.0763 | 12.83 ± 0.12% |
| IC2ML | 166.8828 ± 7.3884 | 154.9905 ± 9.9430 | 13.77 ± 0.45% |
| iTransformer | 166.7551 ± 10.1439 | 138.1514 ± 2.5486 | 12.97 ± 0.11% |
| LSTM | 158.5198 ± 1.7628 | 145.0293 ± 3.2521 | 13.12 ± 0.12% |
| MICN | 162.9383 ± 10.6030 | 136.0664 ± 2.9848 | 12.61 ± 0.29% |
| MLP | 162.4681 ± 4.0041 | 151.3387 ± 3.9288 | 13.31 ± 0.11% |
| PatchTST | 224.5182 ± 9.5085 | 181.6441 ± 8.1904 | 14.90 ± 0.67% |
| Severson | 263.6643 ± 0.0000 | 234.1276 ± 0.0000 | 21.62 ± 0.00% |
| TimeMixer | 160.7581 ± 5.2362 | 149.2538 ± 5.1948 | 13.23 ± 0.16% |
| Transformer | <u>150.5835 ± 18.6582</u> | <u>129.4043 ± 10.1495</u> | <u>11.69 ± 1.41%</u> |

#### Na-ion

##### SOH Trajectory

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Autoformer | 0.0279 ± 0.0005 | 0.0191 ± 0.0006 | 2.23 ± 0.09% |
| BiGRU | 0.0340 ± 0.0044 | 0.0253 ± 0.0027 | 2.92 ± 0.29% |
| BiLSTM | 0.0332 ± 0.0062 | 0.0228 ± 0.0041 | 2.62 ± 0.45% |
| CNN | 0.0257 ± 0.0004 | 0.0189 ± 0.0008 | 2.21 ± 0.09% |
| DLinear | 0.0357 ± 0.0001 | 0.0246 ± 0.0003 | 2.84 ± 0.04% |
| GRU | 0.0390 ± 0.0021 | 0.0270 ± 0.0019 | 3.10 ± 0.20% |
| IC2ML | <u>0.0208 ± 0.0001</u> | <u>0.0123 ± 0.0004</u> | <u>1.46 ± 0.04%</u> |
| iTransformer | 0.0361 ± 0.0012 | 0.0236 ± 0.0003 | 2.70 ± 0.04% |
| LSTM | 0.0388 ± 0.0023 | 0.0272 ± 0.0022 | 3.13 ± 0.24% |
| MICN | 0.0279 ± 0.0009 | 0.0189 ± 0.0010 | 2.23 ± 0.12% |
| MLP | 0.0366 ± 0.0007 | 0.0252 ± 0.0008 | 2.90 ± 0.09% |
| PatchTST | 0.0315 ± 0.0027 | 0.0232 ± 0.0023 | 2.68 ± 0.26% |
| Severson | 0.0239 ± 0.0000 | 0.0158 ± 0.0000 | 1.85 ± 0.00% |
| TimeMixer | 0.0373 ± 0.0055 | 0.0254 ± 0.0032 | 2.91 ± 0.35% |
| Transformer | 0.0270 ± 0.0026 | 0.0178 ± 0.0017 | 2.09 ± 0.20% |

##### SOH Point

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Autoformer | 0.0521 ± 0.0002 | 0.0394 ± 0.0001 | 4.67 ± 0.01% |
| BiGRU | 0.0439 ± 0.0019 | 0.0324 ± 0.0014 | 3.78 ± 0.14% |
| BiLSTM | 0.0455 ± 0.0026 | 0.0335 ± 0.0018 | 3.90 ± 0.17% |
| CNN | 0.0497 ± 0.0013 | 0.0371 ± 0.0004 | 4.28 ± 0.06% |
| DLinear | 0.0648 ± 0.0142 | 0.0509 ± 0.0122 | 5.88 ± 1.40% |
| GRU | 0.0430 ± 0.0005 | 0.0318 ± 0.0005 | 3.72 ± 0.06% |
| IC2ML | 0.0461 ± 0.0039 | 0.0342 ± 0.0025 | 3.97 ± 0.26% |
| iTransformer | 0.0464 ± 0.0008 | 0.0349 ± 0.0006 | 4.12 ± 0.08% |
| LSTM | 0.0448 ± 0.0035 | 0.0329 ± 0.0024 | 3.84 ± 0.26% |
| MICN | 0.0479 ± 0.0011 | 0.0353 ± 0.0009 | 4.08 ± 0.11% |
| MLP | 0.0470 ± 0.0016 | 0.0345 ± 0.0009 | 4.00 ± 0.12% |
| PatchTST | 0.0336 ± 0.0010 | <u>0.0263 ± 0.0011</u> | <u>3.13 ± 0.15%</u> |
| Severson | <u>0.0327 ± 0.0000</u> | 0.0265 ± 0.0000 | 3.14 ± 0.00% |
| TimeMixer | 0.0506 ± 0.0020 | 0.0376 ± 0.0016 | 4.35 ± 0.17% |
| Transformer | 0.0415 ± 0.0003 | 0.0307 ± 0.0002 | 3.58 ± 0.03% |

##### RUL

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Autoformer | 31.5470 ± 1.5705 | 27.9832 ± 0.5732 | 17.94 ± 0.66% |
| BiGRU | 31.1202 ± 1.3051 | 29.2437 ± 0.7538 | 17.47 ± 0.41% |
| BiLSTM | 28.6130 ± 0.8142 | 27.6486 ± 0.7217 | 16.71 ± 0.47% |
| CNN | 32.6544 ± 1.7111 | 30.2983 ± 1.0210 | 18.88 ± 0.89% |
| DLinear | 31.3505 ± 0.1125 | 25.7642 ± 0.2325 | 16.99 ± 0.07% |
| GRU | 31.9082 ± 2.2543 | 29.5268 ± 1.3410 | 17.99 ± 1.31% |
| IC2ML | 30.5636 ± 3.4352 | 28.9260 ± 2.2562 | 17.51 ± 0.87% |
| iTransformer | 31.6871 ± 3.0907 | 28.1804 ± 2.7111 | 17.22 ± 0.97% |
| LSTM | 29.8415 ± 1.5228 | 27.7123 ± 0.7196 | 16.47 ± 0.77% |
| MICN | 29.0705 ± 1.3718 | 27.1892 ± 0.7362 | 17.07 ± 0.46% |
| MLP | 30.1898 ± 1.2962 | 27.2257 ± 0.4407 | 17.20 ± 0.42% |
| PatchTST | 33.7688 ± 3.6230 | 28.8968 ± 3.2946 | 17.56 ± 2.03% |
| Severson | 32.3217 ± 0.0000 | 27.7462 ± 0.0000 | 17.81 ± 0.00% |
| TimeMixer | <u>27.6457 ± 0.7488</u> | 26.4938 ± 0.6807 | 16.41 ± 0.43% |
| Transformer | 28.1542 ± 3.6330 | <u>25.0173 ± 4.5655</u> | <u>15.18 ± 2.94%</u> |

#### Zn-ion

##### SOH Trajectory

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Autoformer | 0.0557 ± 0.0006 | 0.0431 ± 0.0004 | 956.91 ± 4.89% |
| BiGRU | 0.0573 ± 0.0006 | 0.0429 ± 0.0007 | 957.95 ± 7.12% |
| BiLSTM | 0.0569 ± 0.0006 | 0.0435 ± 0.0006 | 955.80 ± 7.19% |
| CNN | 0.0550 ± 0.0016 | 0.0409 ± 0.0004 | 964.53 ± 12.80% |
| DLinear | 0.0604 ± 0.0012 | 0.0473 ± 0.0015 | 971.38 ± 4.21% |
| GRU | 0.0572 ± 0.0011 | 0.0433 ± 0.0004 | 957.27 ± 4.14% |
| IC2ML | <u>0.0506 ± 0.0011</u> | <u>0.0372 ± 0.0010</u> | 971.51 ± 14.43% |
| iTransformer | 0.0552 ± 0.0009 | 0.0419 ± 0.0009 | 972.38 ± 7.80% |
| LSTM | 0.0566 ± 0.0004 | 0.0435 ± 0.0002 | 954.21 ± 4.33% |
| MICN | 0.0559 ± 0.0029 | 0.0411 ± 0.0030 | 974.27 ± 13.46% |
| MLP | 0.0582 ± 0.0008 | 0.0462 ± 0.0008 | 930.96 ± 8.22% |
| PatchTST | 0.0790 ± 0.0010 | 0.0632 ± 0.0006 | <u>906.69 ± 8.03%</u> |
| Severson | 0.0715 ± 0.0000 | 0.0495 ± 0.0000 | 978.99 ± 0.00% |
| TimeMixer | 0.0552 ± 0.0016 | 0.0411 ± 0.0007 | 1090.15 ± 60.71% |
| Transformer | 0.0527 ± 0.0027 | 0.0384 ± 0.0024 | 985.16 ± 52.01% |

##### SOH Point

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Autoformer | 0.1583 ± 0.0002 | 0.1302 ± 0.0007 | 723.55 ± 3.37% |
| BiGRU | 0.1295 ± 0.0152 | 0.1026 ± 0.0161 | 624.44 ± 158.36% |
| BiLSTM | 0.1192 ± 0.0225 | 0.0929 ± 0.0216 | 656.05 ± 37.49% |
| CNN | 0.0872 ± 0.0064 | 0.0655 ± 0.0051 | 390.92 ± 149.05% |
| DLinear | 0.1885 ± 0.0439 | 0.1484 ± 0.0442 | 772.92 ± 101.95% |
| GRU | 0.1412 ± 0.0025 | 0.1121 ± 0.0025 | 722.53 ± 29.48% |
| IC2ML | 0.0888 ± 0.0040 | 0.0669 ± 0.0029 | 338.77 ± 19.99% |
| iTransformer | 0.0799 ± 0.0124 | 0.0604 ± 0.0102 | 543.35 ± 92.69% |
| LSTM | 0.1415 ± 0.0049 | 0.1125 ± 0.0044 | 708.63 ± 26.00% |
| MICN | 0.1104 ± 0.0051 | 0.0821 ± 0.0035 | <u>255.23 ± 42.45%</u> |
| MLP | 0.1492 ± 0.0039 | 0.1190 ± 0.0031 | 722.56 ± 28.43% |
| PatchTST | <u>0.0743 ± 0.0036</u> | <u>0.0498 ± 0.0037</u> | 725.18 ± 37.77% |
| Severson | 0.0925 ± 0.0000 | 0.0757 ± 0.0000 | 540.61 ± 0.00% |
| TimeMixer | 0.0886 ± 0.0117 | 0.0651 ± 0.0097 | 374.38 ± 63.73% |
| Transformer | 0.1207 ± 0.0237 | 0.0929 ± 0.0212 | 611.02 ± 191.57% |

##### RUL

| Model | RMSE | MAE | MAPE |
|---|---:|---:|---:|
| Autoformer | 268.8958 ± 24.9171 | 186.5861 ± 23.2360 | 90.38 ± 14.73% |
| BiGRU | 400.3597 ± 1.2082 | 319.3447 ± 14.6099 | 145.00 ± 16.48% |
| BiLSTM | 400.5899 ± 1.8732 | 317.4677 ± 16.2167 | 142.88 ± 18.30% |
| CNN | 208.9290 ± 12.3716 | 132.0302 ± 15.2837 | 57.50 ± 10.82% |
| DLinear | 419.5346 ± 4.6148 | 287.1526 ± 7.2475 | 105.96 ± 8.44% |
| GRU | 399.9932 ± 0.7725 | 318.9993 ± 12.1273 | 144.60 ± 13.68% |
| IC2ML | 233.6156 ± 38.9599 | 119.7139 ± 20.6861 | <u>34.68 ± 6.45%</u> |
| iTransformer | 400.8350 ± 0.9193 | 331.9076 ± 8.1840 | 159.18 ± 9.23% |
| LSTM | 399.8031 ± 0.7016 | 325.4490 ± 6.4834 | 151.89 ± 7.31% |
| MICN | 263.2702 ± 90.5468 | 155.3700 ± 44.9571 | 51.73 ± 9.31% |
| MLP | 400.8128 ± 1.7592 | 322.4335 ± 17.1524 | 149.44 ± 18.15% |
| PatchTST | 315.6581 ± 6.6512 | 200.2309 ± 7.3261 | 72.22 ± 6.79% |
| Severson | <u>148.4920 ± 0.0000</u> | <u>103.5360 ± 0.0000</u> | 42.99 ± 0.00% |
| TimeMixer | 408.4130 ± 15.2340 | 295.6373 ± 27.5797 | 117.68 ± 34.29% |
| Transformer | 272.0325 ± 66.1570 | 177.0715 ± 75.5077 | 76.45 ± 39.09% |

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
    └── run_<split>.sh          Full benchmark entry points for each split

## TODO

- [ ] Extend the benchmark with additional battery chemistries and public datasets.
