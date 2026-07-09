# BatteryBench

A clean, extensible benchmark for battery degradation prediction. BatteryBench provides a unified framework to train and evaluate a wide range of baseline models across three tasks — RUL prediction, SOH single-point estimation, and SOH degradation trajectory forecasting — on multiple battery datasets with configurable cross-domain evaluation splits.

## Tasks

| Task | Description | Output shape |
|------|-------------|--------------|
| `rul` | Remaining Useful Life prediction (cycles until EOL) | `(B, 1)` scalar |
| `soh_point` | State of Health at observation cycle | `(B, 1)` scalar ∈ [0, 1] |
| `soh_traj` | Full SOH degradation trajectory | `(B, n_future)` sequence |

## Models

Each model is implemented independently per task (no shared heads). Models reading `batch['Q']` (shape `B × S × N`, the discharge capacity–voltage matrix) unless noted.

| Category | Models |
|----------|--------|
| **MLP** | MLP, DLinear |
| **RNN** | GRU, BiGRU, LSTM, BiLSTM |
| **CNN** | CNN, MICN |
| **Transformer** | Transformer, PatchTST, Autoformer, iTransformer, TimeMixer |
| **Battery-specific** | IC²ML, BatLiNet, Severson ElasticNet |

## Datasets & Evaluation Splits

Two evaluation modes are supported, configured via `--domain`:

**Standard (random / stratified)**  
Each domain trains on one chemistry pool and evaluates with held-out splits.

| Domain | Description |
|--------|-------------|
| `li_ion` | 14 Li-ion datasets (MATR, HUST, RWTH, SDU, Stanford, Tongji, MICH, …) |
| `calb` | CALB LFP cells |
| `na_ion` | Na-ion cells |
| `zn_ion` | Zn-ion cells |

**Three-level generalization** (`--domain three_level`)  
Fixed train pool; three held-out test tiers:

| Level | Generalization | Example test sets |
|-------|---------------|-------------------|
| L1 | Same chemistry, cross-dataset | HUST_batch8, MATR_b4 |
| L2 | Li-ion, cross-cathode | CALCE, HNEI |
| L3 | Cross-ion system | NAion, ZNion |

## Evaluation Metrics

| Task | Metrics |
|------|---------|
| `rul` | MAE, MSE, RMSE, MAPE, 15%-Acc |
| `soh_point` | MAE, MSE, RMSE, MAPE |
| `soh_traj` | MAE, MSE, RMSE, MAPE |

## Data Download

All raw datasets are publicly available. Download them and place the files under `data/raw/`.

| Dataset | Link |
|---------|------|
| Zn-ion, Na-ion, CALB | [Zenodo](https://zenodo.org/records/17960956) · [HuggingFace](https://huggingface.co/datasets/Hongwxx/BatteryLife_Raw/tree/main) · [tutorial](./assets/Data_download.md#how-to-download-the-raw-data-from-huggingface) |
| CALCE | https://calce.umd.edu/battery-data |
| MATR (batches 1-3) | https://data.matr.io/1/projects/5c48dd2bc625d700019f3204 |
| MATR (batch 9) | https://data.matr.io/1/projects/5d80e633f405260001c0b60a/batches/5dcef1fe110002c7215b2c94 |
| HUST | https://data.mendeley.com/datasets/nsc7hnsg4s/2 |
| RWTH | https://publications.rwth-aachen.de/record/818642/files/Rawdata.zip |
| ISU_ILCC | https://iastate.figshare.com/articles/dataset/_b_ISU-ILCC_Battery_Aging_Dataset_b_/22582234 |
| XJTU | https://zenodo.org/records/10963339 |
| Tongji | https://zenodo.org/records/6405084 |
| Stanford | https://data.matr.io/8/ |
| HNEI, SNL, MICH, MICH_EXP, UL_PUR | https://www.batteryarchive.org/index.html |
| SDU | https://zenodo.org/records/14859405 |

## Environment Setup

```bash
conda create -n BatteryBench python=3.10
conda activate BatteryBench
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

## Quick Start

```bash
# train a single model
python scripts/train.py --domain li_ion --model gru --task rul --gpu 0

# train all models on all domains (parallel, 4 GPUs × 3 slots)
bash run_all.sh

# evaluate
python scripts/evaluate.py --domain li_ion --model gru --task rul --gpu 0
```

Results are saved to `results/<domain>/<task>/<model>/results.json`.  
Training logs go to `log/<timestamp>/<domain>_<task>_<model>.log`.

## Key Config Options (`configs/default.yaml`)

```yaml
data:
  task: rul              # rul | soh_point | soh_traj
  n_future: 100          # trajectory length for soh_traj
  split_strategy: random # random | stratified | three_level
  n_cycles: 100          # observation window length
  n_grid: 200            # voltage grid resolution

train:
  lr: 1.0e-3
  epochs: 300
  patience: 30
  use_log_rul: false     # log1p transform on RUL labels (rul task only)
```

Domain-specific overrides live in `configs/domains/<domain>.yaml` and are deep-merged on top of `default.yaml`.

## Adding a New Model

1. Create `src/models/<task>/<model_name>.py` alongside `baseline/` with a single `nn.Module`:

```python
class MyModel(nn.Module):
    def __init__(self, cfg: dict):
        super().__init__()
        m = cfg.get('model', {})
        n_future = cfg.get('data', {}).get('n_future', 100)  # for soh_traj only
        # build layers ...

    def forward(self, batch: dict):
        Q = batch['Q']   # (B, S, N)
        # ...
        return pred, None   # pred: (B,1) for rul/soh_point, (B,n_future) for soh_traj
```

2. Register it in `src/models/registry.py` under the appropriate task:

```python
'mymodel': ModelSpec(
    build_fn    = lambda cfg: MyModel(cfg),
    dataset_cls = BatteryDataset,
    train_fn    = _task_base,   # e.g. _rul_base, _sp_base, _st_base
),
```

That's all. `scripts/train.py` and `scripts/evaluate.py` pick it up automatically.

If your model needs a custom training loop (e.g. a contrastive loss), add `src/train/<task>/train_<name>.py` with the same signature as `train_base.py`:

```python
def train(model, train_loader, val_loader, config, save_path, device) -> model:
    ...
```

Then point `train_fn` in the registry to it.

## Project Structure

```
BatteryBench/
├── configs/
│   ├── default.yaml
│   └── domains/           li_ion / calb / na_ion / zn_ion / three_level
├── scripts/
│   ├── train.py           --domain --model --task --gpu
│   └── evaluate.py        --domain --model --task --gpu
├── src/
│   ├── data/dataset.py    BatteryDataset (Q, delta_q, rul, soh_point, soh_traj, lognf)
│   ├── evaluate/          rul.py  soh_point.py  soh_traj.py
│   ├── models/
│   │   ├── registry.py    get_spec(name, task) → ModelSpec
│   │   ├── rul/
│   │   │   ├── baseline/  16 models (mlp gru bigru lstm bilstm cnn dlinear patchtst
│   │   │   │              transformer autoformer itransformer micn timemixer ic2ml
│   │   │   │              batlinet severson)
│   │   │   └── mymodel.py ← your custom model goes here
│   │   ├── soh_point/
│   │   │   ├── baseline/  16 models (same as rul)
│   │   │   └── mymodel.py
│   │   └── soh_traj/
│   │       ├── baseline/  16 models (same as rul)
│   │       └── mymodel.py
│   └── train/
│       ├── rul/           train_base  train_batlinet  train_severson
│       ├── soh_point/     train_base  train_batlinet  train_severson
│       └── soh_traj/      train_base  train_batlinet  train_severson
├── run_all.sh             parallel training across all models / domains / tasks
└── STRUCTURE.md           detailed architecture and extension guide
```

## Results

All metrics reported as mean over 3 random splits (standard domains) or weighted mean over test sets per level (three-level). MAPE and ACC15 are in [0, 1] range (not ×100).

### Table 1 — Standard Domains (Li-ion / CALB / Na-ion / Zn-ion)

#### RUL — MAPE ↓ / ACC15 ↑

| Model | Li-ion<br>MAPE | Li-ion<br>ACC15 | CALB<br>MAPE | CALB<br>ACC15 | Na-ion<br>MAPE | Na-ion<br>ACC15 | Zn-ion<br>MAPE | Zn-ion<br>ACC15 |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|
| Autoformer | 0.5500 | 0.3245 | 0.2368 | 0.3170 | 0.1651 | 0.4572 | 0.6370 | 0.1623 |
| BatLiNet | 0.4063 | 0.3400 | 0.2081 | 0.4689 | 0.1834 | 0.4500 | 0.5566 | 0.2495 |
| BiGRU | 0.2620 | 0.5426 | 0.1910 | 0.4644 | 0.1955 | 0.3111 | 0.8929 | 0.0404 |
| BiLSTM | 0.2625 | 0.5143 | 0.1934 | 0.3991 | 0.1961 | 0.3450 | 0.9529 | 0.0177 |
| CNN | 0.2786 | 0.5280 | 0.1985 | 0.4402 | 0.1963 | 0.4411 | 0.6394 | 0.2060 |
| DLinear | 0.9114 | 0.1906 | 0.2154 | 0.4172 | 0.9132 | 0.1317 | 0.9858 | 0.0342 |
| GRU | 0.2726 | 0.5322 | 0.1979 | 0.3997 | 0.1840 | 0.3861 | 0.9476 | 0.0677 |
| IC2ML | 0.3840 | 0.3866 | 0.1439 | 0.7127 | 0.1565 | 0.5817 | 0.7323 | 0.0868 |
| iTransformer | 0.3673 | 0.4215 | 0.1952 | 0.3997 | 0.1924 | 0.3778 | 1.0065 | 0.0428 |
| LSTM | 0.2818 | 0.4809 | 0.1840 | 0.5317 | 0.1799 | 0.3900 | 1.0383 | 0.0400 |
| MICN | 0.3679 | 0.4288 | 0.1726 | 0.5164 | 0.1899 | 0.2922 | 0.5011 | 0.2293 |
| MLP | 0.5389 | 0.3148 | 0.2071 | 0.3621 | 0.1984 | 0.3228 | 0.9535 | 0.0609 |
| PatchTST | 0.3776 | 0.4162 | 0.2455 | 0.4468 | 0.1965 | 0.4222 | 0.3742 | 0.3230 |
| Severson | 0.9817 | 0.1685 | 0.4199 | 0.0524 | 0.1911 | 0.4456 | 1.0329 | 0.0856 |
| TimeMixer | 0.2882 | 0.5299 | 0.1997 | 0.4609 | 0.2136 | 0.2928 | 0.7990 | 0.0881 |
| Transformer | 0.2508 | 0.5360 | 0.1844 | 0.5331 | 0.2187 | 0.2222 | 0.5341 | 0.2696 |

#### SOH Point — MAPE ↓ / RMSE ↓

| Model | Li-ion<br>MAPE | Li-ion<br>RMSE | CALB<br>MAPE | CALB<br>RMSE | Na-ion<br>MAPE | Na-ion<br>RMSE | Zn-ion<br>MAPE | Zn-ion<br>RMSE |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|
| Autoformer | 0.0227 | 0.0296 | 0.0030 | 0.0076 | 0.0144 | 0.0159 | 0.0210 | 0.0344 |
| BatLiNet | 0.0191 | 0.0256 | 0.0061 | 0.0072 | 0.0072 | 0.0079 | 0.0214 | 0.0289 |
| BiGRU | 0.0126 | 0.0173 | 0.0009 | 0.0013 | 0.0115 | 0.0131 | 0.0196 | 0.0369 |
| BiLSTM | 0.0138 | 0.0187 | 0.0016 | 0.0027 | 0.0110 | 0.0126 | 0.0181 | 0.0371 |
| CNN | 0.0127 | 0.0175 | 0.0047 | 0.0060 | 0.0060 | 0.0072 | 0.0120 | 0.0199 |
| DLinear | 0.0351 | 0.0425 | 0.0026 | 0.0032 | 0.0152 | 0.0171 | 0.0207 | 0.0349 |
| GRU | 0.0120 | 0.0161 | 0.0025 | 0.0040 | 0.0142 | 0.0157 | 0.0190 | 0.0369 |
| IC2ML | 0.0131 | 0.0202 | 0.0017 | 0.0021 | 0.0093 | 0.0104 | 0.0172 | 0.0287 |
| iTransformer | 0.0224 | 0.0287 | 0.0017 | 0.0024 | 0.0155 | 0.0168 | 0.0182 | 0.0351 |
| LSTM | 0.0140 | 0.0189 | 0.0017 | 0.0039 | 0.0134 | 0.0154 | 0.0184 | 0.0373 |
| MICN | 0.0146 | 0.0197 | 0.0026 | 0.0044 | 0.0113 | 0.0133 | 0.0201 | 0.0337 |
| MLP | 0.0289 | 0.0344 | 0.0025 | 0.0032 | 0.0114 | 0.0126 | 0.0201 | 0.0337 |
| PatchTST | 0.0161 | 0.0203 | 0.0100 | 0.0153 | 0.0185 | 0.0204 | 0.0083 | 0.0147 |
| Severson | 0.0341 | 0.0399 | 0.0333 | 0.0599 | 0.0113 | 0.0127 | 0.0148 | 0.0212 |
| TimeMixer | 0.0153 | 0.0204 | 0.0022 | 0.0029 | 0.0103 | 0.0117 | 0.0179 | 0.0317 |
| Transformer | 0.0132 | 0.0181 | 0.0009 | 0.0011 | 0.0084 | 0.0091 | 0.0191 | 0.0361 |

#### SOH Trajectory — MAPE ↓ / RMSE ↓

| Model | Li-ion<br>MAPE | Li-ion<br>RMSE | CALB<br>MAPE | CALB<br>RMSE | Na-ion<br>MAPE | Na-ion<br>RMSE | Zn-ion<br>MAPE | Zn-ion<br>RMSE |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|
| Autoformer | 0.0305 | 0.0355 | 0.0069 | 0.0103 | 0.0139 | 0.0145 | 0.0465 | 0.0543 |
| BatLiNet | 0.0298 | 0.0357 | 0.0067 | 0.0092 | 0.0138 | 0.0134 | 0.0425 | 0.0500 |
| BiGRU | 0.0207 | 0.0260 | 0.0069 | 0.0107 | 0.0140 | 0.0155 | 0.0470 | 0.0533 |
| BiLSTM | 0.0228 | 0.0286 | 0.0062 | 0.0093 | 0.0145 | 0.0154 | 0.0459 | 0.0531 |
| CNN | 0.0234 | 0.0287 | 0.0062 | 0.0094 | 0.0132 | 0.0140 | 0.0385 | 0.0481 |
| DLinear | 0.0498 | 0.0540 | 0.0058 | 0.0089 | 0.0168 | 0.0174 | 0.0477 | 0.0542 |
| GRU | 0.0222 | 0.0276 | 0.0060 | 0.0091 | 0.0155 | 0.0163 | 0.0460 | 0.0529 |
| IC2ML | 0.0231 | 0.0288 | 0.0065 | 0.0099 | 0.0074 | 0.0086 | 0.0405 | 0.0504 |
| iTransformer | 0.0270 | 0.0328 | 0.0059 | 0.0090 | 0.0166 | 0.0173 | 0.0468 | 0.0533 |
| LSTM | 0.0236 | 0.0290 | 0.0063 | 0.0091 | 0.0176 | 0.0184 | 0.0467 | 0.0536 |
| MICN | 0.0241 | 0.0294 | 0.0065 | 0.0111 | 0.0113 | 0.0114 | 0.0413 | 0.0507 |
| MLP | 0.0350 | 0.0393 | 0.0060 | 0.0094 | 0.0166 | 0.0174 | 0.0493 | 0.0558 |
| PatchTST | 0.0349 | 0.0422 | 0.0235 | 0.0330 | 0.0147 | 0.0152 | 0.0481 | 0.0585 |
| Severson | 0.0386 | 0.0407 | 0.0477 | 0.0845 | 0.0139 | 0.0153 | 0.0455 | 0.1077 |
| TimeMixer | 0.0227 | 0.0275 | 0.0075 | 0.0131 | 0.0157 | 0.0173 | 0.0435 | 0.0529 |
| Transformer | 0.0206 | 0.0257 | 0.0060 | 0.0096 | 0.0158 | 0.0162 | 0.0368 | 0.0481 |

---

### Table 2 — Three-Level Generalization

Train on Li-ion pool; test on L1 (same chemistry, cross-dataset), L2 (Li-ion cross-cathode), L3 (cross-ion system).

#### RUL — MAPE ↓ / ACC15 ↑

| Model | L1<br>MAPE | L1<br>ACC15 | L2<br>MAPE | L2<br>ACC15 | L3<br>MAPE | L3<br>ACC15 |
|-------|-----------|------------|-----------|------------|-----------|------------|
| Autoformer | 0.4555 | 0.0396 | 0.7577 | 0.1996 | 2.3683 | 0.0265 |
| BatLiNet | 0.3815 | 0.1273 | 0.4488 | 0.1126 | 0.9671 | 0.0750 |
| BiGRU | 0.3526 | 0.1340 | 0.7650 | 0.0719 | 2.0016 | 0.0244 |
| BiLSTM | 0.3159 | 0.2440 | 0.8406 | 0.3337 | 4.9960 | 0.0532 |
| CNN | 0.4940 | 0.0042 | 0.6693 | 0.0941 | 7.2616 | 0.0069 |
| DLinear | 0.1685 | 0.4693 | 1.5674 | 0.0378 | 2.5204 | 0.0294 |
| GRU | 0.4100 | 0.0698 | 0.6557 | 0.0085 | 1.9536 | 0.0459 |
| IC2ML | 0.3449 | 0.1556 | 0.4151 | 0.2444 | 1.7078 | 0.0282 |
| iTransformer | 0.3669 | 0.1464 | 1.0200 | 0.0589 | 10.6136 | 0.0115 |
| LSTM | 0.3592 | 0.1484 | 0.6843 | 0.1363 | 3.2158 | 0.0552 |
| MICN | 0.3955 | 0.0387 | 1.1001 | 0.0070 | 3.0635 | 0.0286 |
| MLP | 0.3585 | 0.0740 | 0.7071 | 0.3393 | 2.2186 | 0.0298 |
| PatchTST | 0.2857 | 0.1858 | 1.0998 | 0.3326 | 2.0722 | 0.0281 |
| Severson | 0.2992 | 0.5531 | 1.1739 | 0.1241 | 3.1841 | 0.0565 |
| TimeMixer | 0.5382 | 0.0038 | 0.9095 | 0.1778 | 0.8921 | 0.0526 |
| Transformer | 0.3666 | 0.1502 | 1.2585 | 0.1411 | 1.6593 | 0.0306 |

#### SOH Point — MAPE ↓ / RMSE ↓

| Model | L1<br>MAPE | L1<br>RMSE | L2<br>MAPE | L2<br>RMSE | L3<br>MAPE | L3<br>RMSE |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|
| Autoformer | 0.0201 | 0.0217 | 0.0509 | 0.0488 | 0.0494 | 0.0514 |
| BatLiNet | 0.0169 | 0.0178 | 0.0260 | 0.0274 | 0.0278 | 0.0326 |
| BiGRU | 0.0202 | 0.0214 | 0.0267 | 0.0269 | 0.0605 | 0.0601 |
| BiLSTM | 0.0160 | 0.0176 | 0.0277 | 0.0283 | 0.0595 | 0.0604 |
| CNN | 0.0155 | 0.0166 | 0.0267 | 0.0266 | 0.0521 | 0.0604 |
| DLinear | 0.0190 | 0.0209 | 0.0464 | 0.0454 | 0.0496 | 0.0510 |
| GRU | 0.0207 | 0.0217 | 0.0291 | 0.0283 | 0.0663 | 0.0693 |
| IC2ML | 0.0138 | 0.0149 | 0.0094 | 0.0116 | 0.0140 | 0.0231 |
| iTransformer | 0.0114 | 0.0133 | 0.0499 | 0.0484 | 0.0473 | 0.0548 |
| LSTM | 0.0142 | 0.0177 | 0.0350 | 0.0338 | 0.0789 | 0.0824 |
| MICN | 0.0211 | 0.0246 | 0.0367 | 0.0364 | 0.0823 | 0.0836 |
| MLP | 0.0232 | 0.0249 | 0.0348 | 0.0350 | 0.0628 | 0.0705 |
| PatchTST | 0.0295 | 0.0314 | 0.0459 | 0.0457 | 0.0641 | 0.0650 |
| Severson | 0.0351 | 0.5446 | 0.0449 | 0.0455 | 0.0517 | 0.0511 |
| TimeMixer | 0.0197 | 0.0209 | 0.0452 | 0.0445 | 0.0645 | 0.0655 |
| Transformer | 0.0235 | 0.0243 | 0.0458 | 0.0446 | 0.0587 | 0.0576 |

#### SOH Trajectory — MAPE ↓ / RMSE ↓

| Model | L1<br>MAPE | L1<br>RMSE | L2<br>MAPE | L2<br>RMSE | L3<br>MAPE | L3<br>RMSE |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|
| Autoformer | 0.0445 | 0.0502 | 0.0591 | 0.0561 | 0.0708 | 0.0751 |
| BatLiNet | 0.0361 | 0.0442 | 0.0320 | 0.0378 | 0.0548 | 0.0646 |
| BiGRU | 0.0481 | 0.0566 | 0.0638 | 0.0600 | 0.0769 | 0.0777 |
| BiLSTM | 0.0547 | 0.0628 | 0.0715 | 0.0673 | 0.0883 | 0.0897 |
| CNN | 0.0482 | 0.0551 | 0.0559 | 0.0517 | 0.1686 | 0.1799 |
| DLinear | 0.0341 | 0.0361 | 0.0658 | 0.0615 | 0.0754 | 0.0740 |
| GRU | 0.0613 | 0.0662 | 0.0664 | 0.0614 | 0.0868 | 0.0905 |
| IC2ML | 0.0520 | 0.0639 | 0.0338 | 0.0393 | 0.0524 | 0.0639 |
| iTransformer | 0.0429 | 0.0511 | 0.0748 | 0.0698 | 0.0835 | 0.0936 |
| LSTM | 0.0466 | 0.0543 | 0.0741 | 0.0680 | 0.0816 | 0.0847 |
| MICN | 0.0657 | 0.0769 | 0.0607 | 0.0563 | 0.2039 | 0.2013 |
| MLP | 0.0479 | 0.0539 | 0.0444 | 0.0451 | 0.0783 | 0.0825 |
| PatchTST | 0.0752 | 0.0797 | 0.0693 | 0.0674 | 0.0839 | 0.0880 |
| Severson | 0.0328 | 0.0469 | 0.0406 | 0.0364 | 0.0455 | 0.0460 |
| TimeMixer | 0.0545 | 0.0603 | 0.0630 | 0.0592 | 0.1183 | 0.1282 |
| Transformer | 0.0502 | 0.0584 | 0.0838 | 0.0761 | 0.0750 | 0.0766 |

## Acknowledgements

BatteryBench is built on top of:

- [**BatteryML**](https://github.com/microsoft/BatteryML) (Microsoft) — battery data preprocessing pipeline and dataset abstractions
- [**BatteryLife**](https://github.com/Ruifeng-Tan/BatteryLife) — multi-chemistry battery degradation benchmark and dataset collection
