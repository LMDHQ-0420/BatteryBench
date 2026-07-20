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

**Four-level generalization** (`--domain four_level`)  
Fixed train pool; four held-out test tiers:

| Level | Generalization | Example test sets |
|-------|---------------|-------------------|
| L1 | Same chemistry, same dataset, cross-batch | HUST_batch8/9 |
| L2 | Same chemistry, cross-dataset (zero-shot charge/discharge protocol) | MATR_b4 |
| L3 | Li-ion, cross-cathode | CALCE, HNEI |
| L4 | Cross-ion system | NAion, ZNion |

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

# evaluate
python scripts/evaluate.py --domain li_ion --model gru --task rul --gpu 0
```

Parallel driver scripts (GPU-slot scheduler, same pattern in all four):

```bash
bash run_all.sh                    # everything: all domains × tasks × models
bash run_domain.sh li_ion          # one domain, all tasks × models
bash run_domain.sh li_ion 0 1      # ...restricted to GPUs 0 and 1
bash run_task.sh soh_point         # one task, all domains × models
bash run_model.sh dlinear          # one model, all domains × tasks
```

Results are saved to `results/<domain>/<task>/<model>/results.json`.  
Training logs go to `log/<timestamp>/<domain>_<task>_<model>.log`.

## Key Config Options (`configs/default.yaml`)

```yaml
data:
  task: rul              # rul | soh_point | soh_traj
  n_future: 100          # trajectory length for soh_traj
  split_strategy: random # random | stratified | four_level
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
    dataset_cls = _RULDataset,  # or _SOHPointDataset / _SOHTrajDataset, matching the task
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
│   └── domains/           li_ion / calb / na_ion / zn_ion / four_level
├── scripts/
│   ├── train.py           --domain --model --task --gpu
│   └── evaluate.py        --domain --model --task --gpu
├── src/
│   ├── data/cycle_dataset.py   RULDataset / SOHPointDataset / SOHTrajDataset (multi-sample + attn mask)
│   ├── splits.py               random / stratified / four_level split strategies
│   ├── evaluate/
│   │   ├── rul/evaluate.py
│   │   ├── soh_point/evaluate.py
│   │   └── soh_traj/evaluate.py
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
├── run_all.sh             parallel training: all domains × tasks × models
├── run_domain.sh          parallel training: one domain, all tasks × models
├── run_task.sh            parallel training: one task, all domains × models
└── run_model.sh           parallel training: one model, all domains × tasks
```

## Results

All metrics reported as mean over 3 random splits (standard domains) or weighted mean over test sets per level (four-level). MAPE and ACC15 are in [0, 1] range (not ×100).

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
| Autoformer | 0.0235 | 0.0290 | 0.0030 | 0.0072 | 0.0216 | 0.0228 | 0.0236 | 0.0384 |
| BatLiNet | 0.0225 | 0.0275 | 0.0049 | 0.0059 | 0.0112 | 0.0126 | 0.0129 | 0.0218 |
| BiGRU | 0.0101 | 0.0139 | 0.0014 | 0.0024 | 0.0255 | 0.0269 | 0.0163 | 0.0277 |
| BiLSTM | 0.0101 | 0.0142 | 0.0020 | 0.0041 | 0.0209 | 0.0230 | 0.0188 | 0.0329 |
| CNN | 0.0100 | 0.0144 | 0.0044 | 0.0058 | 0.0077 | 0.0095 | 0.0112 | 0.0190 |
| DLinear | 0.0320 | 0.0395 | 0.0025 | 0.0031 | 0.0292 | 0.0298 | 0.0253 | 0.0429 |
| GRU | 0.0086 | 0.0123 | 0.0015 | 0.0023 | 0.0232 | 0.0256 | 0.0241 | 0.0427 |
| IC2ML | 0.0102 | 0.0154 | 0.0017 | 0.0020 | 0.0039 | 0.0045 | 0.0133 | 0.0236 |
| iTransformer | 0.0189 | 0.0253 | 0.0024 | 0.0028 | 0.0279 | 0.0279 | 0.0208 | 0.0396 |
| LSTM | 0.0110 | 0.0152 | 0.0025 | 0.0086 | 0.0169 | 0.0190 | 0.0233 | 0.0451 |
| MICN | 0.0125 | 0.0180 | 0.0058 | 0.0108 | 0.0211 | 0.0248 | 0.0204 | 0.0357 |
| MLP | 0.0282 | 0.0325 | 0.0025 | 0.0031 | 0.0245 | 0.0255 | 0.0259 | 0.0420 |
| PatchTST | 0.0169 | 0.0217 | 0.0084 | 0.0110 | 0.0255 | 0.0281 | 0.0066 | 0.0131 |
| Severson | 0.0326 | 0.0379 | 0.0205 | 0.0205 | 0.0236 | 0.0264 | 0.0245 | 0.0574 |
| TimeMixer | 0.0119 | 0.0166 | 0.0026 | 0.0032 | 0.0239 | 0.0258 | 0.0215 | 0.0371 |
| Transformer | 0.0099 | 0.0143 | 0.0009 | 0.0011 | 0.0150 | 0.0164 | 0.0142 | 0.0277 |

#### SOH Trajectory — MAPE ↓ / RMSE ↓

| Model | Li-ion<br>MAPE | Li-ion<br>RMSE | CALB<br>MAPE | CALB<br>RMSE | Na-ion<br>MAPE | Na-ion<br>RMSE | Zn-ion<br>MAPE | Zn-ion<br>RMSE |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|
| Autoformer | 0.0302 | 0.0354 | 0.0071 | 0.0108 | 0.0220 | 0.0245 | 0.0459 | 0.0529 |
| BatLiNet | 0.0293 | 0.0355 | 0.0080 | 0.0124 | 0.0154 | 0.0170 | 0.0428 | 0.0519 |
| BiGRU | 0.0219 | 0.0286 | 0.0066 | 0.0110 | 0.0235 | 0.0260 | 0.0424 | 0.0514 |
| BiLSTM | 0.0213 | 0.0280 | 0.0054 | 0.0083 | 0.0236 | 0.0258 | 0.0439 | 0.0516 |
| CNN | 0.0221 | 0.0283 | 0.0079 | 0.0140 | 0.0147 | 0.0169 | 0.0365 | 0.0470 |
| DLinear | 0.0519 | 0.0568 | 0.0057 | 0.0085 | 0.0313 | 0.0316 | 0.0461 | 0.0525 |
| GRU | 0.0217 | 0.0283 | 0.0066 | 0.0111 | 0.0228 | 0.0255 | 0.0439 | 0.0515 |
| IC2ML | 0.0226 | 0.0291 | 0.0061 | 0.0088 | 0.0120 | 0.0140 | 0.0380 | 0.0470 |
| iTransformer | 0.0249 | 0.0309 | 0.0055 | 0.0084 | 0.0296 | 0.0306 | 0.0443 | 0.0519 |
| LSTM | 0.0226 | 0.0291 | 0.0051 | 0.0079 | 0.0202 | 0.0217 | 0.0442 | 0.0513 |
| MICN | 0.0245 | 0.0307 | 0.0072 | 0.0122 | 0.0228 | 0.0248 | 0.0423 | 0.0516 |
| MLP | 0.0361 | 0.0412 | 0.0062 | 0.0092 | 0.0303 | 0.0306 | 0.0464 | 0.0528 |
| PatchTST | 0.0422 | 0.0514 | 0.0239 | 0.0333 | 0.0177 | 0.0221 | 0.0473 | 0.0571 |
| Severson | 0.0407 | 0.0421 | 0.0296 | 0.0303 | 0.0251 | 0.0266 | 0.0411 | 0.0945 |
| TimeMixer | 0.0220 | 0.0279 | 0.0076 | 0.0148 | 0.0158 | 0.0207 | 0.0394 | 0.0479 |
| Transformer | 0.0213 | 0.0272 | 0.0061 | 0.0106 | 0.0161 | 0.0235 | 0.0361 | 0.0468 |

---

### Table 2 — Four-Level Generalization

Train pool: HUST + MATR + RWTH + SDU + Stanford + Tongji + ISU-ILCC + MICH + CALB + XJTU; test on L1 (same chemistry, same dataset, cross-batch), L2 (same chemistry, cross-dataset, zero-shot charge/discharge protocol), L3 (Li-ion, cross-cathode), L4 (cross-ion system).

#### RUL — MAPE ↓ / ACC15 ↑

L1 (HUST batch 8/9) is structurally empty for RUL: all 16 cells have `eol=None` (undegraded within the 100-cycle record) and are excluded by `RULDataset.REQUIRES_EOL`, so there is no data to report for that column:

| Model | L1<br>MAPE | L1<br>ACC15 | L2<br>MAPE | L2<br>ACC15 | L3<br>MAPE | L3<br>ACC15 | L4<br>MAPE | L4<br>ACC15 |
|-------|-----------|------------|-----------|------------|-----------|------------|-----------|------------|
| Autoformer | -- | -- | 0.4160 | 0.0853 | 0.7920 | 0.1878 | 2.2796 | 0.0377 |
| BatLiNet | -- | -- | 0.3462 | 0.1593 | 0.7523 | 0.0322 | 1.2175 | 0.0373 |
| BiGRU | -- | -- | 0.4021 | 0.1442 | 0.2919 | 0.4033 | 7.0625 | 0.0328 |
| BiLSTM | -- | -- | 0.3339 | 0.1776 | 1.3372 | 0.0804 | 2.6961 | 0.0401 |
| CNN | -- | -- | 0.4112 | 0.0451 | 0.3441 | 0.2515 | 16.4001 | 0.0280 |
| DLinear | -- | -- | 0.1920 | 0.5127 | 1.2398 | 0.0733 | 3.0862 | 0.0492 |
| GRU | -- | -- | 0.2120 | 0.3833 | 0.3073 | 0.4756 | 2.6966 | 0.0403 |
| IC2ML | -- | -- | 0.3576 | 0.1360 | 0.4532 | 0.3344 | 1.5016 | 0.1168 |
| iTransformer | -- | -- | 0.3001 | 0.1822 | 0.9896 | 0.1063 | 5.2083 | 0.0731 |
| LSTM | -- | -- | 0.3011 | 0.1531 | 0.4726 | 0.1519 | 2.6420 | 0.0251 |
| MICN | -- | -- | 0.4186 | 0.0129 | 0.9501 | 0.0304 | 1.7492 | 0.0373 |
| MLP | -- | -- | 0.3074 | 0.1260 | 0.6851 | 0.2904 | 2.2399 | 0.0284 |
| PatchTST | -- | -- | 0.3008 | 0.1796 | 1.5958 | 0.1256 | 1.4539 | 0.0395 |
| Severson | -- | -- | 0.2569 | 0.5356 | 1.1674 | 0.0759 | 3.1323 | 0.0548 |
| TimeMixer | -- | -- | 0.4011 | 0.0276 | 0.7065 | 0.2026 | 1.2654 | 0.0640 |
| Transformer | -- | -- | 0.3193 | 0.1762 | 1.1758 | 0.2241 | 3.4670 | 0.0546 |

#### SOH Point — MAPE ↓ / RMSE ↓

| Model | L1<br>MAPE | L1<br>RMSE | L2<br>MAPE | L2<br>RMSE | L3<br>MAPE | L3<br>RMSE | L4<br>MAPE | L4<br>RMSE |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Autoformer | 0.0120 | 0.0148 | 0.0200 | 0.0217 | 0.0423 | 0.0423 | 0.0573 | 0.0657 |
| BatLiNet | 0.0097 | 0.0109 | 0.0148 | 0.0160 | 0.0248 | 0.0263 | 0.0301 | 0.0346 |
| BiGRU | 0.0021 | 0.0034 | 0.0178 | 0.0184 | 0.0337 | 0.0337 | 0.0549 | 0.0576 |
| BiLSTM | 0.0025 | 0.0041 | 0.0050 | 0.0063 | 0.0333 | 0.0325 | 0.0622 | 0.0653 |
| CNN | 0.0036 | 0.0051 | 0.0055 | 0.0076 | 0.0321 | 0.0317 | 0.0570 | 0.0589 |
| DLinear | 0.0273 | 0.0275 | 0.0143 | 0.0159 | 0.0456 | 0.0447 | 0.0565 | 0.0582 |
| GRU | 0.0036 | 0.0054 | 0.0268 | 0.0273 | 0.0331 | 0.0320 | 0.0561 | 0.0626 |
| IC2ML | 0.0015 | 0.0018 | 0.0172 | 0.0176 | 0.0104 | 0.0128 | 0.0175 | 0.0274 |
| iTransformer | 0.0229 | 0.0230 | 0.0189 | 0.0197 | 0.0438 | 0.0433 | 0.0567 | 0.0590 |
| LSTM | 0.0036 | 0.0059 | 0.0079 | 0.0095 | 0.0277 | 0.0273 | 0.0627 | 0.0648 |
| MICN | 0.0049 | 0.0064 | 0.0167 | 0.0188 | 0.0399 | 0.0398 | 0.1143 | 0.1171 |
| MLP | 0.0105 | 0.0127 | 0.0311 | 0.0317 | 0.0425 | 0.0427 | 0.1171 | 0.1232 |
| PatchTST | 0.0051 | 0.0070 | 0.0139 | 0.0153 | 0.0547 | 0.0541 | 0.0490 | 0.0499 |
| Severson | 0.0289 | 0.0297 | 0.0107 | 0.0369 | 0.0461 | 0.0459 | 0.0560 | 0.0568 |
| TimeMixer | 0.0036 | 0.0052 | 0.0193 | 0.0210 | 0.0350 | 0.0343 | 0.0583 | 0.0586 |
| Transformer | 0.0019 | 0.0033 | 0.0146 | 0.0158 | 0.0451 | 0.0447 | 0.0564 | 0.0617 |

#### SOH Trajectory — MAPE ↓ / RMSE ↓

| Model | L1<br>MAPE | L1<br>RMSE | L2<br>MAPE | L2<br>RMSE | L3<br>MAPE | L3<br>RMSE | L4<br>MAPE | L4<br>RMSE |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|-----------|
| Autoformer | 0.0397 | 0.0442 | 0.0243 | 0.0299 | 0.0548 | 0.0540 | 1.2405 | 0.0721 |
| BatLiNet | 0.0294 | 0.0358 | 0.0215 | 0.0286 | 0.0434 | 0.0499 | 1.2121 | 0.0666 |
| BiGRU | 0.0200 | 0.0290 | 0.0273 | 0.0341 | 0.0535 | 0.0532 | 1.1588 | 0.0856 |
| BiLSTM | 0.0203 | 0.0302 | 0.0254 | 0.0298 | 0.0644 | 0.0601 | 1.2129 | 0.1107 |
| CNN | 0.0235 | 0.0301 | 0.0215 | 0.0285 | 0.0639 | 0.0588 | 1.2758 | 0.1099 |
| DLinear | 0.0626 | 0.0654 | 0.0248 | 0.0310 | 0.0800 | 0.0745 | 1.2382 | 0.0726 |
| GRU | 0.0182 | 0.0270 | 0.0263 | 0.0309 | 0.0564 | 0.0546 | 1.2113 | 0.0817 |
| IC2ML | 0.0219 | 0.0311 | 0.0209 | 0.0266 | 0.0244 | 0.0273 | 1.1816 | 0.0614 |
| iTransformer | 0.0206 | 0.0293 | 0.0214 | 0.0286 | 0.0652 | 0.0615 | 1.2932 | 0.0821 |
| LSTM | 0.0207 | 0.0272 | 0.0205 | 0.0248 | 0.0337 | 0.0371 | 1.2119 | 0.0791 |
| MICN | 0.0207 | 0.0296 | 0.0268 | 0.0322 | 0.0711 | 0.0649 | 1.3958 | 0.3023 |
| MLP | 0.0340 | 0.0397 | 0.0322 | 0.0374 | 0.0400 | 0.0405 | 1.2113 | 0.0975 |
| PatchTST | 0.0505 | 0.0634 | 0.0463 | 0.0538 | 0.0724 | 0.0714 | 1.1760 | 0.0933 |
| Severson | 0.0496 | 0.0491 | 0.0255 | 0.0550 | 0.0542 | 0.0492 | 6.3073 | 0.0503 |
| TimeMixer | 0.0179 | 0.0261 | 0.0227 | 0.0296 | 0.0670 | 0.0624 | 1.2472 | 0.1103 |
| Transformer | 0.0254 | 0.0351 | 0.0214 | 0.0278 | 0.0915 | 0.0822 | 1.3134 | 0.0988 |

## Acknowledgements

BatteryBench is built on top of:

- [**BatteryML**](https://github.com/microsoft/BatteryML) (Microsoft) — battery data preprocessing pipeline and dataset abstractions
- [**BatteryLife**](https://github.com/Ruifeng-Tan/BatteryLife) — multi-chemistry battery degradation benchmark and dataset collection
