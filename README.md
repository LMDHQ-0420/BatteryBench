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
| **Battery-specific** | IC²ML, BatLiNet, BatteryMFormer, Severson ElasticNet |

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

# train all models on all domains (parallel, 4 GPUs × 2 slots)
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
│   │   │   ├── baseline/  17 models (mlp gru bigru lstm bilstm cnn dlinear patchtst
│   │   │   │              transformer autoformer itransformer micn timemixer ic2ml
│   │   │   │              batlinet batterymformer severson)
│   │   │   └── mymodel.py ← your custom model goes here
│   │   ├── soh_point/
│   │   │   ├── baseline/  17 models (same as rul)
│   │   │   └── mymodel.py
│   │   └── soh_traj/
│   │       ├── baseline/  17 models (same as rul)
│   │       └── mymodel.py
│   └── train/
│       ├── rul/           train_base  train_batlinet  train_severson
│       ├── soh_point/     train_base  train_batlinet  train_severson
│       └── soh_traj/      train_base  train_batlinet  train_severson
├── run_all.sh             parallel training across all models / domains / tasks
└── STRUCTURE.md           detailed architecture and extension guide
```

## Results

All metrics reported as mean over 3 random splits. MAPE and ACC15 are in [0, 1] range (not ×100).

### Table 1 — Standard Domains (Li-ion / CALB / Na-ion / Zn-ion)

#### RUL — MAPE ↓ / ACC15 ↑

| Model | Li-ion<br>MAPE | Li-ion<br>ACC15 | CALB<br>MAPE | CALB<br>ACC15 | Na-ion<br>MAPE | Na-ion<br>ACC15 | Zn-ion<br>MAPE | Zn-ion<br>ACC15 |
|---|---|---|---|---|---|---|---|---|
| Autoformer | 0.8961 | 0.2741 | 0.1476 | 0.6667 | 2.6887 | 0.1212 | 5.6243 | 0.0556 |
| BatLiNet | 1.0710 | 0.1711 | 0.5708 | 0.0000 | 0.8796 | 0.0303 | 0.9228 | 0.0000 |
| BatteryMFormer | 0.6372 | 0.3377 | 0.0052 | 1.0000 | 1.1239 | 0.3333 | 5.7124 | 0.0417 |
| BiGRU | 0.8115 | 0.3487 | 0.0064 | 1.0000 | 2.6431 | 0.1212 | 5.2840 | 0.0694 |
| BiLSTM | 0.7866 | 0.3333 | 0.0032 | 1.0000 | 2.6965 | 0.1212 | 5.3467 | 0.0694 |
| CNN | 0.8671 | 0.3202 | 0.0697 | 1.0000 | 3.2016 | 0.1212 | 5.8845 | 0.0556 |
| DLinear | 1.2851 | 0.0877 | 0.0187 | 1.0000 | 1.0713 | 0.0606 | 0.9533 | 0.0278 |
| GRU | 0.7740 | 0.3487 | 0.0059 | 1.0000 | 2.6370 | 0.0909 | 5.2997 | 0.0694 |
| IC2ML | 0.6646 | 0.3991 | 0.0063 | 1.0000 | 0.8993 | 0.2121 | 5.4421 | 0.0417 |
| iTransformer | 0.7661 | 0.2763 | 0.0071 | 1.0000 | 1.7832 | 0.1515 | 5.7777 | 0.0556 |
| LSTM | 0.7915 | 0.3553 | 0.0064 | 1.0000 | 2.6569 | 0.0909 | 5.3698 | 0.0694 |
| MICN | 1.0090 | 0.1886 | 0.1376 | 0.5333 | 3.0071 | 0.3030 | 5.7629 | 0.0417 |
| MLP | 0.9651 | 0.2061 | 0.0733 | 0.9333 | 3.1454 | 0.1515 | 5.4311 | 0.0833 |
| PatchTST | 0.8766 | 0.2478 | 0.0048 | 1.0000 | 2.7760 | 0.0909 | 5.9267 | 0.0417 |
| Severson | 1.8186 | 0.1272 | 0.0451 | 1.0000 | 1.6248 | 0.1515 | 6.1822 | 0.0417 |
| TimeMixer | 1.1463 | 0.1096 | 0.0128 | 1.0000 | 2.7616 | 0.0606 | 5.3243 | 0.0694 |
| Transformer | 0.9667 | 0.2763 | 0.0380 | 1.0000 | 2.6496 | 0.1212 | 5.8140 | 0.0556 |

#### SOH Point — MAPE ↓ / RMSE ↓

| Model | Li-ion<br>MAPE | Li-ion<br>RMSE | CALB<br>MAPE | CALB<br>RMSE | Na-ion<br>MAPE | Na-ion<br>RMSE | Zn-ion<br>MAPE | Zn-ion<br>RMSE |
|---|---|---|---|---|---|---|---|---|
| Autoformer | 0.0326 | 0.0403 | 0.0019 | 0.0019 | 0.0421 | 0.0407 | 0.0464 | 0.0755 |
| BatLiNet | 0.3895 | 0.4028 | 0.0798 | 0.0885 | 0.3144 | 0.2812 | 0.5950 | 0.5785 |
| BatteryMFormer | 0.0192 | 0.0301 | 0.0143 | 0.0182 | 0.0456 | 0.0439 | 0.1185 | 0.1439 |
| BiGRU | 0.0212 | 0.0289 | 0.0507 | 0.0572 | 0.0414 | 0.0400 | 0.0462 | 0.0755 |
| BiLSTM | 0.0216 | 0.0283 | 0.0505 | 0.0564 | 0.0425 | 0.0413 | 0.0471 | 0.0744 |
| CNN | 0.0202 | 0.0274 | 0.0252 | 0.0299 | 0.0176 | 0.0167 | 0.0465 | 0.0748 |
| DLinear | 0.0312 | 0.0401 | 0.0575 | 0.0619 | 0.0114 | 0.0112 | 0.1567 | 0.1671 |
| GRU | 0.0213 | 0.0282 | 0.0558 | 0.0701 | 0.0394 | 0.0384 | 0.0460 | 0.0755 |
| IC2ML | 0.0248 | 0.0319 | 0.0138 | 0.0160 | 0.0061 | 0.0061 | 0.0472 | 0.0741 |
| iTransformer | 0.0268 | 0.0337 | 0.0476 | 0.0566 | 0.0449 | 0.0430 | 0.0474 | 0.0736 |
| LSTM | 0.0256 | 0.0325 | 0.0157 | 0.0222 | 0.0421 | 0.0406 | 0.0462 | 0.0754 |
| MICN | 0.0309 | 0.0387 | 0.0554 | 0.0570 | 0.1047 | 0.1120 | 0.0797 | 0.0906 |
| MLP | 0.0391 | 0.0441 | 0.0726 | 0.0853 | 0.0130 | 0.0123 | 0.0463 | 0.0679 |
| PatchTST | 0.0351 | 0.0435 | 0.0036 | 0.0036 | 0.0435 | 0.0422 | 0.0460 | 0.0758 |
| Severson | 0.0380 | 0.0420 | 0.1790 | 0.3278 | 0.0167 | 0.0165 | 0.0222 | 0.0677 |
| TimeMixer | 0.0292 | 0.0349 | 0.0471 | 0.0510 | 0.0180 | 0.0171 | 0.0468 | 0.0740 |
| Transformer | 0.0313 | 0.0387 | 0.0050 | 0.0050 | 0.0429 | 0.0416 | 0.0462 | 0.0754 |

#### SOH Trajectory — MAPE ↓ / RMSE ↓

| Model | Li-ion<br>MAPE | Li-ion<br>RMSE | CALB<br>MAPE | CALB<br>RMSE | Na-ion<br>MAPE | Na-ion<br>RMSE | Zn-ion<br>MAPE | Zn-ion<br>RMSE |
|---|---|---|---|---|---|---|---|---|
| Autoformer | 0.5179 | 0.1308 | 0.0244 | 0.0282 | 0.0470 | 0.0508 | 0.1259 | 0.1287 |
| BatLiNet | 0.8419 | 0.4819 | 0.0611 | 0.0696 | 0.2949 | 0.2684 | 0.9738 | 0.8523 |
| BatteryMFormer | 0.0198 | 0.0247 | 0.0074 | 0.0094 | 0.0328 | 0.0337 | 0.0184 | 0.0320 |
| BiGRU | 0.4881 | 0.0923 | 0.0394 | 0.0496 | 0.0514 | 0.0565 | 0.1248 | 0.1286 |
| BiLSTM | 0.4469 | 0.0928 | 0.0537 | 0.0628 | 0.0424 | 0.0465 | 0.1271 | 0.1285 |
| CNN | 0.5044 | 0.0950 | 0.0224 | 0.0276 | 0.0210 | 0.0290 | 0.1268 | 0.1292 |
| DLinear | 0.6795 | 0.1668 | 0.0575 | 0.0752 | 0.0162 | 0.0255 | 0.2905 | 0.2819 |
| GRU | 0.4938 | 0.0932 | 0.0600 | 0.0685 | 0.0463 | 0.0508 | 0.1261 | 0.1284 |
| IC2ML | 0.4877 | 0.0966 | 0.0211 | 0.0257 | 0.0245 | 0.0312 | 0.1299 | 0.1316 |
| iTransformer | 0.4867 | 0.1050 | 0.0456 | 0.0545 | 0.0469 | 0.0503 | 0.1254 | 0.1282 |
| LSTM | 0.4981 | 0.0924 | 0.0479 | 0.0592 | 0.0438 | 0.0472 | 0.1263 | 0.1283 |
| MICN | 0.5780 | 0.1468 | 0.1707 | 0.2133 | 0.1791 | 0.1820 | 0.3652 | 0.3665 |
| MLP | 0.6777 | 0.1424 | 0.9108 | 1.0524 | 0.0552 | 0.0621 | 0.1233 | 0.1245 |
| PatchTST | 0.5065 | 0.1248 | 0.0367 | 0.0433 | 0.0463 | 0.0517 | 0.1296 | 0.1300 |
| Severson | 0.7583 | 0.1313 | 0.1762 | 0.0068 | 0.2025 | 0.0074 | 0.9403 | 0.2192 |
| TimeMixer | 0.5250 | 0.1419 | 0.0350 | 0.0439 | 0.0259 | 0.0328 | 0.1276 | 0.1296 |
| Transformer | 0.5141 | 0.1290 | 0.0221 | 0.0265 | 0.0455 | 0.0497 | 0.1283 | 0.1294 |

---

### Table 2 — Three-Level Generalization

Train on Li-ion pool; test on L1 (same chemistry, cross-dataset), L2 (Li-ion cross-cathode), L3 (cross-ion system).

#### RUL — MAPE ↓ / ACC15 ↑

| Model | L1<br>MAPE | L1<br>ACC15 | L2<br>MAPE | L2<br>ACC15 | L3<br>MAPE | L3<br>ACC15 |
|---|---|---|---|---|---|---|
| Autoformer | 0.1996 | 0.4590 | 1.4672 | 0.0000 | 12.3778 | 0.0347 |
| BatLiNet | 0.3340 | 0.2295 | 0.6418 | 0.1481 | 3.4644 | 0.0694 |
| BatteryMFormer | 0.2135 | 0.4262 | 0.7782 | 0.0370 | 23.1621 | 0.0231 |
| BiGRU | 0.3545 | 0.1967 | 2.4482 | 0.0000 | 16.4815 | 0.0000 |
| BiLSTM | 0.2087 | 0.4426 | 1.3882 | 0.0000 | 5.8106 | 0.0347 |
| CNN | 0.2212 | 0.4754 | 1.7379 | 0.0000 | 11.1235 | 0.0636 |
| DLinear | 0.7013 | 0.0000 | 1.5244 | 0.0000 | 1.9535 | 0.0058 |
| GRU | 0.1992 | 0.4098 | 2.1668 | 0.0000 | 15.8212 | 0.0000 |
| IC2ML | 0.4671 | 0.1639 | 0.6864 | 0.2593 | 27.5620 | 0.0405 |
| iTransformer | 0.6624 | 0.1639 | 1.7208 | 0.2963 | 16.3647 | 0.0000 |
| LSTM | 0.1921 | 0.4918 | 1.8938 | 0.0000 | 14.6309 | 0.0173 |
| MICN | 0.3656 | 0.2787 | 1.1881 | 0.0000 | 13.5352 | 0.0462 |
| MLP | 0.2083 | 0.4754 | 0.5823 | 0.2222 | 1.7501 | 0.0347 |
| PatchTST | 0.2075 | 0.4426 | 1.2916 | 0.0000 | 13.0546 | 0.0347 |
| Severson | 0.3433 | 0.2623 | 1.8568 | 0.0370 | 18.5970 | 0.0520 |
| TimeMixer | 0.3413 | 0.2951 | 1.1365 | 0.1111 | 9.6773 | 0.0347 |
| Transformer | 0.2175 | 0.4426 | 1.4930 | 0.0000 | 12.6611 | 0.0347 |

#### SOH Point — MAPE ↓ / RMSE ↓

| Model | L1<br>MAPE | L1<br>RMSE | L2<br>MAPE | L2<br>RMSE | L3<br>MAPE | L3<br>RMSE |
|---|---|---|---|---|---|---|
| Autoformer | 0.0293 | 0.0303 | 0.0617 | 0.0605 | 0.0791 | 0.0815 |
| BatLiNet | 0.1919 | 0.2030 | 0.2758 | 0.2663 | 0.6093 | 0.5782 |
| BatteryMFormer | 0.0148 | 0.0157 | 0.0448 | 0.0426 | 0.0593 | 0.0607 |
| BiGRU | 0.0256 | 0.0271 | 0.0565 | 0.0540 | 0.3677 | 0.3562 |
| BiLSTM | 0.0233 | 0.0247 | 0.0579 | 0.0555 | 0.3414 | 0.3303 |
| CNN | 0.0127 | 0.0136 | 0.0608 | 0.0591 | 0.1174 | 0.1200 |
| DLinear | 0.0714 | 0.1346 | 0.0981 | 0.1921 | 0.0749 | 0.0729 |
| GRU | 0.0177 | 0.0187 | 0.0532 | 0.0505 | 0.4366 | 0.4219 |
| IC2ML | 0.0135 | 0.0144 | 0.0502 | 0.0483 | 0.0827 | 0.0805 |
| iTransformer | 0.0162 | 0.0171 | 0.0634 | 0.0598 | 0.5234 | 0.5037 |
| LSTM | 0.0159 | 0.0170 | 0.0680 | 0.0635 | 0.4129 | 0.3983 |
| MICN | 0.0136 | 0.0147 | 0.0371 | 0.0354 | 0.0635 | 0.0764 |
| MLP | 0.0202 | 0.0220 | 0.0437 | 0.0413 | 0.0764 | 0.0741 |
| PatchTST | 0.0214 | 0.0219 | 0.0389 | 0.0373 | 0.0785 | 0.0802 |
| Severson | 0.0145 | 0.0154 | 0.0511 | 0.0516 | 0.0727 | 0.0723 |
| TimeMixer | 0.0234 | 0.0271 | 0.0481 | 0.0507 | 0.1082 | 0.1062 |
| Transformer | 0.0176 | 0.0186 | 0.0413 | 0.0390 | 0.0784 | 0.0794 |

#### SOH Trajectory — MAPE ↓ / RMSE ↓

| Model | L1<br>MAPE | L1<br>RMSE | L2<br>MAPE | L2<br>RMSE | L3<br>MAPE | L3<br>RMSE |
|---|---|---|---|---|---|---|
| Autoformer | 0.0641 | 0.0744 | 0.3627 | 0.2352 | 0.2357 | 0.2281 |
| BatLiNet | 0.4718 | 0.4516 | 0.3631 | 0.2988 | 0.6692 | 0.6000 |
| BatteryMFormer | 0.0122 | 0.0140 | 0.0563 | 0.0560 | 0.0497 | 0.0577 |
| BiGRU | 0.0765 | 0.0964 | 0.3708 | 0.2452 | 0.3614 | 0.3521 |
| BiLSTM | 0.0686 | 0.0878 | 0.3666 | 0.2350 | 0.4154 | 0.3902 |
| CNN | 0.0718 | 0.0927 | 0.3200 | 0.2124 | 0.2201 | 0.1979 |
| DLinear | 0.1446 | 0.1437 | 0.3800 | 0.2517 | 0.2100 | 0.2094 |
| GRU | 0.0631 | 0.0823 | 0.3457 | 0.2260 | 0.4467 | 0.4119 |
| IC2ML | 0.0684 | 0.0852 | 0.3925 | 0.2530 | 0.2149 | 0.1954 |
| iTransformer | 0.0747 | 0.0930 | 0.3242 | 0.2139 | 0.3033 | 0.3015 |
| LSTM | 0.0709 | 0.0907 | 0.3540 | 0.2282 | 0.3162 | 0.3066 |
| MICN | 0.0935 | 0.1094 | 0.3235 | 0.2102 | 0.2269 | 0.2194 |
| MLP | 0.0768 | 0.0960 | 0.2041 | 0.1395 | 0.2087 | 0.1980 |
| PatchTST | 0.0726 | 0.0825 | 0.3515 | 0.2259 | 0.2197 | 0.2132 |
| Severson | 0.2878 | 0.0833 | 0.1984 | 0.0327 | 2.0502 | 0.1028 |
| TimeMixer | 0.1074 | 0.1232 | 0.3083 | 0.2043 | 0.3223 | 0.3126 |
| Transformer | 0.0688 | 0.0763 | 0.3429 | 0.2245 | 0.2368 | 0.2315 |

## Acknowledgements

BatteryBench is built on top of:

- [**BatteryML**](https://github.com/microsoft/BatteryML) (Microsoft) — battery data preprocessing pipeline and dataset abstractions
- [**BatteryLife**](https://github.com/Ruifeng-Tan/BatteryLife) — multi-chemistry battery degradation benchmark and dataset collection
