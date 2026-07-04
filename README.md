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
|-------|--------|--------|--------|--------|--------|--------|--------|--------|
| Autoformer | 0.5920 | 0.4518 | 0.0056 | 1.0000 | 2.7088 | 0.2424 | 11.9178 | 0.0694 |
| BatLiNet | 1.5365 | 0.2281 | 0.3323 | 0.3333 | 1.5807 | 0.0000 | 1.6068 | 0.0278 |
| BatteryMFormer | 0.6889 | 0.3640 | 0.0165 | 1.0000 | 1.1549 | 0.4848 | 8.9556 | 0.0556 |
| BiGRU | 0.4627 | 0.4561 | 0.0065 | 1.0000 | 1.9283 | 0.5152 | 11.1803 | 0.0278 |
| BiLSTM | 0.4696 | 0.4189 | 0.0075 | 1.0000 | 1.2307 | 0.5758 | 11.8281 | 0.0694 |
| CNN | 0.5463 | 0.3816 | 0.0103 | 1.0000 | 1.4357 | 0.6667 | 2.1176 | 0.1528 |
| DLinear | 2.6890 | 0.0636 | 0.0254 | 0.9333 | 3.6492 | 0.1515 | 53.2764 | 0.0417 |
| GRU | 0.5073 | 0.3882 | 0.0087 | 1.0000 | 0.9961 | 0.5758 | 13.3618 | 0.0556 |
| IC2ML | 0.6063 | 0.3158 | 0.0082 | 1.0000 | 1.0757 | 0.4242 | 3.7143 | 0.1111 |
| iTransformer | 0.6293 | 0.3289 | 0.0045 | 1.0000 | 2.7177 | 0.1818 | 11.6882 | 0.0833 |
| LSTM | 0.5332 | 0.4211 | 0.0072 | 1.0000 | 1.1491 | 0.3333 | 13.3223 | 0.0694 |
| MICN | 0.3056 | 0.4846 | 0.0073 | 1.0000 | 1.6015 | 0.4848 | 3.4106 | 0.0556 |
| MLP | 0.7795 | 0.3531 | 0.0273 | 1.0000 | 3.5601 | 0.3030 | 8.2656 | 0.0417 |
| PatchTST | 0.6262 | 0.3531 | 0.0060 | 1.0000 | 1.6699 | 0.3030 | 9.2145 | 0.0556 |
| Severson | 1.9590 | 0.1491 | 0.0451 | 1.0000 | 1.6248 | 0.1515 | 4.2022 | 0.0694 |
| TimeMixer | 0.5119 | 0.3904 | 0.0077 | 1.0000 | 1.9814 | 0.5758 | 10.6352 | 0.0278 |
| Transformer | 0.5710 | 0.3706 | 0.0033 | 1.0000 | 1.0278 | 0.5455 | 6.0752 | 0.1389 |

#### SOH Point — MAPE ↓ / RMSE ↓

| Model | Li-ion<br>MAPE | Li-ion<br>RMSE | CALB<br>MAPE | CALB<br>RMSE | Na-ion<br>MAPE | Na-ion<br>RMSE | Zn-ion<br>MAPE | Zn-ion<br>RMSE |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|
| Autoformer | 0.0099 | 0.0135 | 0.0066 | 0.0070 | 0.0448 | 0.0435 | 0.0462 | 0.0736 |
| BatLiNet | 0.1930 | 0.2091 | 0.0597 | 0.0693 | 0.2536 | 0.2254 | 0.2656 | 0.2779 |
| BatteryMFormer | 0.0214 | 0.0338 | 0.0147 | 0.0184 | 0.0440 | 0.0424 | 0.1185 | 0.1438 |
| BiGRU | 0.0134 | 0.0182 | 0.0046 | 0.0047 | 0.0424 | 0.0411 | 0.0477 | 0.0715 |
| BiLSTM | 0.0112 | 0.0161 | 0.0030 | 0.0030 | 0.0426 | 0.0412 | 0.0467 | 0.0738 |
| CNN | 0.0132 | 0.0172 | 0.0192 | 0.0241 | 0.0319 | 0.0336 | 0.0337 | 0.0396 |
| DLinear | 1.3529 | 1.5237 | 0.2647 | 0.2697 | 1.0362 | 1.0342 | 1.0759 | 1.5648 |
| GRU | 0.0123 | 0.0167 | 0.0025 | 0.0026 | 0.0423 | 0.0405 | 0.0462 | 0.0746 |
| IC2ML | 0.0172 | 0.0223 | 0.0124 | 0.0137 | 0.0061 | 0.0065 | 0.0311 | 0.0403 |
| iTransformer | 0.0286 | 0.0361 | 0.0038 | 0.0050 | 0.0438 | 0.0424 | 0.0463 | 0.0740 |
| LSTM | 0.0118 | 0.0159 | 0.0013 | 0.0013 | 0.0435 | 0.0421 | 0.0468 | 0.0740 |
| MICN | 0.0122 | 0.0161 | 0.0976 | 0.1392 | 0.1704 | 0.1853 | 0.1516 | 0.1847 |
| MLP | 0.0309 | 0.0360 | 0.0706 | 0.0771 | 0.1653 | 0.1702 | 0.0998 | 0.1133 |
| PatchTST | 0.0174 | 0.0229 | 0.0031 | 0.0035 | 0.0448 | 0.0432 | 0.0463 | 0.0757 |
| Severson | 4.7480 | 0.3520 | 0.0718 | 0.0737 | 0.0300 | 0.0407 | 0.0276 | 0.0301 |
| TimeMixer | 0.0250 | 0.0295 | 0.0267 | 0.0346 | 0.0430 | 0.0510 | 0.0467 | 0.0513 |
| Transformer | 0.0100 | 0.0133 | 0.0036 | 0.0043 | 0.0436 | 0.0421 | 0.0470 | 0.0716 |

#### SOH Trajectory — MAPE ↓ / RMSE ↓

| Model | Li-ion<br>MAPE | Li-ion<br>RMSE | CALB<br>MAPE | CALB<br>RMSE | Na-ion<br>MAPE | Na-ion<br>RMSE | Zn-ion<br>MAPE | Zn-ion<br>RMSE |
|-------|--------|--------|--------|--------|--------|--------|--------|--------|
| Autoformer | 0.5012 | 0.0865 | 0.0111 | 0.0140 | 0.0458 | 0.0505 | 0.1263 | 0.1281 |
| BatLiNet | 0.7167 | 0.5426 | 0.1221 | 0.1289 | 0.2500 | 0.2288 | 0.7695 | 0.6916 |
| BatteryMFormer | 0.0178 | 0.0243 | 0.0157 | 0.0184 | 0.0344 | 0.0362 | 0.0190 | 0.0326 |
| BiGRU | 0.4569 | 0.0822 | 0.0110 | 0.0140 | 0.0473 | 0.0524 | 0.1195 | 0.1241 |
| BiLSTM | 0.4882 | 0.0825 | 0.0083 | 0.0103 | 0.0478 | 0.0521 | 0.1262 | 0.1282 |
| CNN | 0.4888 | 0.0854 | 0.0152 | 0.0199 | 0.0334 | 0.0394 | 0.0839 | 0.0908 |
| DLinear | 1.0589 | 0.2425 | 1.5726 | 2.0041 | 0.0317 | 0.0367 | 0.1226 | 0.1217 |
| GRU | 0.4661 | 0.0787 | 0.0106 | 0.0131 | 0.0500 | 0.0545 | 0.1267 | 0.1285 |
| IC2ML | 0.6410 | 0.1378 | 0.0247 | 0.0303 | 0.0279 | 0.0348 | 0.0978 | 0.1141 |
| iTransformer | 0.5390 | 0.0993 | 0.0103 | 0.0129 | 0.0504 | 0.0546 | 0.1266 | 0.1286 |
| LSTM | 0.5223 | 0.0865 | 0.0088 | 0.0117 | 0.0471 | 0.0503 | 0.1274 | 0.1294 |
| MICN | 0.5448 | 0.0918 | 0.0910 | 0.1250 | 0.6772 | 0.7165 | 0.1373 | 0.1387 |
| MLP | 0.7038 | 0.1287 | 0.0495 | 0.0626 | 0.0728 | 0.0933 | 0.1529 | 0.1491 |
| PatchTST | 0.5884 | 0.0932 | 0.0120 | 0.0154 | 0.0481 | 0.0525 | 0.1275 | 0.1295 |
| Severson | 0.7150 | 0.1147 | 0.1762 | 0.0068 | 0.2025 | 0.0074 | 0.6919 | 0.1387 |
| TimeMixer | 0.4444 | 0.0903 | 0.0398 | 0.0498 | 0.0474 | 0.0537 | 0.0894 | 0.0967 |
| Transformer | 0.5481 | 0.0935 | 0.0105 | 0.0135 | 0.0473 | 0.0515 | 0.1242 | 0.1268 |

---

### Table 2 — Three-Level Generalization

Train on Li-ion pool; test on L1 (same chemistry, cross-dataset), L2 (Li-ion cross-cathode), L3 (cross-ion system).

#### RUL — MAPE ↓ / ACC15 ↑

| Model | L1<br>MAPE | L1<br>ACC15 | L2<br>MAPE | L2<br>ACC15 | L3<br>MAPE | L3<br>ACC15 |
|-------|-----------|------------|-----------|------------|-----------|------------|
| Autoformer | 0.1781 | 0.4918 | 0.4735 | 0.1481 | 17.1633 | 0.0116 |
| BatLiNet | 0.2304 | 0.4262 | 1.2637 | 0.2593 | 9.9198 | 0.0347 |
| BatteryMFormer | 0.1788 | 0.6230 | 1.5932 | 0.0370 | 21.3272 | 0.0809 |
| BiGRU | 0.1915 | 0.5082 | 0.2984 | 0.1481 | 65.1448 | 0.0000 |
| BiLSTM | 0.1801 | 0.5574 | 0.1930 | 0.5556 | 73.8363 | 0.0000 |
| CNN | 0.1627 | 0.5902 | 1.1119 | 0.1852 | 18.7228 | 0.0636 |
| DLinear | 2.4162 | 0.1148 | 4.0183 | 0.0370 | 246.8123 | 0.0000 |
| GRU | 0.2266 | 0.5574 | 0.2932 | 0.1111 | 69.1897 | 0.0000 |
| IC2ML | 0.2210 | 0.5082 | 0.6424 | 0.0741 | 9.7440 | 0.0462 |
| iTransformer | 0.1740 | 0.6557 | 1.0640 | 0.0370 | 10.2724 | 0.0809 |
| LSTM | 0.1636 | 0.5902 | 0.2642 | 0.2222 | 80.6114 | 0.0000 |
| MICN | 0.1771 | 0.4590 | 0.2758 | 0.2963 | 148.5721 | 0.0000 |
| MLP | 0.2444 | 0.4098 | 0.5240 | 0.1111 | 83.9738 | 0.0000 |
| PatchTST | 0.2302 | 0.3279 | 0.3175 | 0.0741 | 68.4689 | 0.0000 |
| Severson | 0.3173 | 0.1311 | 2.3168 | 0.0741 | 16.1778 | 0.0116 |
| TimeMixer | 0.2133 | 0.3934 | 0.5705 | 0.0741 | 90.3321 | 0.0000 |
| Transformer | 0.2082 | 0.4098 | 0.2556 | 0.4444 | 56.8442 | 0.0000 |

#### SOH Point — MAPE ↓ / RMSE ↓

| Model | L1<br>MAPE | L1<br>RMSE | L2<br>MAPE | L2<br>RMSE | L3<br>MAPE | L3<br>RMSE |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|
| Autoformer | 0.0075 | 0.0090 | 0.0663 | 0.0634 | 0.0738 | 0.0781 |
| BatLiNet | 0.3890 | 0.3819 | 0.3914 | 0.3615 | 0.4719 | 0.4644 |
| BatteryMFormer | 0.0114 | 0.0140 | 0.0912 | 0.0854 | 0.1317 | 0.1466 |
| BiGRU | 0.0110 | 0.0136 | 0.0552 | 0.0527 | 0.0498 | 0.0517 |
| BiLSTM | 0.0160 | 0.0173 | 0.0309 | 0.0305 | 0.0787 | 0.0929 |
| CNN | 0.0049 | 0.0058 | 0.0485 | 0.0468 | 0.0537 | 0.0582 |
| DLinear | 1.8530 | 1.9689 | 2.3117 | 2.0967 | 13.0439 | 12.4847 |
| GRU | 0.0125 | 0.0145 | 0.0312 | 0.0317 | 0.0707 | 0.0744 |
| IC2ML | 0.0123 | 0.0128 | 0.0131 | 0.0186 | 0.0355 | 0.0410 |
| iTransformer | 0.0173 | 0.0200 | 0.0195 | 0.0206 | 0.0770 | 0.0837 |
| LSTM | 0.0135 | 0.0163 | 0.0315 | 0.0324 | 0.0735 | 0.0847 |
| MICN | 0.0178 | 0.0192 | 0.0422 | 0.0412 | 0.2490 | 0.2438 |
| MLP | 0.0460 | 0.0463 | 0.0177 | 0.0179 | 0.5618 | 0.5431 |
| PatchTST | 0.0140 | 0.0163 | 0.0355 | 0.0343 | 0.0717 | 0.0699 |
| Severson | N/A | N/A | N/A | N/A | N/A | N/A |
| TimeMixer | 0.0228 | 0.0249 | 0.0513 | 0.0488 | 0.4412 | 0.5436 |
| Transformer | 0.0096 | 0.0114 | 0.0631 | 0.0607 | 0.0760 | 0.0808 |

#### SOH Trajectory — MAPE ↓ / RMSE ↓

| Model | L1<br>MAPE | L1<br>RMSE | L2<br>MAPE | L2<br>RMSE | L3<br>MAPE | L3<br>RMSE |
|-------|-----------|-----------|-----------|-----------|-----------|-----------|
| Autoformer | 0.0736 | 0.0900 | 0.1773 | 0.1274 | 0.2860 | 0.3001 |
| BatLiNet | 0.3564 | 0.3413 | 0.5345 | 0.3706 | 0.6461 | 0.6096 |
| BatteryMFormer | 0.0121 | 0.0139 | 0.0650 | 0.0637 | 0.0507 | 0.0573 |
| BiGRU | 0.0730 | 0.0898 | 0.2066 | 0.1381 | 0.1536 | 0.1442 |
| BiLSTM | 0.0740 | 0.0872 | 0.1775 | 0.1209 | 0.1577 | 0.1482 |
| CNN | 0.0692 | 0.0882 | 0.1859 | 0.1237 | 0.2265 | 0.2015 |
| DLinear | 0.1711 | 0.1796 | 0.1402 | 0.0972 | 0.1651 | 0.1503 |
| GRU | 0.0709 | 0.0875 | 0.1876 | 0.1306 | 0.1690 | 0.1612 |
| IC2ML | 0.0731 | 0.0957 | 0.2169 | 0.1493 | 0.1270 | 0.1214 |
| iTransformer | 0.0806 | 0.0944 | 0.1925 | 0.1278 | 0.1985 | 0.1815 |
| LSTM | 0.0888 | 0.1028 | 0.1611 | 0.1115 | 0.1692 | 0.1588 |
| MICN | 0.0769 | 0.0861 | 0.2945 | 0.1844 | 2.1498 | 1.8472 |
| MLP | 0.0842 | 0.0953 | 0.1306 | 0.0913 | 0.6507 | 0.5476 |
| PatchTST | 0.0714 | 0.0865 | 0.1399 | 0.0984 | 0.2721 | 0.2946 |
| Severson | 0.2878 | 0.0833 | 0.1984 | 0.0327 | 2.0502 | 0.1028 |
| TimeMixer | 0.0777 | 0.0925 | 0.2567 | 0.1801 | 2.9444 | 2.6025 |
| Transformer | 0.0703 | 0.0878 | 0.1615 | 0.1127 | 0.2558 | 0.2583 |

## Acknowledgements

BatteryBench is built on top of:

- [**BatteryML**](https://github.com/microsoft/BatteryML) (Microsoft) — battery data preprocessing pipeline and dataset abstractions
- [**BatteryLife**](https://github.com/Ruifeng-Tan/BatteryLife) — multi-chemistry battery degradation benchmark and dataset collection
