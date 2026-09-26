<p align="center">
  <img src="asset/logo.svg" alt="BatteryBench logo" width="520">
</p>

<h1 align="center">BatteryBench</h1>

<p align="center">
  面向多种电池体系与泛化层级的统一寿命和健康状态预测基准。
</p>

<p align="center">
  <a href="README.md">English</a>
</p>

## 新闻

- 📊 **[2026/09/26]** 发布 15 个普通基线完整的五种子 Four-Level 结果；BatLiNet 和其余普通域实验仍在运行。
- 🔒 **[2026/09/19]** 三个任务统一改用完整充电曲线，并去除 SOH 点估计中的容量泄漏。
- 🧪 **[2026/09/19]** 开始按新的 400 点输入方案重新训练全部基线。
- 🧭 **[2026/07/15]** 引入跨批次、跨数据集、跨正极材料和跨离子体系的 Four-Level 泛化协议。
- 🚀 **[2026/06/26]** 发布 BatteryBench 基准与训练流水线。

## 项目介绍

BatteryBench 为电池寿命和健康状态预测提供统一的训练与测评接口，覆盖锂离子、CALB、钠离子和锌离子实验，并通过 Four-Level 协议评估同数据集跨批次（L1）、跨数据集（L2）、跨正极材料（L3）和跨离子体系（L4）泛化能力。

三个任务都只使用正电流对应的完整充电阶段，每条曲线线性重采样为 400 点。电压除以当前充电段最大电压，电流和充电容量除以电池额定容量。放电容量只用于生成 SOH、EOL 和 RUL 标签，绝不进入模型输入。

## 输入方案

| 任务 | 已观测输入 | 张量形状 | 通道 | 预测目标 |
|---|---|---|---|---|
| SOH Point | 当前循环的完整充电曲线 | <code>[B, 1, 2, 400]</code> | V, I | 当前循环 SOH |
| SOH Trajectory | 第 1 圈到第 u 圈的完整充电曲线 | <code>[B, S, 3, 400]</code> | V, I, Q<sub>charge</sub> | 第 u+1 圈到 EOL 的 SOH |
| RUL | 第 1 圈到第 u 圈的完整充电曲线 | <code>[B, S, 3, 400]</code> | V, I, Q<sub>charge</sub> | 电池寿命/EOL |

未观测的历史位置补零，并通过 <code>curve_attn_mask</code> 屏蔽。SOH 轨迹的损失和指标只计算最后观测圈之后的部分。所有任务都不使用放电曲线；SOH Point 对全部模型都不提供充电或放电容量，包括 IC2ML、BatLiNet 和 Severson。

## 数据划分

所有数据都在**电池层面**进行划分，同一块电池的不同循环不会出现在多个集合中。划分固定使用 <code>split_seed: 1</code>，训练种子 1–5 共用完全相同的训练、验证和测试电池；不同训练种子只改变模型初始化和优化过程。

| 数据域 | 训练集 | 验证集 | 测试集 | 划分方式 |
|---|---:|---:|---:|---|
| 锂离子 | 70% | 10% | 20% | 固定的电池级随机划分 |
| CALB | 60% | 20% | 20% | 固定的电池级随机划分 |
| 钠离子 | 60% | 20% | 20% | 固定的电池级随机划分 |
| 锌离子 | 60% | 20% | 20% | 固定的电池级随机划分 |

**Four-Level** 用来衡量测试分布逐步偏离训练池时的不同泛化能力。各层级代表不同类型的分布偏移，并不保证所有指标一定从 L1 到 L4 单调变差。训练池包括 HUST 第 1–7、10 批，MATR 第 1–3 批，以及 RWTH、SDU、Stanford、Tongji、ISU-ILCC、MICH、CALB 和 XJTU；每种正极材料分组固定抽取 8% 作为验证集。下列测试电池固定隔离，不参与训练和验证。

| 层级 | 泛化设置 | 固定测试集 |
|---|---|---|
| L1 | 相同化学体系、相同数据集，泛化到未见批次 | HUST 第 8–9 批 |
| L2 | 相同化学体系，泛化到未见批次/工况分布 | MATR 第 4 批 |
| L3 | 泛化到未见的锂离子正极材料与数据集分布 | CALCE、HNEI |
| L4 | 泛化到未见的离子化学体系 | 钠离子、锌离子 |

## 模型

| 类型 | 模型 |
|---|---|
| MLP / 线性 | MLP、DLinear |
| 循环网络 | GRU、BiGRU、LSTM、BiLSTM |
| 卷积网络 | CNN、MICN |
| Transformer / 混合器 | Transformer、PatchTST、Autoformer、iTransformer、TimeMixer |
| 电池专用模型 | IC2ML、BatLiNet、Severson |

## 安装

    conda create -n BatteryBench python=3.11
    conda activate BatteryBench
    pip install -r requirements.txt

原始数据放在 <code>data/raw/</code> 下，数据路径和实验参数位于 <code>configs/</code>。

## 快速开始

训练并测评一个实验：

    python scripts/train.py --domain li_ion --model gru --task rul --seed 1 --gpu 0
    python scripts/evaluate.py --domain li_ion --model gru --task rul --seed 1 --gpu 0

恢复完整流水线：

    bash run_pipeline.sh

流水线先依次运行 Four-Level 的 SOH Trajectory、SOH Point、RUL 的 seed 1，然后运行 Four-Level 的 seed 2–5、所有普通域实验，最后运行全部 BatLiNet。普通模型在 GPU 0 使用 2 个槽，在 GPU 1–3 各使用 3 个槽；BatLiNet 只在 GPU 1–3 各使用 1 个槽，绝不使用 GPU 0。恢复判定同时检查完整权重和测评产物。

## 结果

旧输入定义下的结果已经移除。下表展示已完成全部五个种子的 15 个普通基线中，各 Four-Level 层级平均 RMSE 最低的模型。数值为种子 1–5 的**均值 ± 总体标准差**。SOH 的 MAE、RMSE 使用 SOH 比例，RUL 误差使用循环数，MAPE 使用百分数。当前 RUL 构造下 L1 测试集没有有效样本，因此留空。

| 任务 | 层级 | 模型 | MAE | RMSE | MAPE |
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

完整的模型—层级结果与模型—测试集结果分别见 [Four-Level 层级汇总](results/four_level/summary_levels.md) 和 [测试集汇总](results/four_level/summary_test_sets.md)。仓库当前包含 284 个完整实验：15 个普通 Four-Level 基线的全部五个种子，以及 59 个锂离子 SOH Trajectory 实验。BatLiNet 和其余普通域实验尚未完成。

结果目录如下：

    results/<domain>/<task>/<model>/
    ├── seed<seed>/
    │   ├── results.json
    │   └── test/
    │       ├── predictions.csv
    │       └── <level_dataset>/    # 仅 Four-Level
    └── summary.json

Git 发布 JSON 汇总、CSV 预测和 Markdown 表格；所有 `.pt`、`.pkl` 和 `.npz` 文件只保留在本地。

## 项目结构

    BatteryBench/
    ├── asset/                  项目图片
    ├── configs/                默认配置和数据域配置
    ├── data/                   本地原始、预处理和缓存数据
    ├── results/                测评指标和预测
    ├── scripts/                训练、测评和流水线入口
    ├── src/
    │   ├── data/               数据集
    │   ├── evaluate/           指标与结果写入
    │   ├── models/             模型注册表与实现
    │   ├── preprocess/         数据预处理
    │   └── train/              训练流程
    ├── tests/                  回归测试
    ├── requirements.txt
    └── run_pipeline.sh

## 待办

- [ ] 完成仅充电输入方案下的全部重新训练和测评。
- [ ] 发布完整的五随机种子结果表。
