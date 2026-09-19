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

流水线保留 GPU 0。测评时每张启用的 GPU 使用 6 个槽，普通模型训练使用 4 个槽，BatLiNet 最后运行且每张 GPU 只使用 1 个槽。恢复判定同时检查完整权重和测评产物。

## 结果

旧结果使用了旧输入定义，现已全部移除。所有模型正在从头训练；普通域的种子 1–5 共用同一个固定训练、验证和测试划分，完成后各指标按五个种子的 **均值 ± 标准差** 汇报。

| 任务 | MAE | RMSE | MAPE | ACC15 |
|---|---:|---:|---:|---:|
| RUL |  |  |  |  |
| SOH Point |  |  |  | — |
| SOH Trajectory |  |  |  | — |

结果目录如下：

    results/<domain>/<task>/<model>/
    ├── seed<seed>/
    │   ├── best.pt
    │   ├── results.json
    │   └── test/
    │       ├── predictions.csv
    │       ├── trajectories.npz
    │       └── <level_dataset>/    # 仅 Four-Level
    └── summary.json

Git 发布 JSON 汇总和 CSV 预测；权重、拟合模型、缓存和轨迹 NPZ 只保留在本地。

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
