# 训练和测评调度

使用独立 conda 环境 `zw@BatteryBench`，默认只使用物理 GPU 1、2、3，GPU 0 不参与。

```bash
bash execute_pipeline.sh --dry-run
bash execute_pipeline.sh
```

后台运行：

```bash
nohup bash execute_pipeline.sh > pipeline_main.log 2>&1 &
```

运行顺序：

1. 测评可信的非 BatLiNet 实验，每卡 6 个槽，总计 18 个并发进程。
2. 训练并测评 156 个未完成的其他模型实验，每卡 4 个槽，总计 12 个并发进程。
3. 测评可信的 BatLiNet 实验，每卡 6 个槽，总计 18 个并发进程。
4. 最后训练并测评 13 个未完成的 BatLiNet 实验，每卡 1 个槽，总计 3 个并发进程。

可信实验合计 1031 个。配置额外独占模型时，这些模型训练排在其他模型训练之后、全部 BatLiNet 任务之前。

每阶段等待全部进程结束，再开始下一阶段；训练任务成功后在同一个槽内接着测评。BatLiNet 不与其他任务共享同一张 GPU。子进程的可见 GPU 被限制到分配的单卡，因此子进程参数 `--gpu 0` 表示所分配的物理 GPU，不是物理 GPU 0。

可信实验清单在 `configs/trusted_experiments.json`；不能仅因为存在权重文件，就将未完成的实验标为可信。普通域每个实验包含三个 split；four_level 包含六个测试集，空测试集必须由新代码明确记录为 `status: empty`。

只有权重/scaler、新格式结果 JSON、完整逐样本 CSV、必要的轨迹 NPZ 和 seed 汇总均齐全，才会跳过该实验。脚本不会复制旧指标或自动推送 Git。

日志保存到 `log/<时间>_pipeline/`，包括每个实验的日志、计划和最终失败清单。恢复状态保存到 `pipeline_state/`。这两个目录的临时日志和状态不需要推送。重跑同一命令可跳过已完成的新格式测评；完整训练成功而测评失败的任务只重试测评；中断的训练会重新训练，半途权重不被当作可信权重。

同一状态目录有进程锁，防止重复启动。Ctrl-C 或 SIGTERM 会停止子进程。单个任务失败会保留错误和日志，其余任务继续，最终脚本以非零状态退出。

默认独占模型是 `batlinet`，可以用 `--large-models` 扩展：

```bash
bash run_pipeline.sh --large-models batlinet soh_traj/mlp
```

所有项目依赖及版本统一维护在 `requirements.txt`，Python 版本由文件中的 `# Python: 3.11` 声明。`execute_pipeline.sh` 读取该声明，必要时创建 `zw@BatteryBench` 环境，再执行 `pip install -r requirements.txt` 和 `pip check`，最后启动任务。已有环境会按相同清单检查和安装。`--dry-run` 仅预演，不安装依赖或启动实验。

已经安装好环境时也可以直接运行 `bash run_pipeline.sh`。环境安装和调度使用同一进程锁，防止运行中修改环境。
