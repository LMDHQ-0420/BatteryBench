"""
registry.py — 模型注册表（按任务分层）

结构: _REGISTRY[task][model_name] = ModelSpec
任务: 'rul', 'soh_point', 'soh_traj'

新增模型只需在此文件对应 task 下添加一条 ModelSpec，其余文件无需改动。
"""

from dataclasses import dataclass
from typing import Callable, Optional, Type

# ── RUL models ────────────────────────────────────────────────────────────────
from src.models.rul.baseline.mlp          import MLP          as RUL_MLP
from src.models.rul.baseline.gru          import GRU          as RUL_GRU
from src.models.rul.baseline.bigru        import BiGRU        as RUL_BiGRU
from src.models.rul.baseline.lstm         import LSTM         as RUL_LSTM
from src.models.rul.baseline.bilstm       import BiLSTM       as RUL_BiLSTM
from src.models.rul.baseline.cnn          import CNN          as RUL_CNN
from src.models.rul.baseline.dlinear      import DLinear      as RUL_DLinear
from src.models.rul.baseline.patchtst     import PatchTST     as RUL_PatchTST
from src.models.rul.baseline.transformer  import Transformer  as RUL_Transformer
from src.models.rul.baseline.autoformer   import Autoformer   as RUL_Autoformer
from src.models.rul.baseline.itransformer import iTransformer as RUL_iTransformer
from src.models.rul.baseline.micn         import MICN         as RUL_MICN
from src.models.rul.baseline.timemixer    import TimeMixer    as RUL_TimeMixer
from src.models.rul.baseline.ic2ml        import IC2ML        as RUL_IC2ML
from src.models.rul.baseline.batlinet     import BatLiNet     as RUL_BatLiNet
from src.models.rul.baseline.batterymformer import BatteryMFormer as RUL_BatteryMFormer

# ── SOH point models ──────────────────────────────────────────────────────────
from src.models.soh_point.baseline.mlp          import MLP          as SP_MLP
from src.models.soh_point.baseline.gru          import GRU          as SP_GRU
from src.models.soh_point.baseline.bigru        import BiGRU        as SP_BiGRU
from src.models.soh_point.baseline.lstm         import LSTM         as SP_LSTM
from src.models.soh_point.baseline.bilstm       import BiLSTM       as SP_BiLSTM
from src.models.soh_point.baseline.cnn          import CNN          as SP_CNN
from src.models.soh_point.baseline.dlinear      import DLinear      as SP_DLinear
from src.models.soh_point.baseline.patchtst     import PatchTST     as SP_PatchTST
from src.models.soh_point.baseline.transformer  import Transformer  as SP_Transformer
from src.models.soh_point.baseline.autoformer   import Autoformer   as SP_Autoformer
from src.models.soh_point.baseline.itransformer import iTransformer as SP_iTransformer
from src.models.soh_point.baseline.micn         import MICN         as SP_MICN
from src.models.soh_point.baseline.timemixer    import TimeMixer    as SP_TimeMixer
from src.models.soh_point.baseline.ic2ml        import IC2ML        as SP_IC2ML
from src.models.soh_point.baseline.batlinet     import BatLiNet     as SP_BatLiNet
from src.models.soh_point.baseline.batterymformer import BatteryMFormer as SP_BatteryMFormer

# ── SOH traj models ───────────────────────────────────────────────────────────
from src.models.soh_traj.baseline.mlp          import MLP          as ST_MLP
from src.models.soh_traj.baseline.gru          import GRU          as ST_GRU
from src.models.soh_traj.baseline.bigru        import BiGRU        as ST_BiGRU
from src.models.soh_traj.baseline.lstm         import LSTM         as ST_LSTM
from src.models.soh_traj.baseline.bilstm       import BiLSTM       as ST_BiLSTM
from src.models.soh_traj.baseline.cnn          import CNN          as ST_CNN
from src.models.soh_traj.baseline.dlinear      import DLinear      as ST_DLinear
from src.models.soh_traj.baseline.patchtst     import PatchTST     as ST_PatchTST
from src.models.soh_traj.baseline.transformer  import Transformer  as ST_Transformer
from src.models.soh_traj.baseline.autoformer   import Autoformer   as ST_Autoformer
from src.models.soh_traj.baseline.itransformer import iTransformer as ST_iTransformer
from src.models.soh_traj.baseline.micn         import MICN         as ST_MICN
from src.models.soh_traj.baseline.timemixer    import TimeMixer    as ST_TimeMixer
from src.models.soh_traj.baseline.ic2ml        import IC2ML        as ST_IC2ML
from src.models.soh_traj.baseline.batlinet     import BatLiNet     as ST_BatLiNet
from src.models.soh_traj.baseline.batterymformer import BatteryMFormer as ST_BatteryMFormer

# ── Datasets（多样本 + attention mask，对齐 BatteryLife/BatteryMFormer）──────────
from src.data.cycle_dataset import (
    RULDataset      as _RULDataset,
    SOHPointDataset as _SOHPointDataset,
    SOHTrajDataset  as _SOHTrajDataset,
)

# ── Train functions ───────────────────────────────────────────────────────────
from src.train.rul.train_base       import train as _rul_base
from src.train.rul.train_batlinet   import train as _rul_batlinet
from src.train.rul.train_severson   import train as _rul_severson
from src.train.soh_point.train_base     import train as _sp_base
from src.train.soh_point.train_batlinet import train as _sp_batlinet
from src.train.soh_point.train_severson import train as _sp_severson
from src.train.soh_traj.train_base      import train as _st_base
from src.train.soh_traj.train_batlinet  import train as _st_batlinet
from src.train.soh_traj.train_severson  import train as _st_severson


@dataclass
class ModelSpec:
    build_fn:       Optional[Callable]   # None for sklearn models (severson)
    dataset_cls:    Type                 # Dataset class to instantiate
    train_fn:       Callable             # training function
    batch_size_cap: Optional[int] = None # hard cap on batch_size (None = no cap)


ALL_TASKS = {'rul', 'soh_point', 'soh_traj'}

_REGISTRY: dict[str, dict[str, ModelSpec]] = {
    'rul': {
        'severson': ModelSpec(
            build_fn=None, dataset_cls=_RULDataset, train_fn=_rul_severson,
        ),
        'mlp': ModelSpec(
            build_fn=lambda cfg: RUL_MLP(cfg), dataset_cls=_RULDataset, train_fn=_rul_base,
        ),
        'gru': ModelSpec(
            build_fn=lambda cfg: RUL_GRU(cfg), dataset_cls=_RULDataset, train_fn=_rul_base,
        ),
        'bigru': ModelSpec(
            build_fn=lambda cfg: RUL_BiGRU(cfg), dataset_cls=_RULDataset, train_fn=_rul_base,
        ),
        'lstm': ModelSpec(
            build_fn=lambda cfg: RUL_LSTM(cfg), dataset_cls=_RULDataset, train_fn=_rul_base,
        ),
        'bilstm': ModelSpec(
            build_fn=lambda cfg: RUL_BiLSTM(cfg), dataset_cls=_RULDataset, train_fn=_rul_base,
        ),
        'cnn': ModelSpec(
            build_fn=lambda cfg: RUL_CNN(cfg), dataset_cls=_RULDataset, train_fn=_rul_base,
        ),
        'dlinear': ModelSpec(
            build_fn=lambda cfg: RUL_DLinear(cfg), dataset_cls=_RULDataset, train_fn=_rul_base,
        ),
        'patchtst': ModelSpec(
            build_fn=lambda cfg: RUL_PatchTST(cfg), dataset_cls=_RULDataset, train_fn=_rul_base,
        ),
        'transformer': ModelSpec(
            build_fn=lambda cfg: RUL_Transformer(cfg), dataset_cls=_RULDataset, train_fn=_rul_base,
        ),
        'autoformer': ModelSpec(
            build_fn=lambda cfg: RUL_Autoformer(cfg), dataset_cls=_RULDataset, train_fn=_rul_base,
        ),
        'itransformer': ModelSpec(
            build_fn=lambda cfg: RUL_iTransformer(cfg), dataset_cls=_RULDataset, train_fn=_rul_base,
        ),
        'micn': ModelSpec(
            build_fn=lambda cfg: RUL_MICN(cfg), dataset_cls=_RULDataset, train_fn=_rul_base,
        ),
        'timemixer': ModelSpec(
            build_fn=lambda cfg: RUL_TimeMixer(cfg), dataset_cls=_RULDataset, train_fn=_rul_base,
        ),
        'ic2ml': ModelSpec(
            build_fn=lambda cfg: RUL_IC2ML(cfg), dataset_cls=_RULDataset, train_fn=_rul_base,
        ),
        'batlinet': ModelSpec(
            build_fn=lambda cfg: RUL_BatLiNet(cfg), dataset_cls=_RULDataset,
            train_fn=_rul_batlinet, batch_size_cap=8,
        ),
        'batterymformer': ModelSpec(
            build_fn=lambda cfg: RUL_BatteryMFormer(cfg), dataset_cls=_RULDataset,
            train_fn=_rul_base,
        ),
    },
    'soh_point': {
        'severson': ModelSpec(
            build_fn=None, dataset_cls=_SOHPointDataset, train_fn=_sp_severson,
        ),
        'mlp': ModelSpec(
            build_fn=lambda cfg: SP_MLP(cfg), dataset_cls=_SOHPointDataset, train_fn=_sp_base,
        ),
        'gru': ModelSpec(
            build_fn=lambda cfg: SP_GRU(cfg), dataset_cls=_SOHPointDataset, train_fn=_sp_base,
        ),
        'bigru': ModelSpec(
            build_fn=lambda cfg: SP_BiGRU(cfg), dataset_cls=_SOHPointDataset, train_fn=_sp_base,
        ),
        'lstm': ModelSpec(
            build_fn=lambda cfg: SP_LSTM(cfg), dataset_cls=_SOHPointDataset, train_fn=_sp_base,
        ),
        'bilstm': ModelSpec(
            build_fn=lambda cfg: SP_BiLSTM(cfg), dataset_cls=_SOHPointDataset, train_fn=_sp_base,
        ),
        'cnn': ModelSpec(
            build_fn=lambda cfg: SP_CNN(cfg), dataset_cls=_SOHPointDataset, train_fn=_sp_base,
        ),
        'dlinear': ModelSpec(
            build_fn=lambda cfg: SP_DLinear(cfg), dataset_cls=_SOHPointDataset, train_fn=_sp_base,
        ),
        'patchtst': ModelSpec(
            build_fn=lambda cfg: SP_PatchTST(cfg), dataset_cls=_SOHPointDataset, train_fn=_sp_base,
        ),
        'transformer': ModelSpec(
            build_fn=lambda cfg: SP_Transformer(cfg), dataset_cls=_SOHPointDataset, train_fn=_sp_base,
        ),
        'autoformer': ModelSpec(
            build_fn=lambda cfg: SP_Autoformer(cfg), dataset_cls=_SOHPointDataset, train_fn=_sp_base,
        ),
        'itransformer': ModelSpec(
            build_fn=lambda cfg: SP_iTransformer(cfg), dataset_cls=_SOHPointDataset, train_fn=_sp_base,
        ),
        'micn': ModelSpec(
            build_fn=lambda cfg: SP_MICN(cfg), dataset_cls=_SOHPointDataset, train_fn=_sp_base,
        ),
        'timemixer': ModelSpec(
            build_fn=lambda cfg: SP_TimeMixer(cfg), dataset_cls=_SOHPointDataset, train_fn=_sp_base,
        ),
        'ic2ml': ModelSpec(
            build_fn=lambda cfg: SP_IC2ML(cfg), dataset_cls=_SOHPointDataset, train_fn=_sp_base,
        ),
        'batlinet': ModelSpec(
            build_fn=lambda cfg: SP_BatLiNet(cfg), dataset_cls=_SOHPointDataset,
            train_fn=_sp_batlinet, batch_size_cap=8,
        ),
        'batterymformer': ModelSpec(
            build_fn=lambda cfg: SP_BatteryMFormer(cfg), dataset_cls=_SOHPointDataset,
            train_fn=_sp_base,
        ),
    },
    'soh_traj': {
        'severson': ModelSpec(
            build_fn=None, dataset_cls=_SOHTrajDataset, train_fn=_st_severson,
        ),
        'mlp': ModelSpec(
            build_fn=lambda cfg: ST_MLP(cfg), dataset_cls=_SOHTrajDataset, train_fn=_st_base,
        ),
        'gru': ModelSpec(
            build_fn=lambda cfg: ST_GRU(cfg), dataset_cls=_SOHTrajDataset, train_fn=_st_base,
        ),
        'bigru': ModelSpec(
            build_fn=lambda cfg: ST_BiGRU(cfg), dataset_cls=_SOHTrajDataset, train_fn=_st_base,
        ),
        'lstm': ModelSpec(
            build_fn=lambda cfg: ST_LSTM(cfg), dataset_cls=_SOHTrajDataset, train_fn=_st_base,
        ),
        'bilstm': ModelSpec(
            build_fn=lambda cfg: ST_BiLSTM(cfg), dataset_cls=_SOHTrajDataset, train_fn=_st_base,
        ),
        'cnn': ModelSpec(
            build_fn=lambda cfg: ST_CNN(cfg), dataset_cls=_SOHTrajDataset, train_fn=_st_base,
        ),
        'dlinear': ModelSpec(
            build_fn=lambda cfg: ST_DLinear(cfg), dataset_cls=_SOHTrajDataset, train_fn=_st_base,
        ),
        'patchtst': ModelSpec(
            build_fn=lambda cfg: ST_PatchTST(cfg), dataset_cls=_SOHTrajDataset, train_fn=_st_base,
        ),
        'transformer': ModelSpec(
            build_fn=lambda cfg: ST_Transformer(cfg), dataset_cls=_SOHTrajDataset, train_fn=_st_base,
        ),
        'autoformer': ModelSpec(
            build_fn=lambda cfg: ST_Autoformer(cfg), dataset_cls=_SOHTrajDataset, train_fn=_st_base,
        ),
        'itransformer': ModelSpec(
            build_fn=lambda cfg: ST_iTransformer(cfg), dataset_cls=_SOHTrajDataset, train_fn=_st_base,
        ),
        'micn': ModelSpec(
            build_fn=lambda cfg: ST_MICN(cfg), dataset_cls=_SOHTrajDataset, train_fn=_st_base,
        ),
        'timemixer': ModelSpec(
            build_fn=lambda cfg: ST_TimeMixer(cfg), dataset_cls=_SOHTrajDataset, train_fn=_st_base,
        ),
        'ic2ml': ModelSpec(
            build_fn=lambda cfg: ST_IC2ML(cfg), dataset_cls=_SOHTrajDataset, train_fn=_st_base,
        ),
        'batlinet': ModelSpec(
            build_fn=lambda cfg: ST_BatLiNet(cfg), dataset_cls=_SOHTrajDataset,
            train_fn=_st_batlinet, batch_size_cap=8,
        ),
        'batterymformer': ModelSpec(
            build_fn=lambda cfg: ST_BatteryMFormer(cfg), dataset_cls=_SOHTrajDataset,
            train_fn=_st_base,
        ),
    },
}

ALL_MODELS = {task: set(models.keys()) for task, models in _REGISTRY.items()}


def get_spec(name: str, task: str = 'rul') -> ModelSpec:
    """返回模型的完整规格。"""
    task = task.lower()
    name = name.lower()
    if task not in _REGISTRY:
        raise ValueError(f"Unknown task '{task}'. Available: {sorted(_REGISTRY)}")
    task_reg = _REGISTRY[task]
    if name not in task_reg:
        raise ValueError(f"Unknown model '{name}' for task '{task}'. Available: {sorted(task_reg)}")
    return task_reg[name]
