"""
config.py — 配置加载与域定义
"""

import os
import yaml
from typing import Union


DOMAIN_CFG = {
    'li_ion':   'configs/domains/li_ion.yaml',
    'na_ion':   'configs/domains/na_ion.yaml',
    'zn_ion':   'configs/domains/zn_ion.yaml',
    'calb':     'configs/domains/calb.yaml',
    'three_level': 'configs/domains/three_level.yaml',
}


def load_config(base_path: str, domain_path: str = None) -> dict:
    """
    加载 base_path（default.yaml），若提供 domain_path 则深度合并覆盖。
    """
    with open(base_path) as f:
        cfg = yaml.safe_load(f)

    if domain_path and os.path.exists(domain_path):
        with open(domain_path) as f:
            domain_cfg = yaml.safe_load(f)
        cfg = _deep_merge(cfg, domain_cfg)

    return cfg


def get_pkl_dir(data_cfg: dict) -> Union[str, list]:
    """
    从 data 配置段返回 pkl 路径（单目录或多目录列表）。
    """
    if 'pkl_dirs' in data_cfg:
        return data_cfg['pkl_dirs']
    if 'pkl_dir' in data_cfg:
        return data_cfg['pkl_dir']
    raise ValueError("data config must have 'pkl_dir' or 'pkl_dirs'")


def _deep_merge(base: dict, override: dict) -> dict:
    result = dict(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result
