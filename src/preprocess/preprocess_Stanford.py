# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import json
import tarfile
import gzip
import shutil
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List
from pathlib import Path

from battery_data import BatteryData, CycleData, CyclingProtocol
from preprocess.base import BasePreprocessor


class StanfordPreprocessor(BasePreprocessor):

    def process(self, parent_dir, **kwargs) -> List[BatteryData]:

        parent_dir = Path(parent_dir)
        raw_file = parent_dir / "data.tar.gz"
        extract_root = parent_dir / "data"
        maccor_path = extract_root / "maccor"

        # =========================
        # Step 1: 解压 tar.gz
        # =========================
        if not extract_root.exists():
            if not self.silent:
                tqdm.write("Unzipping Stanford dataset (tar.gz)...")
            with tarfile.open(raw_file, "r:gz") as tar:
                tar.extractall(path=parent_dir)
        else:
            if not self.silent:
                tqdm.write("Stanford dataset already extracted.")

        # =========================
        # Step 2: 解压所有 .gz → .json
        # =========================
        gz_files = list(maccor_path.glob("*.gz"))

        for gz_file in tqdm(gz_files, desc="Unzipping .gz files", disable=self.silent):
            json_file = gz_file.with_suffix("")

            if not json_file.exists():
                if not self.silent:
                    tqdm.write(f"Unzipping {gz_file.name}")
                with gzip.open(gz_file, "rb") as f_in:
                    with open(json_file, "wb") as f_out:
                        shutil.copyfileobj(f_in, f_out)

        # =========================
        # Step 3: 重新扫描 json 文件
        # =========================
        json_files = list(maccor_path.glob("*.json"))

        process_batteries_num = 0
        skip_batteries_num = 0

        for json_file in tqdm(json_files, desc="Processing Stanford cells"):

            cell_name = "Stanford_" + json_file.stem

            # 跳过特殊文件
            if json_file.name == "Nova_Regular_197.034.json":
                continue

            # 检查是否已处理
            if self.check_processed_file(cell_name):
                skip_batteries_num += 1
                continue

            # =========================
            # 读取数据
            # =========================
            with open(json_file, "r") as f:
                data = json.load(f)

            cell_data = data.get("cycles_interpolated", [])
            if len(cell_data) == 0:
                continue

            # =========================
            # 清洗 + 组织
            # =========================
            cleaned_df = clean_cell(cell_data)
            battery = organize_cell(cleaned_df, cell_name, C=0.24)

            self.dump_single_file(battery)
            process_batteries_num += 1

            if not self.silent:
                tqdm.write(f"Dumped: {battery.cell_id}")

        return process_batteries_num, skip_batteries_num


# =========================================================
# 数据组织函数
# =========================================================

def organize_cell(timeseries_df, name, C):

    timeseries_df = timeseries_df.sort_values("test_time")
    cycle_data = []

    for cycle_index, df in timeseries_df.groupby("cycle_index"):
        cycle_data.append(
            CycleData(
                cycle_number=int(cycle_index),
                voltage_in_V=df["voltage"].tolist(),
                current_in_A=df["current"].tolist(),
                temperature_in_C=df["temperature"].tolist(),
                discharge_capacity_in_Ah=df["discharge_capacity"].tolist(),
                charge_capacity_in_Ah=df["charge_capacity"].tolist(),
                time_in_s=df["test_time"].tolist(),
            )
        )

    charge_protocol = [
        CyclingProtocol(rate_in_C=1.0, start_soc=0.0, end_soc=1.0)
    ]

    discharge_protocol = [
        CyclingProtocol(rate_in_C=0.75, start_soc=1.0, end_soc=0.0)
    ]

    return BatteryData(
        cell_id=name,
        cycle_data=cycle_data,
        form_factor="pouch",
        anode_material="graphite",
        cathode_material="NMC",
        discharge_protocol=discharge_protocol,
        charge_protocol=charge_protocol,
        nominal_capacity_in_Ah=C,
        min_voltage_limit_in_V=3.0,
        max_voltage_limit_in_V=4.4,
        SOC_interval=[0, 1],
    )


# =========================================================
# 数据清洗函数
# =========================================================

def clean_cell(cell_data):

    df = pd.DataFrame(cell_data)

    # 去掉 formation cycle
    unique_cycles = sorted(set(df["cycle_index"]) - {0})
    mapping = {old: new for new, old in enumerate(unique_cycles, start=1)}
    df["cycle_index"] = df["cycle_index"].map(mapping)

    df = df[df["cycle_index"] != 1]
    df["cycle_index"] -= 1

    # ===== 完全向量化版本 =====

    # 删除小负电流
    df = df[~((df["current"] > -0.001) & (df["current"] < 0))]

    # 统一修正容量
    df.loc[df["current"] >= 0, "discharge_capacity"] = 0
    df.loc[df["current"] < 0, "charge_capacity"] = 0

    return df.reset_index(drop=True)