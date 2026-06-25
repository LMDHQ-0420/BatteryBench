# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List
from pathlib import Path

from battery_data import BatteryData, CycleData, CyclingProtocol
from preprocess.base import BasePreprocessor

class NAionPreprocessor(BasePreprocessor):
    def process(self, parent_dir, **kwargs) -> List[BatteryData]:
        path = Path(parent_dir)
        files_path = os.listdir(path)
        batch1_cells = [i for i in files_path if i.endswith('.xlsx')]
        batch2_cells = [i for i in files_path if i.endswith('.csv')]
        process_batteries_num = 0
        skip_batteries_num = 0
        # preprocess batch1 data
        for cell in tqdm(batch1_cells, desc=f'Processing NA-ion batch1 cells'):
            cell_name = cell.split('.')[0]
            if cell_name in FILES_TO_DROP:
                continue
            cell_name = 'NA-ion_' + cell_name

            # Step1: judge whether to skip the processed file if exists
            whether_to_skip = self.check_processed_file(cell_name)
            if whether_to_skip == True:
                skip_batteries_num += 1
                continue

            # Step2: get timeseries and cycle_data from files of each cell
            df = pd.read_excel(path / cell, sheet_name='Record')
            df['Discharge_capacity/Ah'] = df['Capacity/Ah']
            df['Charge_capacity/Ah'] = df['Capacity/Ah']
            capacity = 1.0
            battery = organize_cell(df, cell_name, capacity)
            self.dump_single_file(battery)
            process_batteries_num += 1

            if not self.silent:
                tqdm.write(f'File: {battery.cell_id} dumped to pkl file')

        # preprocess batch2 data
        for cell in tqdm(batch2_cells, desc=f'Processing NA-ion batch2 cells'):
            cell_name = cell.split('.')[0]
            if cell_name in FILES_TO_DROP:
                continue
            cell_name = 'NAion_' + cell_name

            # Step1: judge whether to skip the processed file if exists
            whether_to_skip = self.check_processed_file(cell_name)
            if whether_to_skip == True:
                skip_batteries_num += 1
                continue

            # Step2: get timeseries and cycle_data from files of each cell
            df = pd.read_csv(path / cell, encoding='gbk')
            df = df.iloc[:, 1:8]
            capacity = 1.0
            cleaned_df = clean_data(df)

            battery = organize_cell(cleaned_df, cell_name, capacity)
            self.dump_single_file(battery)
            process_batteries_num += 1

            if not self.silent:
                tqdm.write(f'File: {battery.cell_id} dumped to pkl file')

        return process_batteries_num, skip_batteries_num

def organize_cell(timeseries_df, name, C):
    timeseries_df = timeseries_df.sort_values('TestTime')
    cycle_data = []
    for cycle_index, df in timeseries_df.groupby('Cycle'):
        cycle_data.append(CycleData(
            cycle_number=int(cycle_index),
            voltage_in_V=df['Voltage/V'].tolist(),
            current_in_A=df['Current/A'].tolist(),
            temperature_in_C=None,
            discharge_capacity_in_Ah=df['Discharge_capacity/Ah'].tolist(),
            charge_capacity_in_Ah=df['Charge_capacity/Ah'].tolist(),
            time_in_s=df['TestTime'].tolist()
        ))
    # Charge Protocol is constant current
    charge_protocol = [CyclingProtocol(
        rate_in_C=2.5, start_soc=0.0, end_soc=1.0
    )]
    discharge_protocol = [CyclingProtocol(
        rate_in_C=2.5, start_soc=1.0, end_soc=0.0
    )]

    soc_interval = [0, 1]

    return BatteryData(
        cell_id=name,
        cycle_data=cycle_data,
        form_factor='cylindrical_18650',
        anode_material='Unknown',
        cathode_material='NAion',
        discharge_protocol=discharge_protocol,
        charge_protocol=charge_protocol,
        nominal_capacity_in_Ah=C,
        min_voltage_limit_in_V=2.0,
        max_voltage_limit_in_V=4.0,
        SOC_interval=soc_interval
    )

def clean_data(df):
    cleaned_df = pd.DataFrame()
    cleaned_df['Cycle'] = df['循环号']
    cleaned_df['Voltage/V'] = df['电压(V)'].astype(float)
    cleaned_df['Current/A'] = df['电流(A)'].astype(float)
    cleaned_df['Discharge_capacity/Ah'] = df['放电容量(Ah)'].replace('-', 0).astype(float)
    cleaned_df['Charge_capacity/Ah'] = df['充电容量(Ah)'].replace('-', 0).astype(float)
    cleaned_df['TestTime'] = df['总时间(hh:mm:ss)']
    cleaned_df = cleaned_df.loc[cleaned_df['Cycle'] < len(set(cleaned_df['Cycle'].values))]
    time_in_s = convert_to_s(cleaned_df['TestTime'].values.tolist())
    cleaned_df['TestTime'] = time_in_s
    return cleaned_df

def convert_to_s(time_list):
    time_in_s = []
    for time in time_list:
        h = float(str(time).split(':')[0])
        m = float(str(time).split(':')[1])
        s = float(str(time).split(':')[2])
        seconds = h * 3600 + m * 60 + s
        time_in_s.append(seconds)
    return time_in_s

# problematic cells
FILES_TO_DROP = [
    '2750-30_20250115171823_DefaultGroup_45_2',
    '4000-30_20250115110135_DefaultGroup_45_7',
    '5000-25_20250115110326_DefaultGroup_38_5',
    '5000-25_20250115110326_DefaultGroup_38_7',
    '5000-25_20250115110326_DefaultGroup_38_8',
    '270040-4-8-40'
]