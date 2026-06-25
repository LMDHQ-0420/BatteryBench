# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import json
import h5py
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm
from typing import List
from pathlib import Path

from battery_data import BatteryData, CycleData, CyclingProtocol
from preprocess.base import BasePreprocessor

class TongjiPreprocessor(BasePreprocessor):
    def process(self, parent_dir, **kwargs) -> List[BatteryData]:
        filenames = ['Dataset_1_NCA_battery', 'Dataset_2_NCM_battery', 'Dataset_3_NCM_NCA_battery']
        for filename in filenames:
            path = Path(parent_dir) / filename
            files_path = os.listdir(path)
            cells = [i for i in files_path if i.endswith('.csv')]

            process_batteries_num = 0
            skip_batteries_num = 0
            for cell in tqdm(cells, desc='Processing Tongji file'):
                cell_df = pd.read_csv(path / cell)
                cell = cell.split('.csv')[0]

                if filename == 'Dataset_1_NCA_battery':
                    cell = 'Tongji1_' + cell

                elif filename == 'Dataset_2_NCM_battery':
                    cell = 'Tongji2_' + cell
                elif filename == 'Dataset_3_NCM_NCA_battery':
                    cell = 'Tongji3_' + cell

                cell = cell.replace('-#', '--')
                # Step1: judge whether to skip the processed file
                whether_to_skip = self.check_processed_file(cell)
                if whether_to_skip == True:
                    skip_batteries_num += 1
                    continue

                cell_df = clean(cell_df, filename)

                # Step2: organize the cell data
                battery = organize_cell(cell_df, cell, filename)
                self.dump_single_file(battery)
                process_batteries_num += 1

                if not self.silent:
                    tqdm.write(f'File: {battery.cell_id} dumped to pkl file')

        return process_batteries_num, skip_batteries_num

def organize_cell(timeseries_df, name, path):
    timeseries_df = timeseries_df.sort_values('time/s')
    cycle_data = []
    for cycle_index, df in timeseries_df.groupby('cycle number'):
        if cycle_index < 2:
            continue
        cycle_data.append(CycleData(
            cycle_number=int(cycle_index),
            voltage_in_V=df['Ecell/V'].tolist(),
            current_in_A=list(df['<I>/mA'].values * 0.001),
            temperature_in_C=None,
            discharge_capacity_in_Ah=list(df['Q discharge/mA.h'].values * 0.001),
            charge_capacity_in_Ah=list(df['Q charge/mA.h'].values * 0.001),
            time_in_s=df['time/s'].tolist()
        ))
    # Charge Protocol is constant current
    charge_protocol = [CyclingProtocol(
        rate_in_C='', start_soc=0, end_soc=1.0
    )]
    discharge_protocol = [CyclingProtocol(
        rate_in_C='', start_soc=1.0, end_soc=1
    )]

    if 'Dataset_1_NCA_battery' in path:
        anode_material = 'Graphite/Si'
        cathode_material = 'NCA'
        nominal_capacity_in_Ah = 3.5
        min_voltage_limit_in_V = 2.65

    elif 'Dataset_2_NCM_battery' in path:
        anode_material = 'Graphite/Si'
        cathode_material = 'NMC'
        nominal_capacity_in_Ah = 3.5
        min_voltage_limit_in_V = 2.5

    elif 'Dataset_3_NCM_NCA_battery' in path:
        anode_material = 'Graphite'
        cathode_material = 'NMC+NCA'
        nominal_capacity_in_Ah = 2.5
        min_voltage_limit_in_V = 2.5

    return BatteryData(
        cell_id=name,
        cycle_data=cycle_data,
        form_factor='cylindrical_18650',
        anode_material=anode_material,
        cathode_material=cathode_material,
        discharge_protocol=discharge_protocol,
        charge_protocol=charge_protocol,
        nominal_capacity_in_Ah=nominal_capacity_in_Ah,
        min_voltage_limit_in_V=min_voltage_limit_in_V,
        max_voltage_limit_in_V=4.2,
        SOC_interval=[0, 1]
    )

def clean(df, filename):
    cycle_indices = df['cycle number'].unique()
    remain_df = pd.DataFrame()
    for cycle_index in cycle_indices:
        cycle_df = df.loc[df['cycle number'] == cycle_index]
        Qd = max(cycle_df['Q discharge/mA.h'].values)
        if filename == 'Dataset_1_NCA_battery':
            if Qd <= 2000:
                # remove the cycle whose capacity suddenly drops to a very low value
                continue
        elif filename == 'Dataset_2_NCM_battery':
            if Qd <= 2500:
                continue
        elif filename == 'Dataset_3_NCM_NCA_battery':
            if Qd <= 1600 or Qd >= 2700:
                continue

        remain_df = pd.concat([remain_df, cycle_df], ignore_index=True)
    return remain_df