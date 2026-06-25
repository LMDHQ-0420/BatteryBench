# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List
from pathlib import Path

from battery_data import BatteryData, CycleData, CyclingProtocol
from preprocess.base import BasePreprocessor

class MICHPreprocessor(BasePreprocessor):
    def process(self, parent_dir, **kwargs) -> List[BatteryData]:
        path = Path(parent_dir)
        cells = set(
            x.stem.split('_timeseries')[0]
            for x in path.glob('*MICH*timeseries*'))

        process_batteries_num = 0
        skip_batteries_num = 0
        for cell in tqdm(cells, desc='Processing MICH cells'):
            # Step1: judge whether to skip the processed file if exists
            whether_to_skip = self.check_processed_file(cell)
            if whether_to_skip == True:
                skip_batteries_num += 1
                continue

            # Step2: get timeseries and cycle_data from files of each cell
            timeseries_file = next(path.glob(f'*{cell}*timeseries*'))
            cycle_data_file = next(path.glob(f'*{cell}*cycle_data*'))
            timeseries_df = pd.read_csv(timeseries_file)
            if len(timeseries_df) == 0:
                continue

            # Step3: clean the abnormal data based on the Qd and threshold value
            timeseries_df = clean_cell(timeseries_df)

            # Step4: organize the cell data
            battery = organize_cell(timeseries_df, cell, 2.36)
            self.dump_single_file(battery)
            process_batteries_num += 1

            if not self.silent:
                tqdm.write(f'File: {battery.cell_id} dumped to pkl file')

        return process_batteries_num, skip_batteries_num

def organize_cell(timeseries_df, name, C):
    timeseries_df = timeseries_df.sort_values('Test_Time (s)')
    cycle_data = []
    for cycle_index, df in timeseries_df.groupby('Cycle_Index'):
        if cycle_index < 3:  # First 2 cycles are problematic
            continue
        cycle_data.append(CycleData(
            cycle_number=int(cycle_index - 2),
            voltage_in_V=df['Voltage (V)'].tolist(),
            current_in_A=df['Current (A)'].tolist(),
            temperature_in_C=df['Cell_Temperature (C)'].tolist(),
            discharge_capacity_in_Ah=df['Discharge_Capacity (Ah)'].tolist(),
            charge_capacity_in_Ah=df['Charge_Capacity (Ah)'].tolist(),
            time_in_s=df['Test_Time (s)'].tolist()
        ))
    # Charge Protocol is constant current
    charge_protocol = [CyclingProtocol(
        rate_in_C=1.0, start_soc=0.0, end_soc=1.0
    )]
    discharge_protocol = [CyclingProtocol(
        rate_in_C=1.0, start_soc=1.0, end_soc=0.0
    )]

    soc_interval = [0, 1]

    return BatteryData(
        cell_id=name,
        cycle_data=cycle_data,
        form_factor='pouch',
        anode_material='graphite',
        cathode_material='NMC',
        discharge_protocol=discharge_protocol,
        charge_protocol=charge_protocol,
        nominal_capacity_in_Ah=C,
        min_voltage_limit_in_V=3,
        max_voltage_limit_in_V=4.2,
        SOC_interval=soc_interval
    )

def clean_cell(timeseries_df, RPT_current=2.37):

    cycles = set([i for i in timeseries_df['Cycle_Index']])
    should_exclude = []

    for cycle in cycles:
        cycle_df = timeseries_df[timeseries_df['Cycle_Index'] == cycle]
        first_current_value = cycle_df['Current (A)'].tolist()[0]

        if RPT_current - 0.01 < first_current_value < RPT_current + 0.01:
            should_exclude.append(cycle)

    df = timeseries_df[timeseries_df['Cycle_Index'].isin(should_exclude)]

    # reset the cycle number and drop the first formation cycle
    cycle_number = set([i for i in df['Cycle_Index']])
    for current_index, new_index in zip(cycle_number, range(1, len(cycle_number) + 1)):
        df.loc[df['Cycle_Index'] == current_index, 'Cycle_Index'] = new_index

    return df

