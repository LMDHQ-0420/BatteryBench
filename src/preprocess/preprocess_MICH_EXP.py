# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import numpy as np
import pandas as pd
import re

from tqdm import tqdm
from typing import List
from pathlib import Path

from battery_data import BatteryData, CycleData, CyclingProtocol
from preprocess.base import BasePreprocessor


class MICH_EXPPreprocessor(BasePreprocessor):
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
            cycle_data_df = pd.read_csv(cycle_data_file)
            if len(timeseries_df) == 0:
                continue

            # Step3: clean the abnormal data based on the Qd and threshold value
            timeseries_df, _ = clean_cell(
                timeseries_df, cycle_data_df)

            # Step4: organize the cell data
            battery = organize_cell(timeseries_df, cell, 5.0)
            self.dump_single_file(battery)
            process_batteries_num += 1

            if not self.silent:
                tqdm.write(f'File: {battery.cell_id} dumped to pkl file')

        return process_batteries_num, skip_batteries_num

def organize_cell(timeseries_df, name, C):
    timeseries_df = timeseries_df.sort_values('Test_Time (s)')
    cycle_data = []
    for cycle_index, df in timeseries_df.groupby('Cycle_Index'):
        cycle_data.append(CycleData(
            cycle_number=int(cycle_index),
            voltage_in_V=df['Voltage (V)'].tolist(),
            current_in_A=df['Current (A)'].tolist(),
            temperature_in_C=df['Cell_Temperature (C)'].tolist(),
            discharge_capacity_in_Ah=df['Discharge_Capacity (Ah)'].tolist(),
            charge_capacity_in_Ah=df['Charge_Capacity (Ah)'].tolist(),
            time_in_s=df['Test_Time (s)'].tolist()
        ))
    # Charge Protocol is constant current
    if re.match(r'^MICH_(01|02|03|13|14|15)', name):
        charge_protocol = [CyclingProtocol(
            rate_in_C=0.2, start_soc=0.0, end_soc=1.0
        )]
        discharge_protocol = [CyclingProtocol(
            rate_in_C=0.2, start_soc=1.0, end_soc=0.0
        )]
    elif re.match(r'^MICH_(04|05|06)', name):
        charge_protocol = [CyclingProtocol(
            rate_in_C=1.5, start_soc=0.0, end_soc=1.0
        )]
        discharge_protocol = [CyclingProtocol(
            rate_in_C=1.5, start_soc=1.0, end_soc=0.0
        )]
    elif re.match(r'^MICH_(07|08|09)', name):
        charge_protocol = [CyclingProtocol(
            rate_in_C=2.0, start_soc=0.0, end_soc=1.0
        )]
        discharge_protocol = [CyclingProtocol(
            rate_in_C=2.0, start_soc=1.0, end_soc=0.0
        )]
    elif re.match(r'^MICH_(10|11|12)', name):
        charge_protocol = [CyclingProtocol(
            rate_in_C=0.2, start_soc=0.0, end_soc=1.0
        )]
        discharge_protocol = [CyclingProtocol(
            rate_in_C=1.5, start_soc=1.0, end_soc=0.0
        )]
    elif re.match(r'^MICH_(10|11|12|16|17|18)', name):
        charge_protocol = [CyclingProtocol(
            rate_in_C=0.2, start_soc=0.0, end_soc=1.0
        )]
        discharge_protocol = [CyclingProtocol(
            rate_in_C=1.5, start_soc=1.0, end_soc=0.0
        )]

    if '50-100' in name:
        soc_interval = [0.5, 1]
    else:
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

def clean_cell(timeseries_df, cycle_data_df, shifts=2, **kwargs):
    # step1: get Qd(discharge capacity)
    Qd = cycle_data_df['Discharge_Capacity (Ah)'].values

    # step2: set the comparing shift=2 and clean the data
    # which means code will check the one before and the one after the cycle
    if isinstance(shifts, int):
        shifts = range(1, shifts+1)
    should_exclude = False
    for shift in shifts:# shift=1, 2
        should_exclude |= _clean_helper(Qd, shift, **kwargs)

    cycle_to_exclude = set(
        cycle_data_df[should_exclude]['Cycle_Index'].values.astype(int))
    # Also include those missing cycles into the `cycle_to_exclude`
    cycles = timeseries_df.Cycle_Index.unique()
    for cycle in range(1, int(cycles.max()+1)):
        if cycle not in cycles:
            cycle_to_exclude.add(cycle)

    cdfs, tdfs = [], []
    for cycle in cycle_to_exclude:
        imp_cycle = find_forward_imputation_cycle(cycle, cycle_to_exclude)
        if imp_cycle not in cycle_data_df.Cycle_Index.unique():
            raise ValueError(
                f'No valid imputation cycle ({cycle}->{imp_cycle})!')
        tdf = timeseries_df[timeseries_df.Cycle_Index == imp_cycle].copy()
        cdf = cycle_data_df[cycle_data_df.Cycle_Index == imp_cycle].copy()
        tdf['Cycle_Index'] = cycle
        cdf['Cycle_Index'] = cycle
        tdfs.append(tdf)
        cdfs.append(cdf)
    timeseries_df = pd.concat([
        timeseries_df[~timeseries_df.Cycle_Index.isin(cycle_to_exclude)], *tdfs
    ]).reset_index(drop=True).sort_values('Test_Time (s)')
    cycle_data_df = pd.concat([
        cycle_data_df[~cycle_data_df.Cycle_Index.isin(cycle_to_exclude)], *cdfs
    ]).reset_index(drop=True).sort_values('Test_Time (s)')

    timeseries_df = timeseries_df.dropna(subset=['Charge_Capacity (Ah)', 'Discharge_Capacity (Ah)'])

    return timeseries_df, cycle_data_df

def find_forward_imputation_cycle(cycle, to_exclude):
    # First look back, then look forward
    while cycle > 0 and cycle in to_exclude:
        cycle -= 1
    while cycle == 0 or cycle in to_exclude:
        cycle += 1
    return cycle

def _clean_helper(Qd, shift, **kwargs):
    # calculate left-side deviation
    diff_left = abs(Qd - np.roll(Qd, shift))
    diff_left[:shift] = np.inf

    # calculate right-side deviation
    diff_right = abs(Qd - np.roll(Qd, -shift))
    diff_right[-shift:] = np.inf

    # return a minimal deviation each from left and right side
    diff = np.amin([diff_left, diff_right], 0)
    # should_exclude = find_glitches(diff, alpha)

    # drop the abnormal value
    should_exclude = hampel_filter(diff, **kwargs)
    return should_exclude

def hampel_filter(num, ths=3):
    med = np.median(num)
    diff_with_med = abs(num - med)
    ths = np.median(diff_with_med) * ths
    return diff_with_med > ths
