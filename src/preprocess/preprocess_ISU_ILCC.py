# Licensed under the MIT License.
# Copyright (c) Microsoft Corporation.

import os
import json
import zipfile
import numpy as np
import pandas as pd

from tqdm import tqdm
from typing import List
from pathlib import Path

from battery_data import BatteryData, CycleData, CyclingProtocol
from preprocess.base import BasePreprocessor


class ISU_ILCCPreprocessor(BasePreprocessor):

    def process(self, parentdir, **kwargs) -> List[BatteryData]:
        raw_file = Path(parentdir) / '22582234.zip'

        # ===================== unzip main =====================
        extract_root = raw_file.parent / '22582234'
        if not extract_root.exists():
            with zipfile.ZipFile(raw_file, 'r') as zip_ref:
                zip_ref.extractall(extract_root)
        else:
            if not self.silent:
                tqdm.write('Skipping ISU_ILCC dataset, already exists')

        # ===================== unzip cycling =====================
        cycle_zip_path = extract_root / 'Cycling_json.zip'
        cycling_root = extract_root / 'Cycling_json'

        if not cycling_root.exists():
            with zipfile.ZipFile(cycle_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_root)
        else:
            if not self.silent:
                tqdm.write('Skipping cycling files, already exists')

        # ===================== unzip rpt =====================
        rpt_zip_path = extract_root / 'RPT_json.zip'
        rpt_root = extract_root / 'RPT_json'

        if not rpt_root.exists():
            with zipfile.ZipFile(rpt_zip_path, 'r') as zip_ref:
                zip_ref.extractall(extract_root)
        else:
            if not self.silent:
                tqdm.write('Skipping RPT files, already exists')

        valid_cells = pd.read_csv(extract_root / 'Valid_cells.csv').values.flatten().tolist()

        batch2 = {
            'G57C1', 'G57C2', 'G57C3', 'G57C4',
            'G58C1', 'G26C3', 'G49C1', 'G49C2',
            'G49C3', 'G49C4', 'G50C1', 'G50C3', 'G50C4'
        }

        process_batteries_num = 0
        skip_batteries_num = 0

        for cell in tqdm(valid_cells, desc='Processing ISU_ILCC cells'):

            # drop abnormal
            if cell in {'G42C4', 'G9C4', 'G25C4'} or 'G26' in cell or 'G11' in cell:
                continue

            if self.check_processed_file('ISU-ILCC_' + cell):
                skip_batteries_num += 1
                continue

            subfolder = 'Release 2.0' if cell in batch2 else 'Release 1.0'

            # ===================== load json once =====================
            cycling_dict = convert_cycling_to_dict(extract_root, cell, subfolder)
            rpt_dict = convert_RPT_to_dict(extract_root, cell, subfolder)

            # ===================== build dataframe fast =====================
            cycle_frames = []

            num_cycles = len(cycling_dict['QV_charge']['I'])

            for index in range(num_cycles):

                charge_I = np.asarray(cycling_dict['QV_charge']['I'][index])
                discharge_I = np.asarray(cycling_dict['QV_discharge']['I'][index])

                charge_V = np.asarray(cycling_dict['QV_charge']['V'][index])
                discharge_V = np.asarray(cycling_dict['QV_discharge']['V'][index])

                charge_t = np.asarray(cycling_dict['QV_charge']['t'][index])
                discharge_t = np.asarray(cycling_dict['QV_discharge']['t'][index])

                charge_Q = np.asarray(cycling_dict['QV_charge']['Q'][index])
                discharge_Q = np.asarray(cycling_dict['QV_discharge']['Q'][index])

                charge_len = len(charge_I)
                discharge_len = len(discharge_I)

                cycle_len = charge_len + discharge_len

                cycle_df = pd.DataFrame({
                    'cycle_number': np.full(cycle_len, index + 1, dtype=np.int32),
                    'I': np.concatenate([charge_I, discharge_I]),
                    'V': np.concatenate([charge_V, discharge_V]),
                    't': np.concatenate([charge_t, discharge_t]),
                    'Q_charge': np.concatenate([charge_Q, np.zeros(discharge_len)]),
                    'Q_discharge': np.concatenate([np.zeros(charge_len), discharge_Q]),
                })

                cycle_frames.append(cycle_df)

            df = pd.concat(cycle_frames, ignore_index=True)

            # ===================== clean =====================
            df = clean_cell_fast(df, cycling_dict, rpt_dict)

            # ===================== organize =====================
            name = 'ISU-ILCC_' + cell
            battery = organize_cell_fast(df, name)

            self.dump_single_file(battery)
            process_batteries_num += 1

            if not self.silent:
                tqdm.write(f'File: {battery.cell_id} dumped to pkl file')

        return process_batteries_num, skip_batteries_num


# ===================== optimized clean =====================

def clean_cell_fast(df, cycling_dict, rpt_dict):

    should_exclude = []
    i = 0
    cycle_start = cycling_dict['QV_discharge']['t']

    for cycle_number in range(len(cycle_start)):
        current_cycle_start = cycle_start[cycle_number][0]

        if i >= len(rpt_dict['start_stop_time']['start']):
            break

        rpt_start = rpt_dict['start_stop_time']['start'][i]

        if not rpt_start:
            continue
        elif rpt_start < current_cycle_start:
            should_exclude.append(cycle_number + 1)
            i += 1

    if should_exclude:
        df = df[~df['cycle_number'].isin(should_exclude)].copy()

    unique_cycles = sorted(df['cycle_number'].unique())
    mapping = {old: new for new, old in enumerate(unique_cycles, 1)}
    new_cycle = df['cycle_number'].map(mapping).to_numpy(dtype=np.int32)
    df.loc[:, 'cycle_number'] = new_cycle

    return df


# ===================== optimized organize =====================

def organize_cell_fast(df, name):

    # vectorized datetime conversion
    if np.issubdtype(df['t'].dtype, np.datetime64):
        df['t'] = df['t'].astype('datetime64[s]').astype(np.int64)

    df = df.sort_values('t')

    cycle_data = []

    for cycle_index, cdf in df.groupby('cycle_number'):

        cycle_data.append(CycleData(
            cycle_number=int(cycle_index),
            voltage_in_V=cdf['V'].tolist(),
            current_in_A=cdf['I'].tolist(),
            temperature_in_C=None,
            discharge_capacity_in_Ah=cdf['Q_discharge'].tolist(),
            charge_capacity_in_Ah=cdf['Q_charge'].tolist(),
            time_in_s=cdf['t'].tolist()
        ))

    charge_start_soc, discharge_end_soc = calculate_soc_start_and_end(df, name)

    rates = CYCLING_RATES[name[:-2]]

    charge_protocol = [CyclingProtocol(
        rate_in_C=float(rates[0]),
        start_soc=charge_start_soc[name],
        end_soc=1.0
    )]

    discharge_protocol = [CyclingProtocol(
        rate_in_C=float(rates[1]),
        start_soc=1.0,
        end_soc=discharge_end_soc[name]
    )]

    return BatteryData(
        cell_id=name,
        cycle_data=cycle_data,
        form_factor=' 502030-size Li-polymer',
        anode_material='graphite',
        cathode_material='NMC',
        discharge_protocol=discharge_protocol,
        charge_protocol=charge_protocol,
        nominal_capacity_in_Ah=0.25,
        min_voltage_limit_in_V=3.0,
        max_voltage_limit_in_V=4.2,
        SOC_interval=[charge_start_soc[name], 1]
    )


# ===================== unchanged helper functions =====================

def convert_cycling_to_dict(zip_path, cell, subfolder):
    with open(zip_path / f'Cycling_json/{subfolder}/{cell}.json', 'r') as file:
        data_dict = json.loads(json.load(file))

    for iii, start_time in enumerate(data_dict['start_stop_time']['start']):
        if start_time != '[]':
            data_dict['start_stop_time']['start'][iii] = np.datetime64(start_time)
            data_dict['start_stop_time']['stop'][iii] = np.datetime64(
                data_dict['start_stop_time']['stop'][iii])
        else:
            data_dict['start_stop_time']['start'][iii] = []
            data_dict['start_stop_time']['stop'][iii] = []

    for iii in range(len(data_dict['time_series_charge']['start'])):
        data_dict['QV_charge']['t'][iii] = list(map(np.datetime64, data_dict['QV_charge']['t'][iii]))
        data_dict['QV_discharge']['t'][iii] = list(map(np.datetime64, data_dict['QV_discharge']['t'][iii]))

    return data_dict


def convert_RPT_to_dict(zip_path, cell, subfolder):
    with open(zip_path / f'RPT_json/{subfolder}/{cell}.json', 'r') as file:
        data_dict = json.loads(json.load(file))

    for iii, start_time in enumerate(data_dict['start_stop_time']['start']):
        if start_time != '[]':
            data_dict['start_stop_time']['start'][iii] = np.datetime64(start_time)
            data_dict['start_stop_time']['stop'][iii] = np.datetime64(
                data_dict['start_stop_time']['stop'][iii])
        else:
            data_dict['start_stop_time']['start'][iii] = []
            data_dict['start_stop_time']['stop'][iii] = []

    return data_dict


def calculate_soc_start_and_end(df, name, nominal_capacity=0.25):
    charge_start_soc, discharge_end_soc = {}, {}

    charge_capacity = df.loc[df['cycle_number'] == 1, 'Q_charge'].max()
    soc_charge_interval = charge_capacity / nominal_capacity
    if soc_charge_interval > 1:
        soc_charge_interval = 1
    charge_start_soc[name] = 1 - soc_charge_interval

    discharge_capacity = df.loc[df['cycle_number'] == 1, 'Q_discharge'].max()
    soc_discharge_interval = discharge_capacity / nominal_capacity
    if soc_discharge_interval > 1:
        soc_discharge_interval = 0
    discharge_end_soc[name] = 1 - soc_discharge_interval

    if charge_start_soc[name] == 1 and discharge_end_soc[name] == 1:
        charge_start_soc[name] = 0
        discharge_end_soc[name] = 1
    return charge_start_soc, discharge_end_soc

CYCLING_RATES = {
    'ISU-ILCC_G1': [0.5, 0.5],
    'ISU-ILCC_G2': [0.5, 0.5],
    'ISU-ILCC_G3': [0.5, 0.5],
    'ISU-ILCC_G4': [1, 0.5],
    'ISU-ILCC_G5': [1, 0.5],
    'ISU-ILCC_G6': [2, 0.5],
    'ISU-ILCC_G7': [2, 0.5],
    'ISU-ILCC_G8': [2, 0.5],
    'ISU-ILCC_G9': [2, 0.5],
    'ISU-ILCC_G10': [2.5, 0.5],
    'ISU-ILCC_G12': [3, 0.5],
    'ISU-ILCC_G13': [3, 0.5],
    'ISU-ILCC_G14': [3, 0.5],
    'ISU-ILCC_G15': [3, 0.5],
    'ISU-ILCC_G16': [0.5, 0.5],
    'ISU-ILCC_G17': [1, 0.5],
    'ISU-ILCC_G18': [2.5, 0.5],
    'ISU-ILCC_G19': [2.5, 0.5],
    'ISU-ILCC_G20': [0.8, 0.5],
    'ISU-ILCC_G21': [1.2, 0.5],
    'ISU-ILCC_G22': [1.4, 0.5],
    'ISU-ILCC_G23': [1.6, 0.5],
    'ISU-ILCC_G24': [1.8, 0.5],
    'ISU-ILCC_G25': [1.8, 0.6],
    'ISU-ILCC_G26': [1.4, 2.2],
    'ISU-ILCC_G27': [0.6, 2.4],
    'ISU-ILCC_G28': [2.4, 1.6],
    'ISU-ILCC_G29': [1.6, 1.8],
    'ISU-ILCC_G30': [0.8, 0.8],
    'ISU-ILCC_G31': [1.2, 1],
    'ISU-ILCC_G32': [1, 1.4],
    'ISU-ILCC_G33': [2, 1.2],
    'ISU-ILCC_G34': [2.2, 2],
    'ISU-ILCC_G35': [1.825, 0.5],
    'ISU-ILCC_G36': [2.075, 0.5],
    'ISU-ILCC_G37': [0.725, 0.5],
    'ISU-ILCC_G38': [1.875, 0.5],
    'ISU-ILCC_G39': [1.475, 0.5],
    'ISU-ILCC_G40': [1.825, 1.025],
    'ISU-ILCC_G41': [2.075, 1.775],
    'ISU-ILCC_G42': [0.725, 2.375],
    'ISU-ILCC_G43': [1.875, 2.325],
    'ISU-ILCC_G44': [0.775, 1.275],
    'ISU-ILCC_G45': [1.125, 1.725],
    'ISU-ILCC_G46': [1.225, 2.025],
    'ISU-ILCC_G47': [2.325, 1.925],
    'ISU-ILCC_G48': [2.375, 2.225],
    'ISU-ILCC_G49': [0.975, 0.675],
    'ISU-ILCC_G50': [2.425, 1.625],
    'ISU-ILCC_G51': [2.275, 1.875],
    'ISU-ILCC_G52': [1.425, 0.875],
    'ISU-ILCC_G53': [2.025, 0.825],
    'ISU-ILCC_G54': [0.925, 1.125],
    'ISU-ILCC_G55': [1.025, 2.475],
    'ISU-ILCC_G56': [2.175, 0.975],
    'ISU-ILCC_G57': [1.775, 1.175],
    'ISU-ILCC_G58': [2.475, 0.575],
    'ISU-ILCC_G59': [1.325, 1.825],
    'ISU-ILCC_G60': [0.675, 1.325],
    'ISU-ILCC_G61': [2.125, 1.975],
    'ISU-ILCC_G62': [1.575, 2.425],
    'ISU-ILCC_G63': [1.975, 1.675],
    'ISU-ILCC_G64': [1.175, 1.425],
}