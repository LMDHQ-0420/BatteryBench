import pickle

import numpy as np

import src.data.cycle_dataset as datasets
from src.data.cycle_dataset import RULDataset, SOHPointDataset, SOHTrajDataset
from src.utils import get_charge_curve


def _cycle(soh):
    return {
        'voltage_in_V': [3.0, 3.3, 3.6, 4.0, 4.2, 4.0, 3.4],
        'current_in_A': [1.0, 1.0, 1.0, 1.0, 0.5, -1.0, -1.0],
        'charge_capacity_in_Ah': [0.1, 0.3, 0.5, 0.8, 1.0, 1.0, 1.0],
        'discharge_capacity_in_Ah': [0.0, 0.0, 0.0, 0.0, 0.0, soh / 2, soh],
    }


def _cell():
    return {
        'cell_id': 'synthetic',
        'nominal_capacity_in_Ah': 1.0,
        'SOC_interval': [0.0, 1.0],
        'cycle_data': [_cycle(1.0), _cycle(0.9), _cycle(0.7)],
    }


def test_charge_curve_excludes_discharge_and_capacity_on_soh_point():
    cycle = _cycle(1.0)
    point = get_charge_curve(cycle, 1.0, curve_length=400, include_capacity=False)
    history = get_charge_curve(cycle, 1.0, curve_length=400, include_capacity=True)

    assert point.shape == (2, 400)
    assert history.shape == (3, 400)
    assert np.isclose(point[0].max(), 1.0)
    assert np.all(point[1] > 0)
    assert np.all(np.diff(history[2]) >= -1e-6)
    assert np.isclose(history[2, -1], 1.0)


def test_datasets_expose_only_public_charge_curves_and_overwrite_old_cache(tmp_path, monkeypatch):
    source = tmp_path / 'synthetic.pkl'
    with source.open('wb') as file:
        pickle.dump(_cell(), file)

    cache_root = tmp_path / 'cache'
    monkeypatch.setattr(datasets, 'CACHE_DIR', cache_root)
    stale = cache_root / 'soh_point' / tmp_path.name / 'synthetic.npz'
    stale.parent.mkdir(parents=True)
    np.savez_compressed(stale, Q=np.ones((3, 200)), curves=np.ones((3, 3, 300)))

    point = SOHPointDataset('', pkl_files=[str(source)])
    rul = RULDataset('', pkl_files=[str(source)], early_cycle=2)
    trajectory = SOHTrajDataset('', pkl_files=[str(source)], early_cycle=2)

    point_item = point[0]
    rul_item = rul[0]
    trajectory_item = trajectory[0]
    assert 'Q' not in point_item
    assert 'Q' not in rul_item
    assert 'Q' not in trajectory_item
    assert point_item['cycle_curve_data'].shape == (1, 2, 400)
    assert rul_item['cycle_curve_data'].shape == (2, 3, 400)
    assert trajectory_item['cycle_curve_data'].shape == (2, 3, 400)

    with np.load(stale) as cache:
        assert 'Q' not in cache.files
        assert cache['curves'].shape == (3, 2, 400)
