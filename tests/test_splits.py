from src.splits import make_battery_splits


class DummyDataset:
    def __init__(self, active=range(20)):
        self.active = tuple(active)

    @property
    def n_batteries(self):
        return len(self.active)

    def get_battery_meta(self):
        return [{'cathode_material': 'LFP' if index % 2 else 'NMC'}
                for index in self.active]

    def subset_by_battery(self, indices):
        return DummyDataset(indices)


def test_standard_domains_have_one_reproducible_fixed_split():
    config = {'data': {'split_strategy': 'random', 'val_ratio': 0.1, 'test_ratio': 0.2}}
    first = make_battery_splits(DummyDataset(), config, seed=1)
    second = make_battery_splits(DummyDataset(), config, seed=1)

    assert len(first) == 1
    assert first[0]['train'].active == second[0]['train'].active
    assert first[0]['val'].active == second[0]['val'].active
    assert first[0]['test'].active == second[0]['test'].active
    assert len(first[0]['train'].active) == 14
    assert len(first[0]['val'].active) == 2
    assert len(first[0]['test'].active) == 4
