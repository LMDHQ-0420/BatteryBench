import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "src" / "preprocess" / "preprocess_CALB.py"


def load_module():
    spec = importlib.util.spec_from_file_location("preprocess_CALB", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_continuous_charge_capacity_joins_step_resets(monkeypatch):
    monkeypatch.syspath_prepend(str(MODULE_PATH.parents[1]))
    monkeypatch.syspath_prepend(str(MODULE_PATH.parent))
    function = load_module().continuous_charge_capacity

    capacity = [0.0, 0.2, 0.5, 0.1, 0.4, 0.0]
    current = [0.0, 1.0, 1.0, 0.8, 0.8, -1.0]

    np.testing.assert_allclose(
        function(capacity, current, threshold=0.01),
        [0.0, 0.2, 0.5, 0.6, 0.9, 0.0],
    )
