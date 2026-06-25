"""
soh_traj/severson.py — Severson ElasticNet baseline for SOH trajectory prediction.
Reference: Severson et al., Nature Energy 2019.
This is a classical ML model; it has no forward() method.
Training is handled exclusively by src/train/soh_traj/train_severson.py.
Proxy target: mean(soh_traj) used as scalar for ElasticNet.
"""


class Severson:
    """Marker class — signals registry to use train_severson instead of train_base."""
    pass
