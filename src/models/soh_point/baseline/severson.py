"""
soh_point/severson.py — Severson ElasticNet baseline for SOH point estimation.
Reference: Severson et al., Nature Energy 2019.
This is a classical ML model; it has no forward() method.
Training is handled exclusively by src/train/soh_point/train_severson.py.
"""


class Severson:
    """Marker class — signals registry to use train_severson instead of train_base."""
    pass
