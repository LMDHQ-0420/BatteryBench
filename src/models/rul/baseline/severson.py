"""
rul/severson.py — Severson ElasticNet baseline for RUL prediction.
Reference: Severson et al., Nature Energy 2019.
This is a classical ML model; it has no forward() method.
Training is handled exclusively by src/train/rul/train_severson.py.
"""


class Severson:
    """Marker class — signals registry to use train_severson instead of train_base."""
    pass
