"""
Module 4: Fuzzy Logic Inference Systems (Mamdani & Sugeno)
"""

from .membership import TriangularMF, TrapezoidalMF, GaussianMF
from .mamdani_fis import MamdaniFIS, LinguisticVariable, FuzzyRule
from .sugeno_fis import SugenoFIS, SugenoRule

__all__ = [
    "TriangularMF",
    "TrapezoidalMF",
    "GaussianMF",
    "MamdaniFIS",
    "LinguisticVariable",
    "FuzzyRule",
    "SugenoFIS",
    "SugenoRule",
]
