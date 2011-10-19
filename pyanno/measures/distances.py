"""Definition of distance measures between classes."""

import numpy as np


def diagonal_distance(i, j):
    """Return |i-j|."""
    return abs(i-j)


def binary_distance(i, j):
    """Return 0 if i==j, else 1."""
    return np.asarray(i!=j, dtype=float)
