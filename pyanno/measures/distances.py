# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

"""Definition of distance measures between classes."""

import numpy as np


def diagonal_distance(i, j):
    """Weight function returning `|i-j|`.
    """
    return abs(i-j)


def binary_distance(i, j):
    """Binary weight function returning 0 if i==j, else 1.
    """
    return np.asarray(i!=j, dtype=float)
