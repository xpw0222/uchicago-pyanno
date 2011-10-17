from __future__ import division

import unittest
import numpy as np
import pyanno

from pyanno.measures import confusion_matrix, chance_agreement_frequency, observed_agreement_frequency


class TestMeasures(unittest.TestCase):
    def test_confusion_matrix(self):
        anno1 = np.array([0, 0, 1, 1, 2, 3])
        anno2 = np.array([0, 1, 1, 1, 2, 2])
        expected = np.array(
            [
                [1, 1, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 1, 0],
                [0, 0, 1, 0]
            ])
        cm = confusion_matrix(anno1, anno2, 4)
        np.testing.assert_array_equal(cm, expected)


    def test_confusion_matrix_missing(self):
        """Test confusion matrix with missing data."""
        anno1 = np.array([0, 0, 1, 1, -1, 3])
        anno2 = np.array([0, -1, 1, 1, 2, 2])
        expected = np.array(
            [
                [1, 0, 0, 0],
                [0, 2, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 1, 0]
            ])
        cm = confusion_matrix(anno1, anno2, 4)
        np.testing.assert_array_equal(cm, expected)


    def test_chance_agreement_frequency(self):
        distr = np.array([0.1, 0.5, 0.4])
        anno1 = pyanno.util.random_categorical(distr, nsamples=10000)
        anno2 = pyanno.util.random_categorical(distr, nsamples=10000)

        expected = distr ** 2.
        freqs = chance_agreement_frequency(anno1, anno2, len(distr))

        np.testing.assert_allclose(freqs, expected, atol=1e-2, rtol=0.)


    def test_observed_agreement(self):
        anno1 = np.array([0, 0, 1, 1, -1, 3])
        anno2 = np.array([0, -1, 1, 1, -1, 2])
        nvalid = np.sum((anno1!=-1) & (anno2!=-1))

        expected = np.array([1., 2., 0., 0.]) / nvalid
        freqs = observed_agreement_frequency(anno1, anno2, 4)

        np.testing.assert_array_equal(freqs, expected)

