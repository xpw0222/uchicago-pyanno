from __future__ import division

import unittest
import numpy as np
import pyanno

from pyanno.measures import (confusion_matrix, chance_agreement_same_frequency,
                             observed_agreement_frequency,
                             _fleiss_kappa_nannotations, fleiss_kappa)


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


    def test_chance_agreement_same_frequency(self):
        distr = np.array([0.1, 0.5, 0.4])
        anno1 = pyanno.util.random_categorical(distr, nsamples=10000)
        anno2 = pyanno.util.random_categorical(distr, nsamples=10000)

        expected = distr ** 2.
        freqs = chance_agreement_same_frequency(anno1, anno2, len(distr))

        np.testing.assert_allclose(freqs, expected, atol=1e-2, rtol=0.)


    def test_observed_agreement(self):
        anno1 = np.array([0, 0, 1, 1, -1, 3])
        anno2 = np.array([0, -1, 1, 1, -1, 2])
        nvalid = np.sum((anno1!=-1) & (anno2!=-1))

        expected = np.array([1., 2., 0., 0.]) / nvalid
        freqs = observed_agreement_frequency(anno1, anno2, 4)

        np.testing.assert_array_equal(freqs, expected)


    def test_fleiss_kappa_nannotations(self):
        # same example as
        # http://en.wikipedia.org/wiki/Fleiss%27_kappa#Worked_example
        nannotations = np.array(
            [
                [0, 0, 0, 0, 14],
                [0, 2, 6, 4, 2],
                [0, 0, 3, 5, 6],
                [0, 3, 9, 2, 0],
                [2, 2, 8, 1, 1],
                [7, 7, 0, 0, 0],
                [3, 2, 6, 3, 0],
                [2, 5, 3, 2, 2],
                [6, 5, 2, 1, 0],
                [0, 2, 2, 3, 7]
            ]
        )
        expected = 0.21
        kappa = _fleiss_kappa_nannotations(nannotations)
        self.assertAlmostEqual(kappa, expected, 2)


    def test_fleiss_kappa(self):
        nitems = 100
        nannotators = 5
        nclasses = 4

        # perfect agreement, 2 missing annotations per row
        annotations = np.empty((nitems, nannotators), dtype=int)
        for i in xrange(nitems):
            annotations[i,:] = np.random.randint(nclasses)
            perm = np.random.permutation(nclasses)
            annotations[i,perm[0:2]] = -1

        self.assertEqual(fleiss_kappa(annotations, nclasses), 1.0)

        # unequal number of annotators per row
        annotations[0,:] = -1
        self.assertRaises(ValueError, fleiss_kappa, annotations)