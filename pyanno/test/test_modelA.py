import unittest
import numpy as np
from numpy import testing
from pyanno.modelA import ModelA


class TestModelA(unittest.TestCase):

    def test_generate_incorrectness(self):
        nitems = 50000
        nclasses = 3
        theta = np.array([0.3, 0.6, 0.7])
        model = ModelA.random_model(nclasses)
        incorrect = model._generate_incorrectness(nitems, theta)
        correct_freq = 1. - incorrect.sum(0)/float(nitems)
        testing.assert_allclose(correct_freq, theta, atol=1e-2, rtol=0)


    def test_generate_agreement(self):
        nclasses = 4

        # when all correct, only index 0 (aaa) is possible
        nitems = 100
        model = ModelA.random_model(nclasses)
        incorrect = np.zeros((nitems, 3), dtype=bool)
        agreement = model._generate_agreement(incorrect)
        self.assertTrue(agreement.shape, (nitems,))
        self.assertTrue(np.all(agreement == 0))

        # all incorrect, check frequency corresponds to alpha[3:]
        nitems = 10000
        alpha = np.array([0., 0., 0.,
                          0.2, 0.1, 0.4, 0.12])
        expected = np.r_[alpha[3:], 1.-alpha.sum()]
        model = ModelA.random_model(nclasses, alpha=alpha)
        incorrect = np.ones((nitems, 3), dtype=bool)
        agreement = model._generate_agreement(incorrect)
        frequency = np.bincount(agreement, minlength=5) / float(nitems)
        testing.assert_allclose(frequency, expected, atol=1e-2, rtol=0)


    def test_generate_triplet_annotation(self):
        nitems = 100
        nclasses = 4
        omega = np.array([0.22, 0.38, 0.3, 0.1])
        model = ModelA.random_model(nclasses, omega=omega)

        # check that one gets the expected number of unique items
        # for each agreement pattern
        theta = np.array([0.3, 0.6, 0.7])
        incorrect = model._generate_incorrectness(nitems, theta)
        agreement = model._generate_agreement(incorrect)
        annotations = model._generate_annotations(agreement)

        # map of agreement index to number of different items
        agreement_to_number = {0: 1, 1: 2, 2: 2, 3: 2, 4: 3}
        for i in xrange(nitems):
            self.assertEqual(len(set(annotations[i,:])),
                             agreement_to_number[agreement[i]])

        # always agreeing: frequency should match the omegas^3 (Table 5)
        nitems = 50000
        agreement = np.empty((nitems,), dtype=int)
        agreement.fill(0)  # aaa agreement pattern
        annotations = model._generate_annotations(agreement).flatten()
        frequencies = (np.bincount(annotations, minlength=nclasses)
                       / float(nitems*3))
        expected = model.omega**3. / (model.omega**3.).sum()
        testing.assert_allclose(frequencies, expected, atol=1e-2, rtol=0)


    def test_generate_annotations(self):
        nitems = 50*8
        nclasses = 3

        # create random model
        model = ModelA.random_model(nclasses)
        # create random data
        annotations = model.generate_annotations(nitems)

        self.assertEqual(annotations.shape, (nitems, model.nannotators))
        self.assertTrue(np.all((annotations!=-1).sum(1) == 3))


if __name__ == '__main__':
    unittest.main()
