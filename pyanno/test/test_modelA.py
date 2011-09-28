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
        model = ModelA.random_model(nclasses)
        incorrect = np.ones((nitems, 3), dtype=bool)
        agreement = model._generate_agreement(incorrect)
        frequency = np.bincount(agreement, minlength=5) / float(nitems)
        expected = model._compute_alpha()[3:]
        testing.assert_allclose(frequency[:-1], expected, atol=1e-1, rtol=0)


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
        nitems = 2000*8
        nclasses = 3

        # create random model
        model = ModelA.random_model(nclasses)
        # create random data
        annotations = model.generate_annotations(nitems)

        self.assertEqual(annotations.shape, (nitems, model.nannotators))
        self.assertTrue(np.all((annotations!=-1).sum(1) == 3))

        freqs = (np.array([(annotations==psi).sum() / float(nitems*3)
                           for psi in range(nclasses)]))
        testing.assert_allclose(model.omega, freqs, atol=1e-1, rtol=0.)


    def test_ml_estimation_theta_only(self):
        # test simple model, check that we get to global optimum
        nclasses, nitems = 3, 1000*8
        # create random model and data (this is our ground truth model)
        theta = np.array([0.5, 0.9, 0.6, 0.65, 0.87, 0.2, 0.9, 0.78])
        true_model = ModelA.random_model(nclasses, theta=theta)
        annotations = true_model.generate_annotations(nitems)

        # create a new, empty model and infer back the parameters
        model = ModelA.random_model(nclasses, omega=true_model.omega)
        #before_llhood = model.log_likelihood(annotations)
        model.mle(annotations, estimate_alpha=False, estimate_omega=False)
        #after_llhood = model.log_likelihood(annotations)

        #self.assertGreater(after_llhood, before_llhood)
        testing.assert_allclose(model.theta, true_model.theta, atol=1e-1, rtol=0.)


    def test_ml_estimation(self):
        pass
        #testing.assert_allclose(model.alpha, true_model.alpha, atol=1e-1, rtol=0.)



if __name__ == '__main__':
    unittest.main()
