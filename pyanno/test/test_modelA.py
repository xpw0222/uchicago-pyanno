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
        theta = np.array([0.5, 0.9, 0.6, 0.65, 0.87, 0.54, 0.9, 0.78])
        true_model = ModelA.random_model(nclasses, theta=theta)
        annotations = true_model.generate_annotations(nitems)

        # create a new, empty model and infer back the parameters
        model = ModelA.random_model(nclasses, omega=true_model.omega)
        before_llhood = model.log_likelihood(annotations)
        model.mle(annotations, estimate_alpha=False, estimate_omega=False)
        after_llhood = model.log_likelihood(annotations)

        testing.assert_allclose(model.theta, true_model.theta, atol=1e-1, rtol=0.)
        self.assertGreater(after_llhood, before_llhood)


    def test_ml_estimation(self):
        pass
        #testing.assert_allclose(model.alpha, true_model.alpha, atol=1e-1, rtol=0.)


    def test_log_likelihood(self):
        # check that log likelihood is maximal at true parameters
        nclasses, nitems = 3, 1500*8
        # create random model and data (this is our ground truth model)
        true_model = ModelA.random_model(nclasses)
        annotations = true_model.generate_annotations(nitems)

        max_llhood = true_model.log_likelihood(annotations)
        # perturb omega
        for _ in xrange(20):
            theta = true_model.theta
            omega = np.random.normal(loc=true_model.omega, scale=0.1)
            omega = np.clip(omega, 0.001, 0.999)
            omega /= omega.sum()
            model = ModelA(nclasses, omega=omega, theta=theta)
            llhood = model.log_likelihood(annotations)
            self.assertGreater(max_llhood, llhood)

        # perturb theta
        for _ in xrange(20):
            omega = true_model.omega
            theta = np.random.normal(loc=true_model.theta, scale=0.1)
            theta = np.clip(theta, 0., 1.)
            model = ModelA(nclasses, omega=omega, theta=theta)
            llhood = model.log_likelihood(annotations)
            self.assertGreater(max_llhood, llhood)


    def test_sampling_theta(self):
        nclasses, nitems = 3, 500*8
        nsamples = 1000

        # create random model (this is our ground truth model)
        true_model = ModelA.random_model(nclasses)
        # create random data
        annotations = true_model.generate_annotations(nitems)

        # create a new model
        model = ModelA.random_model(nclasses)
        # get optimal parameters (to make sure we're at the optimum)
        model.mle(annotations)

        # modify parameters, to give false start to sampler
        real_theta = model.theta.copy()
        model.theta = model._random_theta(model.nannotators)
        # save current parameters
        omega_before, theta_before = model.omega.copy(), model.theta.copy()
        samples = model.sample_posterior_over_theta(annotations, nsamples)
        # test: the mean of the sampled parameters is the same as the MLE one
        # (up to 3 standard deviations of the estimate sample distribution)
        testing.assert_array_less(np.absolute(samples.mean(0)-real_theta),
                                  3.*samples.std(0))

        # check that original parameters are intact
        testing.assert_equal(model.omega, omega_before)
        testing.assert_equal(model.theta, theta_before)


    def test_inference(self):
        # annotators agreeing, check that inferred correctness is either
        # CCC or III
        nclasses, nitems = 4, 50*8

        # create random model (this is our ground truth model)
        omega = np.ones((nclasses,)) / float(nclasses)
        theta = np.ones((8,)) * 0.9999
        true_model = ModelA(nclasses, theta, omega)
        # create random data
        annotations = true_model.generate_annotations(nitems)

        posterior = true_model.infer_labels(annotations)
        testing.assert_allclose(posterior.sum(1), 1., atol=1e-6, rtol=0.)
        inferred = posterior.argmax(1)
        expected = annotations.max(1)

        testing.assert_equal(inferred, expected)
        self.assertTrue(np.all(posterior[np.arange(nitems),inferred] > 0.999))

        # at chance, disagreeing annotators: most accurate wins
        omega = np.ones((nclasses,)) / float(nclasses)
        theta = np.ones((8,))
        theta[1:4] = np.array([0.9, 0.6, 0.5])
        model = ModelA(nclasses, theta, omega)

        data = np.array([[-1, 0, 1, 2, -1, -1, -1, -1,]])
        posterior = model.infer_labels(data)
        posterior = posterior[0]
        self.assertTrue(posterior[0] > posterior[1]
                        > posterior[2] > posterior[3])


if __name__ == '__main__':
    unittest.main()
