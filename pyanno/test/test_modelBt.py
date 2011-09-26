import unittest
import numpy as np
from numpy import testing
from pyanno.modelBt import ModelBt


class TestModelBt(unittest.TestCase):

    def test_mle_estimation(self):
        # test simple model, check that we get to global optimum
        nclasses, nitems = 3, 500*8
        # create random model and data (this is our ground truth model)
        true_model = ModelBt.random_model(nclasses, nitems)
        labels = true_model.generate_labels()
        annotations = true_model.generate_annotations(labels)

        # create a new, empty model and infer back the parameters
        model = ModelBt.random_model(nclasses, nitems)
        before_llhood = model.log_likelihood(annotations)
        model.mle(annotations)
        after_llhood = model.log_likelihood(annotations)

        self.assertGreater(after_llhood, before_llhood)
        testing.assert_allclose(model.gamma, true_model.gamma, atol=1e-1, rtol=0.)
        testing.assert_allclose(model.theta, true_model.theta, atol=1e-1, rtol=0.)


    def test_log_likelihood(self):
        # check that log likelihood is maximal at true parameters
        nclasses, nitems = 3, 1500*8
        # create random model and data (this is our ground truth model)
        true_model = ModelBt.random_model(nclasses, nitems, use_priors=False)
        labels = true_model.generate_labels()
        annotations = true_model.generate_annotations(labels)

        max_llhood = true_model.log_likelihood(annotations)
        # perturb gamma
        for _ in xrange(20):
            theta = true_model.theta
            gamma = np.random.normal(loc=true_model.gamma, scale=0.1)
            gamma = np.clip(gamma, 0., 1.)
            gamma /= gamma.sum()
            model = ModelBt(nclasses, nitems, gamma, theta)
            llhood = model.log_likelihood(annotations)
            self.assertGreater(max_llhood, llhood)

        # perturb theta
        for _ in xrange(20):
            gamma = true_model.gamma
            theta = np.random.normal(loc=true_model.theta, scale=0.1)
            theta = np.clip(theta, 0., 1.)
            model = ModelBt(nclasses, nitems, gamma, theta)
            llhood = model.log_likelihood(annotations)
            self.assertGreater(max_llhood, llhood)


    def test_sampling_theta(self):
        nclasses, nitems = 3, 500*8
        nsamples = 1000

        # create random model (this is our ground truth model)
        true_model = ModelBt.random_model(nclasses, nitems)
        # create random data
        labels = true_model.generate_labels()
        annotations = true_model.generate_annotations(labels)

        # create a new model
        # do not start from gamma_i = 1/nclasses: it is a symmetry point
        model = ModelBt.random_model(nclasses, nitems)
        # get optimal parameters (to make sure we're at the optimum
        model.mle(annotations)

        print 'MLE gamma'
        print model.gamma
        print 'MLE theta'
        print model.theta

        def wrap_lhood(x, arguments):
            gamma, theta = model._vector_to_params(x)
            counts, dim, _ = arguments
            return model._log_likelihood_counts(counts)

        # modify parameters, to give false start to sampler
        real_theta = model.theta.copy()
        model.theta = model._random_theta(model.nannotators)
        # save current parameters
        gamma_before, theta_before = model.gamma.copy(), model.theta.copy()
        samples = model.sample_posterior_over_theta(annotations, nsamples)
        # test_ the mean of the sampled parameters is the same as the MLE one
        # (up to 3 standard deviations of the estimate sample distribution)
        testing.assert_array_less(np.absolute(samples.mean(0)-real_theta),
                                  3.*samples.std(0))

        # check that original parameters are intact
        testing.assert_equal(model.gamma, gamma_before)
        testing.assert_equal(model.theta, theta_before)


if __name__ == '__main__':
    unittest.main()
