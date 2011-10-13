import unittest
import numpy as np
from numpy import testing
from pyanno.modelBt import ModelBt


class TestModelBt(unittest.TestCase):

    def test_mle_estimation(self):
        # test simple model, check that we get to global optimum
        nclasses, nitems = 3, 500*8
        # create random model and data (this is our ground truth model)
        true_model = ModelBt.create_initial_state(nclasses)
        labels = true_model.generate_labels(nitems)
        annotations = true_model.generate_annotations(labels)

        # create a new, empty model and infer back the parameters
        model = ModelBt.create_initial_state(nclasses)
        before_llhood = model.log_likelihood(annotations, use_prior=True)
        model.mle(annotations, use_prior=True)
        after_llhood = model.log_likelihood(annotations, use_prior=True)

        testing.assert_allclose(model.gamma, true_model.gamma, atol=1e-1, rtol=0.)
        testing.assert_allclose(model.theta, true_model.theta, atol=1e-1, rtol=0.)
        self.assertGreater(after_llhood, before_llhood)


    def test_log_likelihood(self):
        # check that log likelihood is maximal at true parameters
        nclasses, nitems = 3, 1500*8
        # create random model and data (this is our ground truth model)
        true_model = ModelBt.create_initial_state(nclasses)
        labels = true_model.generate_labels(nitems)
        annotations = true_model.generate_annotations(labels)

        max_llhood = true_model.log_likelihood(annotations)
        # perturb gamma
        for _ in xrange(20):
            theta = true_model.theta
            gamma = np.random.normal(loc=true_model.gamma, scale=0.1)
            gamma = np.clip(gamma, 0., 1.)
            gamma /= gamma.sum()
            model = ModelBt(nclasses, gamma, theta)
            llhood = model.log_likelihood(annotations)
            self.assertGreater(max_llhood, llhood)

        # perturb theta
        for _ in xrange(20):
            gamma = true_model.gamma
            theta = np.random.normal(loc=true_model.theta, scale=0.1)
            theta = np.clip(theta, 0., 1.)
            model = ModelBt(nclasses, gamma, theta)
            llhood = model.log_likelihood(annotations)
            self.assertGreater(max_llhood, llhood)


    def test_sampling_theta(self):
        nclasses, nitems = 3, 500*8
        nsamples = 1000

        # create random model (this is our ground truth model)
        true_model = ModelBt.create_initial_state(nclasses)
        # create random data
        labels = true_model.generate_labels(nitems)
        annotations = true_model.generate_annotations(labels)

        # create a new model
        model = ModelBt.create_initial_state(nclasses)
        # get optimal parameters (to make sure we're at the optimum)
        model.mle(annotations, use_prior=False)

        # modify parameters, to give false start to sampler
        real_theta = model.theta.copy()
        model.theta = model._random_theta(model.nannotators)
        # save current parameters
        gamma_before, theta_before = model.gamma.copy(), model.theta.copy()
        samples = model.sample_posterior_over_theta(annotations, nsamples,
                                                    use_prior=False)
        # test: the mean of the sampled parameters is the same as the MLE one
        # (up to 3 standard deviations of the estimate sample distribution)
        testing.assert_array_less(np.absolute(samples.mean(0)-real_theta),
                                  3.*samples.std(0))

        # check that original parameters are intact
        testing.assert_equal(model.gamma, gamma_before)
        testing.assert_equal(model.theta, theta_before)


    def test_inference(self):
        # perfect annotation, check that inferred label is correct
        nclasses, nitems = 3, 50*8

        # create random model (this is our ground truth model)
        gamma = np.ones((nclasses,)) / float(nclasses)
        theta = np.ones((8,)) * 0.999
        true_model = ModelBt(nclasses, gamma, theta)
        # create random data
        labels = true_model.generate_labels(nitems)
        annotations = true_model.generate_annotations(labels)

        posterior = true_model.infer_labels(annotations)
        testing.assert_allclose(posterior.sum(1), 1., atol=1e-6, rtol=0.)
        inferred = posterior.argmax(1)

        testing.assert_equal(inferred, labels)
        self.assertTrue(np.all(posterior[np.arange(nitems),inferred] > 0.999))

        # at chance annotation, disagreeing annotators: get back prior
        gamma = ModelBt._random_gamma(nclasses)
        theta = np.ones((8,)) / float(nclasses)
        model = ModelBt(nclasses, gamma, theta)

        data = np.array([[-1, 0, 1, 2, -1, -1, -1, -1,]])
        testing.assert_almost_equal(np.squeeze(model.infer_labels(data)),
                                    model.gamma, 6)


    def test_generate_annotations(self):
        # test to check that annotations are masked correctly when the number
        # of items is not divisible by the number of annotators
        nclasses, nitems = 5, 8*30+3

        model = ModelBt.create_initial_state(nclasses)
        labels = model.generate_labels(nitems)
        annotations = model.generate_annotations(labels)

        valid = annotations != -1
        # check that on every row there are exactly 3 annotations
        self.assertTrue(np.all(valid.sum(1) == 3))


if __name__ == '__main__':
    unittest.main()
