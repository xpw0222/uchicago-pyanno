import unittest
import numpy as np
from pyanno.models import ModelB

def assert_is_distributions(distr, axis=0):
    """Check that input array represents a collection of distributions.
    """
    integral = distr.sum(axis=axis)
    np.testing.assert_allclose(integral,
                               np.ones_like(integral), rtol=0., atol=1e-7)


def assert_is_dirichlet(samples, alpha):
    """Checks that samples 'samples' are drawn from Dirichlet(alpha)."""
    assert len(samples.shape) == 2
    alpha0 = alpha.sum(-1)
    expected_mean = alpha / alpha0
    expected_var = expected_mean * (1. - expected_mean) / (alpha0 + 1.)
    np.testing.assert_allclose(samples.mean(0), expected_mean, rtol=0.1)
    np.testing.assert_allclose(samples.var(0),
                               expected_var, rtol=0.2)


class TestModelB(unittest.TestCase):
    def test_random_model(self):
        nclasses = 4
        nannotators = 6
        nitems = 8

        # check size of parameters
        model = ModelB.random_model(nclasses, nannotators, nitems)
        self.assertEqual(model.pi.shape, (nclasses,))
        assert_is_distributions(model.pi)
        self.assertEqual(model.theta.shape, (nannotators, nclasses, nclasses))
        assert_is_distributions(model.theta, axis=2)

        # check mean and variance of distribution
        beta = np.array([10., 2., 30., 5.])
        alpha = np.random.randint(1, 30, size=(nclasses, nclasses)).astype(float)
        # collect random parameters
        nsamples = 1000
        pis = np.zeros((nsamples, nclasses))
        thetas = np.zeros((nsamples, nannotators, nclasses, nclasses))
        for n in xrange(nsamples):
            model = ModelB.random_model(nclasses, nannotators, nitems,
                                        alpha, beta)
            pis[n,:] = model.pi
            thetas[n,...] = model.theta
        assert_is_dirichlet(pis, beta)
        for j in xrange(nannotators):
            for k in xrange(nclasses):
                assert_is_dirichlet(thetas[:,j,k,:], alpha[k,:])


    def test_generate_samples(self):
        nclasses = 4
        nannotators = 6
        nitems = 8
        model = ModelB.random_model(nclasses, nannotators, nitems)

        nsamples = 1000
        labels = np.empty((nsamples, nitems), dtype=int)
        for i in xrange(nsamples):
            labels[i] = model.generate_labels()

        # NOTE here we make use of the fact that the prior is the same for all
        # items
        freq = np.bincount(labels.flat) / float(np.prod(labels.shape))
        np.testing.assert_almost_equal(freq, model.pi, 2)

    def test_generate_annotations(self):
        pass

if __name__ == '__main__':
    unittest.main()