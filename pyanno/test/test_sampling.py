import unittest
import numpy as np
from numpy import testing
import scipy.stats
from pyanno.modelAB import sample_distribution, optimum_jump
from pyanno.util import log_beta_pdf

class TestSampling(unittest.TestCase):
    def test_samples_distribution(self):
        # check sample_distribution's ability to sample from a beta distribution
        nclasses = 8
        nitems = 1000
        a = np.random.uniform(1., 5., size=(nclasses,))
        b = np.random.uniform(4., 6., size=(nclasses,))

        def beta_likelihood(params, values):
            a = params[:nclasses].copy()
            b = params[nclasses:].copy()
            llhood = 0.
            for k in range(nclasses):
                llhood += log_beta_pdf(values[:,k], a[k], b[k]).sum()
            return -llhood

        values = np.empty((nitems, nclasses))
        for k in range(nclasses):
            values[:,k] = scipy.stats.beta.rvs(a[k], b[k], size=nitems)
        arguments = values

        x_lower = np.zeros((nclasses*2,)) + 0.5
        x_upper = np.zeros((nclasses*2,)) + 8.
        x0 = np.random.uniform(1., 7.5, size=(nclasses*2,))

        dx = optimum_jump(beta_likelihood, x0.copy(), arguments,
                          x_upper, x_lower,
                          500, 100, 0.3, 0.05, 'Everything')

        njumps = 3000
        samples = sample_distribution(beta_likelihood, x0.copy(), arguments,
                                      dx, njumps, x_lower, x_upper, 'Everything')

        z = np.absolute((samples.mean(0) - np.r_[a,b]) / samples.std(0))

        np.testing.assert_array_less(z, 3.)

if __name__ == '__main__':
    unittest.main()
