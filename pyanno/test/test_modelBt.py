import unittest
import scipy as sp
from numpy import testing
from pyanno.modelBt import ModelBt


class TestModelB(unittest.TestCase):

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
        true_model = ModelBt.random_model(nclasses, nitems)
        labels = true_model.generate_labels()
        annotations = true_model.generate_annotations(labels)

        max_llhood = true_model.log_likelihood(annotations)
        # perturb gamma
        for _ in xrange(20):
            theta = true_model.theta
            gamma = sp.random.normal(loc=true_model.gamma, scale=0.1)
            gamma = sp.clip(gamma, 0., 1.)
            gamma /= gamma.sum()
            model = ModelBt(nclasses, nitems, gamma, theta)
            llhood = model.log_likelihood(annotations)
            self.assertGreater(max_llhood, llhood)

        # perturb theta
        for _ in xrange(20):
            gamma = true_model.gamma
            theta = sp.random.normal(loc=true_model.theta, scale=0.1)
            theta = sp.clip(theta, 0., 1.)
            model = ModelBt(nclasses, nitems, gamma, theta)
            llhood = model.log_likelihood(annotations)
            self.assertGreater(max_llhood, llhood)

if __name__ == '__main__':
    unittest.main()
