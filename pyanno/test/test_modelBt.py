import unittest
import scipy as sp
from numpy import testing
from pyanno.modelBt import ModelBt


class TestModelB(unittest.TestCase):

    def test_mle_estimation(self):
        # test simple model, check that we get to global optimum
        nclasses, nitems = 3, 5000*8
        # create random model and data (this is our graund truth model)
        true_model = ModelBt.random_model(nclasses, nitems)
        labels = true_model.generate_labels()
        annotations = true_model.generate_annotations(labels)

        # create a new, empty model and infer back the parameters
        model = ModelBt.random_model(nclasses, nitems)
        model.mle(annotations)

        testing.assert_allclose(model.gamma, true_model.gamma, atol=1e-2, rtol=0.)
        testing.assert_allclose(model.theta, true_model.theta, atol=1e-2, rtol=0.)


if __name__ == '__main__':
    unittest.main()

