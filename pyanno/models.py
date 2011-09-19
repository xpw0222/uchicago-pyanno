"""This file contains the classes defining the models."""
import numpy as np

# TODO generalize beta prior: different items could have different priors
# TODO arguments checking

class ModelB(object):

    def __init__(self, nclasses, nannotators, nitems, pi, theta):
        self.nclasses = nclasses
        self.nannotators = nannotators
        self.nitems = nitems
        self.pi = pi
        self.theta = theta

    @staticmethod
    def random_model(nclasses, nannotators, nitems, alpha=None, beta=None):
        """Factory method that returns a random model.


        Input:
        nclasses -- number of categories
        nitems -- number of items being annotated
        nannotators -- number of annotators
        alpha -- Parameters of Dirichlet prior over annotator choices
                 Default: peaks at correct annotation, decays to 1
        beta -- Parameters of Dirichlet prior over model categories
                Default: beta[i] = 1.0
        """

        if alpha is None:
            alpha = np.empty((nclasses, nclasses))
            for k in xrange(nclasses):
                alpha[k,:] = np.ones((nclasses,))

        if beta is None:
            beta = np.ones(shape=(nclasses,))

        # generate random distributions of prevalence and accuracy
        pi = np.random.dirichlet(beta)
        theta = np.empty((nannotators, nclasses, nclasses))
        for j in xrange(nannotators):
            for k in xrange(nclasses):
                theta[j,k,:] = np.random.dirichlet(alpha[k,:])

        return ModelB(nclasses, nannotators, nitems, pi, theta)

    def generate_samples(self):
        # generate samples from model
        pass
