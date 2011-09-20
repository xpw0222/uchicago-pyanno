"""This file contains the classes defining the models."""
import numpy as np
from pyanno.util import random_categorical

# TODO generalize beta prior: different items could have different priors
# TODO arguments checking
# TODO default for alpha, beta

class ModelB(object):

    def __init__(self, nclasses, nannotators, nitems, pi, theta,
                 alpha=None, beta=None):
        self.nclasses = nclasses
        self.nannotators = nannotators
        self.nitems = nitems
        self.pi = pi
        # theta[j,k,:] is P(annotator j chooses : | real label = k)
        self.theta = theta
        # initialize prior parameters if not specified
        self.alpha = alpha
        self.beta = beta

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
            for k1 in xrange(nclasses):
                for k2 in xrange(nclasses):
                    # using Bob Carpenter's choice as a prior
                    # TODO more meaningful choice
                    alpha[k1,k2] = max(1, (nclasses + (0.5 if k1 == k2 else 0)
                                           - abs(k1 - k2)) ** 4)

        if beta is None:
            beta = np.ones(shape=(nclasses,))

        # generate random distributions of prevalence and accuracy
        pi = np.random.dirichlet(beta)
        theta = np.empty((nannotators, nclasses, nclasses))
        for j in xrange(nannotators):
            for k in xrange(nclasses):
                theta[j,k,:] = np.random.dirichlet(alpha[k,:])

        return ModelB(nclasses, nannotators, nitems, pi, theta, alpha, beta)

    def generate_labels(self):
        """Generate random labels from the model."""
        return random_categorical(self.pi, self.nitems)

    def generate_annotations(self, labels):
        """Generate random annotations given labels."""
        annotations = np.empty((self.nannotators, self.nitems), dtype=int)
        for j in xrange(self.nannotators):
            for i in xrange(self.nitems):
                annotations[j,i]  = (
                    random_categorical(self.theta[j,labels[i],:], 1))
        return annotations
