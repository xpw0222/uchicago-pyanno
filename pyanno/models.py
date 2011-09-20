"""This file contains the classes defining the models."""
import numpy as np
from pyanno.util import random_categorical, create_band_matrix
import pyanno.multinom

# TODO generalize beta prior: different items could have different priors
# TODO arguments checking

class ModelB(object):
    """See Model.txt for a description of the model."""

    def __init__(self, nclasses, nannotators, nitems,
                 pi=None, theta=None,
                 alpha=None, beta=None):
        self.nclasses = nclasses
        self.nannotators = nannotators
        self.nitems = nitems
        if pi is None:
            self.pi = np.ones((nclasses,)) / nclasses
        else:
            self.pi = pi.copy()
        # theta[j,k,:] is P(annotator j chooses : | real label = k)
        if theta is None:
            self.theta = np.ones((nannotators, nclasses, nclasses)) / nclasses
        else:
            self.theta = theta.copy()
        # initialize prior parameters if not specified
        if alpha is None:
            self.alpha = create_band_matrix((nclasses, nclasses), [4., 2., 1.])
        else:
            self.alpha = alpha
        if beta is None:
            self.beta = np.ones((nclasses,))
        else:
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

        # TODO more meaningful choice for priors
        if alpha is None:
            alpha = np.empty((nclasses, nclasses))
            for k1 in xrange(nclasses):
                for k2 in xrange(nclasses):
                    # using Bob Carpenter's choice as a prior
                    alpha[k1,k2] = max(1, (nclasses + (0.5 if k1 == k2 else 0)
                                           - abs(k1 - k2)) ** 4)

        if beta is None:
            beta = 2.*np.ones(shape=(nclasses,))

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

    # TODO start from sample frequencies
    def map(self, annotations, epsilon=0.00001, init_accuracy=0.6):
        # FIXME temporary code to interface legacy code
        item = np.repeat(np.arange(self.nitems), self.nannotators)
        anno = np.tile(np.arange(self.nannotators), self.nitems)
        label = np.ravel(annotations.T)

        (diff,ll,lp,prev_map,
         cat_map,accuracy_map) = pyanno.multinom.map(item, anno, label,
                                                     self.alpha.tolist(), self.beta.tolist(),
                                                     init_accuracy, epsilon)
        self.pi = prev_map
        self.theta = accuracy_map

        return diff, ll, lp, cat_map
