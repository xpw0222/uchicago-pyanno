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
    def map(self, annotations,
            epsilon=0.00001, init_accuracy=0.6, max_epochs=1000):
        """Computes maximum a posteriori (MAP) estimate of parameters.

        See the documentation for pyanno.multinom.mle() in this module for
        a description of all but the following inputs:

        Input:
        annotations -- annotations[j,i] is the annotation of annotator `j`
                       for item `i`

        Output:
        Tuple (diff,ll,lp,cat) consisting of final difference, log likelihood,
        log prior p(acc|alpha) * p(prev|beta), and item category estimates

        The estimates of the label frequency and accuracy parameters,
        are stored in the class attributes `pi` and `theta`.
        """

        if epsilon < 0.0:
            raise ValueError("epislon < 0.0")
        if max_epochs < 0:
            raise ValueError("max_epochs < 0")

        llp_curve = []
        epoch = 0
        diff = np.inf
        item = np.repeat(np.arange(self.nitems), self.nannotators)
        anno = np.tile(np.arange(self.nannotators), self.nitems)
        label = np.ravel(annotations.T)
        map_em_generator = pyanno.multinom.map_em(item, anno, label,
                                                  self.alpha, self.beta,
                                                  init_accuracy)
        for lp, ll, prev_map, cat_map, accuracy_map in map_em_generator:
            print "  epoch={0:6d}  log lik={1:+10.4f}  log prior={2:+10.4f}  llp={3:+10.4f}   diff={4:10.4f}".\
            format(epoch, ll, lp, ll + lp, diff)
            llp_curve.append(ll + lp)
            # stopping conditions
            if epoch > max_epochs:
                break
            if len(llp_curve) > 10:
                diff = (llp_curve[epoch] - llp_curve[epoch - 10]) / 10.0
                if abs(diff) < epsilon:
                    break
            epoch += 1

        self.pi = prev_map
        self.theta = accuracy_map

        return diff, ll, lp, cat_map

    def _map_em_step(self, annotations, init_accuracy=0.6):
        # FIXME temporary code to interface legacy code
        pass
