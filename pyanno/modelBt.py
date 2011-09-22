"""Definition of model B-with-theta."""

import scipy as sp
from pyanno.modelAB import random_startBt8, likeBt8, compute_counts
from pyanno.util import (random_categorical,
                         warn_missing_vals, normalize)


class ModelBt(object):

    def __init__(self, nclasses, nitems, gamma, theta):
        self.nclasses = nclasses
        self.nannotators = 8
        self.nitems = nitems
        # number of annotators rating each item in the loop design
        self.annotators_per_item = 3
        self.gamma = gamma
        self.theta = theta


    @staticmethod
    def random_model(nclasses, nitems,
                     gamma=None, theta=None):
        """Factory method returning a random model.

        Input:
        nclasses -- number of categories
        nitems -- number of items being annotated
        gamma -- probability of each annotation value
        theta -- the parameters of P( v_i | psi ) (one for each annotator)
        """

        if gamma is None:
            beta = 2.*sp.ones((nclasses,))
            gamma = sp.random.dirichlet(beta)

        if theta is None:
            nannotators = 8
            theta = sp.random.uniform(low=0.6, high=0.9,
                                      size=(nannotators,))

        model = ModelBt(nclasses, nitems, gamma, theta)
        return model


    def generate_labels(self):
        """Generate random labels from the model."""
        return random_categorical(self.gamma, self.nitems)


    def _theta_to_categorical(self, theta, psi):
        """Returns P( v_i = psi | theta_i ) as a distribution."""
        distr = sp.empty((self.nclasses,))
        distr.fill((1.-theta)/(self.nclasses-1.))
        distr[psi] = theta
        assert sp.allclose(distr.sum(), 1.)
        return distr


    # FIXME: different conventions on orientation of annotations here and in ModelB
    def generate_annotations(self, labels):
        """Generate random annotations given labels."""
        theta = self.theta
        nannotators = self.nannotators
        nitems_per_loop = self.nitems // nannotators

        annotations = sp.empty((self.nitems, nannotators), dtype=int)
        for j in xrange(nannotators):
            for i in xrange(self.nitems):
                distr = self._theta_to_categorical(theta[j], labels[i])
                annotations[i,j]  = random_categorical(distr, 1)

        # mask annotation value according to loop design
        for l in xrange(nannotators):
            label_idx = sp.arange(l+self.annotators_per_item, l+nannotators) % 8
            annotations[l*nitems_per_loop:(l+1)*nitems_per_loop, label_idx] = -1

        return annotations


    def mle(self, annotations, use_priors=1, use_omegas=1):
        nclasses = self.nclasses

        counts = compute_counts(annotations, self.nclasses)
        arguments = ((counts, nclasses, use_priors),)
        x0 = random_startBt8(nclasses, use_omegas, counts, report='Everything')
        # TODO: use gradient, constrained optimization
        x_best = sp.optimize.fmin(likeBt8, x0, args=arguments,
                                  xtol=1e-4, ftol=1e-4,
                                  disp=True, maxiter=1e+10,
                                  maxfun=1e+30)
        print 'x_best', x_best
        # parse arguments and update
        self.gamma[:nclasses-1] = x_best[:nclasses-1]
        self.gamma[-1] = 1. - self.gamma[:nclasses-1].sum()
        self.theta = x_best[nclasses-1:]
