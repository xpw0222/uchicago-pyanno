"""This file contains the classes defining the models."""
import numpy as np
from pyanno.util import (random_categorical, create_band_matrix,
                         warn_missing_vals, normalize, benchmark)
from pyanno.multinom import dirichlet_llhood, map_em

# TODO arguments checking
# TODO MLE estimation
# TODO compute log likelihood

class ModelB(object):
    """See Model.txt for a description of the model."""

    def __init__(self, nclasses, nannotators,
                 pi=None, theta=None,
                 alpha=None, beta=None):
        self.nclasses = nclasses
        self.nannotators = nannotators

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

    ##### Model and data generation methods ###################################

    @staticmethod
    def create_initial_state(nclasses, nannotators, alpha=None, beta=None):
        """Factory method that returns a random model.

        Input:
        nclasses -- number of categories
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

        return ModelB(nclasses, nannotators, pi, theta, alpha, beta)

    def generate_labels(self, nitems):
        """Generate random labels from the model."""
        return random_categorical(self.pi, nitems)

    def generate_annotations(self, labels):
        """Generate random annotations given labels."""
        nitems = labels.shape[0]
        annotations = np.empty((nitems, self.nannotators), dtype=int)
        for j in xrange(self.nannotators):
            for i in xrange(nitems):
                annotations[i,j]  = (
                    random_categorical(self.theta[j,labels[i],:], 1))
        return annotations


    ##### Parameters estimation methods #######################################

    # TODO start from sample frequencies
    # TODO argument verbose=False
    def map(self, annotations,
            epsilon=0.00001, init_accuracy=0.6, max_epochs=1000):
        """Computes maximum a posteriori (MAP) estimate of parameters.

        See the documentation for pyanno.multinom.mle() in this module for
        a description of all but the following inputs:

        Input:
        annotations -- annotations[i,j] is the annotation of annotator `j`
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

        map_em_generator = self._map_em_step(annotations, init_accuracy)
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

    # TODO check equations with paper
    def _map_em_step(self, annotations, init_accuracy=0.6):
        nannotators = self.nannotators
        nclasses = self.nclasses
        nitems = annotations.shape[0]
        alpha = self.alpha
        beta = self.beta

        # True if annotations is missing
        missing_mask = (annotations==-1)
        valid_mask = (annotations!=-1)
        missing_mask_nclasses = np.tile(missing_mask[:,:,None], (1,1,nclasses))

        # TODO move argument checking to map_estimate
        if not np.all(beta > 0.):
            raise ValueError("beta should be larger than 0")
        if not np.all(alpha > 0.):
            raise ValueError("alpha should be larger than 0")

        if annotations.shape != (nitems, nannotators):
            raise ValueError("size of `annotations` should be nitems x nannotators")
        if init_accuracy < 0.0 or init_accuracy > 1.0:
            raise ValueError("init_accuracy not in [0,1]")
        if len(alpha) != nclasses:
            raise ValueError("len(alpha) != K")
        for k in xrange(nclasses):
            if len(alpha[k]) != nclasses:
                raise ValueError("len(alpha[k]) != K")
        if len(beta) != nclasses:
            raise ValueError("len(beta) != K")

        # FIXME: at the moment, this check is rather poitnless
        warn_missing_vals("anno", annotations.flatten())

        # initialize params
        alpha_prior_count = alpha - 1.
        beta_prior_count = beta - 1.

        # ??? I think this is wrong, it should rather be initialize at the mean
        # of the dirichlet, i.e., beta / beta.sum()
        # prevalence is P( category )
        prevalence = normalize(beta_prior_count)

        # accuracy[j,k,k'] is P(annotation_j = k' | category=k)
        accuracy = np.empty((nannotators, nclasses, nclasses))
        accuracy.fill((1.-init_accuracy) / (nclasses-1.))
        for k in xrange(nclasses):
            accuracy[:,k,k] = init_accuracy

        annotators = np.arange(nannotators)[None,:]
        while True:
            # -------------------------
            # Expectation step (E-step)
            # compute marginal likelihood P(category[i] | model, data)

            # unnorm_category is P(category[i] = k | model, data) (unnormalized)
            unnorm_category = np.tile(prevalence, (nitems, 1))
            # mask missing annotations
            tmp = np.ma.masked_array(accuracy[annotators,:,annotations],
                                     mask=missing_mask_nclasses)
            unnorm_category *= tmp.prod(1)

            # need log p(prev|beta) + SUM_k log p(acc[k]|alpha[k])
            # log likelihood here to reuse intermediate category calc
            log_likelihood = np.log(unnorm_category.sum(1)).sum()

            log_prior = dirichlet_llhood(prevalence, beta)
            for j in xrange(nannotators):
                for k in xrange(nclasses):
                    log_prior += dirichlet_llhood(accuracy[j,k,:], alpha[k])
            if np.isnan(log_prior) or np.isinf(log_prior):
                log_prior = 0.0

            # category is P(category[i] = k | model, data)
            category = unnorm_category / unnorm_category.sum(1)[:,None]

            # return here with E[cat|prev,acc] and LL(prev,acc;y)
            yield (log_prior, log_likelihood, prevalence, category, accuracy)

            # Maximization step (M-step)
            # update parameters to maximize likelihood
            # M: prevalence* + accuracy*
            prevalence = normalize(beta_prior_count + category.sum(0))

            accuracy = np.tile(alpha_prior_count, (nannotators, 1, 1))
            for i in xrange(nitems):
                valid = valid_mask[i,:]
                accuracy[annotators[:,valid],:,annotations[i,valid]] += category[i,:]
            accuracy /= accuracy.sum(2)[:,:,None]

