"""This file contains the classes defining the models."""
import scipy as sp
from pyanno.util import (random_categorical, create_band_matrix,
                         warn_missing_vals, normalize)
from pyanno.multinom import dir_ll

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
            self.pi = sp.ones((nclasses,)) / nclasses
        else:
            self.pi = pi.copy()
        # theta[j,k,:] is P(annotator j chooses : | real label = k)
        if theta is None:
            self.theta = sp.ones((nannotators, nclasses, nclasses)) / nclasses
        else:
            self.theta = theta.copy()

        # initialize prior parameters if not specified
        if alpha is None:
            self.alpha = create_band_matrix((nclasses, nclasses), [4., 2., 1.])
        else:
            self.alpha = alpha
        if beta is None:
            self.beta = sp.ones((nclasses,))
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
            alpha = sp.empty((nclasses, nclasses))
            for k1 in xrange(nclasses):
                for k2 in xrange(nclasses):
                    # using Bob Carpenter's choice as a prior
                    alpha[k1,k2] = max(1, (nclasses + (0.5 if k1 == k2 else 0)
                                           - abs(k1 - k2)) ** 4)

        if beta is None:
            beta = 2.*sp.ones(shape=(nclasses,))

        # generate random distributions of prevalence and accuracy
        pi = sp.random.dirichlet(beta)
        theta = sp.empty((nannotators, nclasses, nclasses))
        for j in xrange(nannotators):
            for k in xrange(nclasses):
                theta[j,k,:] = sp.random.dirichlet(alpha[k,:])

        return ModelB(nclasses, nannotators, nitems, pi, theta, alpha, beta)

    def generate_labels(self):
        """Generate random labels from the model."""
        return random_categorical(self.pi, self.nitems)

    def generate_annotations(self, labels):
        """Generate random annotations given labels."""
        annotations = sp.empty((self.nannotators, self.nitems), dtype=int)
        for j in xrange(self.nannotators):
            for i in xrange(self.nitems):
                annotations[j,i]  = (
                    random_categorical(self.theta[j,labels[i],:], 1))
        return annotations

    # TODO start from sample frequencies
    # TODO argument verbose=False
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
        diff = sp.inf
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
        # FIXME temporary code to interface legacy code
        item = sp.repeat(sp.arange(self.nitems), self.nannotators)
        anno = sp.tile(sp.arange(self.nannotators), self.nitems)
        label = sp.ravel(annotations.T)

        nitems = self.nitems
        nannotators = self.nannotators
        nclasses = self.nclasses
        alpha = self.alpha
        beta = self.beta
        N = len(item)
        Ns = range(N)

        # TODO move argument checking to map_estimate
        if not sp.all(beta > 0.):
            raise ValueError("beta should be larger than 0")
        if not sp.all(alpha > 0.):
            raise ValueError("alpha should be larger than 0")

        if annotations.shape != (nannotators, nitems):
            raise ValueError("size of `annotations` should be nannotators x nitems")
        if init_accuracy < 0.0 or init_accuracy > 1.0:
            raise ValueError("init_accuracy not in [0,1]")
        for n in xrange(N):
            if item[n] < 0:
                raise ValueError("item[n] < 0")
            if anno[n] < 0:
                raise ValueError("anno[n] < 0")
            if label[n] < 0:
                raise ValueError("label[n] < 0")
        if len(alpha) != nclasses:
            raise ValueError("len(alpha) != K")
        for k in xrange(nclasses):
            if len(alpha[k]) != nclasses:
                raise ValueError("len(alpha[k]) != K")
        if len(beta) != nclasses:
            raise ValueError("len(beta) != K")

        warn_missing_vals("item", item)
        warn_missing_vals("anno", anno)
        warn_missing_vals("label", label)

        # initialize params
        alpha_prior_count = alpha - 1.
        beta_prior_count = beta - 1.

        # ??? I think this is wrong, it should rather be initialize at the mean
        # of the dirichlet, i.e., beta / beta.sum()
        prevalence = normalize(beta_prior_count)

        # category is P(category[i] | model, data)
        category = sp.zeros((nitems, nclasses))

        # ??? shouldn't this be normalized?
        accuracy = sp.zeros((nannotators, nclasses, nclasses))
        accuracy += init_accuracy

        while True:
            # -------------------------
            # Expectation step (E-step)
            # compute marginal likelihood P(category[i] | model, data)
            for i in xrange(nitems):
                category[i,:] = prevalence.copy()

            for n in Ns:
                for k in xrange(nclasses):
                    category[item[n]][k] *= accuracy[anno[n]][k][label[n]]

            # need log p(prev|beta) + SUM_k log p(acc[k]|alpha[k])
            # log likelihood here to reuse intermediate category calc
            log_likelihood = 0.0
            for i in xrange(nitems):
                likelihood_i = 0.0
                for k in xrange(nclasses):
                    likelihood_i += category[i][k]
                if likelihood_i < 0.0:
                    print "likelihood_i=", likelihood_i, "cat[i]=", category[i]
                log_likelihood_i = sp.log(likelihood_i)
                log_likelihood += log_likelihood_i

            log_prior = 0.0
            prevalence_a = np.array(prevalence[0:(nclasses - 1)])
            log_prior += dir_ll(prevalence_a, beta)
            for j in xrange(nannotators):
                for k in xrange(nclasses):
                    acc_j_k_a = np.array(accuracy[j][k][0:(nclasses - 1)])
                    log_prior += dir_ll(acc_j_k_a, alpha[k])
            if np.isnan(log_prior) or np.isinf(log_prior):
            if sp.isnan(log_prior) or sp.isinf(log_prior):
                log_prior = 0.0

            for i in xrange(nitems):
                category[i,:] = normalize(category[i,:])

            # return here with E[cat|prev,acc] and LL(prev,acc;y)
            yield (log_prior, log_likelihood, prevalence, category, accuracy)

            # Maximization step (M-step)
            # update parameters to maximize likelihood
            # M: prevalence* + accuracy*
            prevalence = beta_prior_count.copy()
            for i in xrange(nitems):
                prevalence += category[i,:]
            prevalence = normalize(prevalence)

            for j in xrange(nannotators):
                accuracy[j,:] = alpha_prior_count.copy()
            for n in xrange(N):
                for k in xrange(nclasses):
                    accuracy[anno[n]][k][label[n]] += category[item[n]][k]
            for j in xrange(nannotators):
                for k in xrange(nclasses):
                    accuracy[j][k] = normalize(accuracy[j][k])
