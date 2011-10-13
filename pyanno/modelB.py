"""This file contains the classes defining the models."""
import numpy as np
from pyanno.sampling import optimum_jump, sample_distribution
from pyanno.util import (random_categorical, create_band_matrix,
                         warn_missing_vals, normalize, dirichlet_llhood)

# TODO arguments checking

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

        map_em_generator = self._map_em_step(annotations, init_accuracy)
        return self._parameter_estimation(map_em_generator, epsilon, max_epochs)


    def mle(self, annotations,
            epsilon=0.00001, init_accuracy=0.6, max_epochs=1000):
        """Computes maximum likelihood estimate (MLE) of parameters.

        Input:
        annotations -- annotations[i,j] is the annotation of annotator `j`
                       for item `i`

        Output:
        Tuple (diff,ll,lp,cat) consisting of final difference, log likelihood,
        log prior p(acc|alpha) * p(prev|beta), and item category estimates

        The estimates of the label frequency and accuracy parameters,
        are stored in the class attributes `pi` and `theta`.
        """

        mle_em_generator = self._mle_em_step(annotations, init_accuracy)
        return self._parameter_estimation(mle_em_generator, epsilon, max_epochs)


    def _parameter_estimation(self, learning_iterator, epsilon, max_epochs):

        if epsilon < 0.0: raise ValueError("epsilon < 0.0")
        if max_epochs < 0: raise ValueError("max_epochs < 0")

        epoch = 0
        obj_history = []
        diff = np.inf
        for objective, prev_est, cat_est, acc_est in learning_iterator:
            print "  epoch={0:6d}  obj={1:+10.4f}   diff={2:10.4f}".format(epoch, objective, diff)
            obj_history.append(objective)
            # stopping conditions
            if epoch > max_epochs: break
            if epoch > 10:
                diff = (obj_history[epoch] - obj_history[epoch-10]) / 10.0
                if abs(diff) < epsilon:
                    break
            epoch += 1

        # update internal parameters
        self.pi = prev_est
        self.theta = acc_est

        return cat_est


    def _map_em_step(self, annotations, init_accuracy=0.6):
       # TODO move argument checking to traits
#        if not np.all(beta > 0.):
#            raise ValueError("beta should be larger than 0")
#        if not np.all(alpha > 0.):
#            raise ValueError("alpha should be larger than 0")
#
#        if annotations.shape != (nitems, nannotators):
#            raise ValueError("size of `annotations` should be nitems x nannotators")
#        if init_accuracy < 0.0 or init_accuracy > 1.0:
#            raise ValueError("init_accuracy not in [0,1]")
#        if len(alpha) != nclasses:
#            raise ValueError("len(alpha) != K")
#        for k in xrange(nclasses):
#            if len(alpha[k]) != nclasses:
#                raise ValueError("len(alpha[k]) != K")
#        if len(beta) != nclasses:
#            raise ValueError("len(beta) != K")

        # FIXME: at the moment, this check is rather pointless
        warn_missing_vals("anno", annotations.flatten())

        # True if annotations is missing
        missing_mask_nclasses = self._missing_mask(annotations)

        # prevalence is P( category )
        prevalence = self._compute_prevalence()
        accuracy = self._initial_accuracy(init_accuracy)

        while True:
            # Expectation step (E-step)
            # compute marginal likelihood P(category[i] | model, data)

            log_likelihood, unnorm_category = (
                self._log_likelihood_core(annotations,
                                          prevalence,
                                          accuracy,
                                          missing_mask_nclasses)
            )
            log_prior = self._log_prior(prevalence, accuracy)

            # category is P(category[i] = k | model, data)
            category = unnorm_category / unnorm_category.sum(1)[:,None]

            # return here with E[cat|prev,acc] and LL(prev,acc;y)
            yield (log_prior+log_likelihood, prevalence, category, accuracy)

            # Maximization step (M-step)
            # update parameters to maximize likelihood
            prevalence = self._compute_prevalence(category)
            accuracy = self._compute_accuracy(category, annotations,
                                              use_prior=True)


    def _mle_em_step(self, annotations, init_accuracy=0.6):
        # True if annotations is missing
        missing_mask_nclasses = self._missing_mask(annotations)

        # prevalence is P( category )
        prevalence = np.empty((self.nclasses,))
        prevalence.fill(1. / float(self.nclasses))
        accuracy = self._initial_accuracy(init_accuracy)

        while True:
            # Expectation step (E-step)
            # compute marginal likelihood P(category[i] | model, data)

            log_likelihood, unnorm_category = (
                self._log_likelihood_core(annotations,
                                          prevalence,
                                          accuracy,
                                          missing_mask_nclasses)
            )

            # category is P(category[i] = k | model, data)
            category = unnorm_category / unnorm_category.sum(1)[:,None]

            # return here with E[cat|prev,acc] and LL(prev,acc;y)
            yield (log_likelihood, prevalence, category, accuracy)

            # Maximization step (M-step)
            # update parameters to maximize likelihood
            prevalence = normalize(category.sum(0))
            accuracy = self._compute_accuracy(category, annotations,
                                              use_prior=False)


    def _compute_prevalence(self, category=None):
        """Return prevalence, P( category )."""
        beta_prior_count = self.beta - 1.
        if category is None:
            # ??? this may be wrong, it should rather be initialize at the mean
            #     of the dirichlet, i.e., beta / beta.sum()
            return normalize(beta_prior_count)
        else:
            return normalize(beta_prior_count + category.sum(0))


    def _initial_accuracy(self, init_accuracy):
        """Return initial estimation of accuracy."""
        nannotators = self.nannotators
        nclasses = self.nclasses

        # accuracy[j,k,k'] is P(annotation_j = k' | category=k)
        accuracy = np.empty((nannotators, nclasses, nclasses))
        accuracy.fill((1. - init_accuracy) / (nclasses - 1.))
        for k in xrange(nclasses):
            accuracy[:, k, k] = init_accuracy
        return accuracy


    def _compute_accuracy(self, category, annotations, use_prior):
        """Return accuracy, P(annotation_j = k' | category=k)

        accuracy[j,k,k'] = P(annotation_j = k' | category=k).
        """
        nitems, nannotators = annotations.shape
        alpha_prior_count = self.alpha - 1.
        valid_mask = annotations!=-1

        annotators = np.arange(nannotators)[None,:]
        if use_prior:
            accuracy = np.tile(alpha_prior_count, (nannotators, 1, 1))
        else:
            accuracy = np.zeros((nannotators, self.nclasses, self.nclasses))
        for i in xrange(nitems):
            valid = valid_mask[i,:]
            accuracy[annotators[:,valid],:,annotations[i,valid]] += category[i,:]
        accuracy /= accuracy.sum(2)[:, :, None]
        return accuracy


    def _compute_category(self, annotations, prevalence, accuracy,
                          missing_mask_nclasses=None, normalize=True):
        """Compute P(category[i] = k | model, annotations).

        Input:
        accuracy -- the theta parameters
        prevalence -- the pi parameters
        normalize -- if False, do not normalize the posterior
        """

        nitems, nannotators = annotations.shape

        # compute mask of invalid entries in annotations if necessary
        if missing_mask_nclasses is None:
            missing_mask_nclasses = self._missing_mask(annotations)

        # unnorm_category is P(category[i] = k | model, data), unnormalized
        unnorm_category = np.tile(prevalence.copy(), (nitems, 1))
        # mask missing annotations
        annotators = np.arange(nannotators)[None, :]
        tmp = np.ma.masked_array(accuracy[annotators, :, annotations],
                                 mask=missing_mask_nclasses)
        unnorm_category *= tmp.prod(1)

        if normalize:
            return unnorm_category / unnorm_category.sum(1)[:,None]

        return unnorm_category


    def _missing_mask(self, annotations):
        missing_mask = (annotations == -1)
        missing_mask_nclasses = np.tile(missing_mask[:, :, None],
            (1, 1, self.nclasses))
        return missing_mask_nclasses


    ##### Model likelihood methods ############################################

    def log_likelihood(self, annotations):
        missing_mask_nclasses = self._missing_mask(annotations)
        llhood, _ = self._log_likelihood_core(annotations,
                                              self.pi, self.theta,
                                              missing_mask_nclasses)
        return llhood


    def _log_likelihood_core(self, annotations,
                             prevalence, accuracy,
                             missing_mask_nclasses):

        unnorm_category = self._compute_category(annotations,
                                                 prevalence, accuracy,
                                                 missing_mask_nclasses,
                                                 normalize=False)

        llhood = np.log(unnorm_category.sum(1)).sum()
        if np.isnan(llhood):
            llhood = -1e20

        return llhood, unnorm_category


    def _log_prior(self, prevalence, accuracy):
        alpha = self.alpha
        log_prior = dirichlet_llhood(prevalence, self.beta)
        for j in xrange(self.nannotators):
            for k in xrange(self.nclasses):
                log_prior += dirichlet_llhood(accuracy[j,k,:], alpha[k])
        return log_prior


    ##### Sampling posterior over parameters ##################################

    def not_sample_posterior_over_theta(self, annotations, nsamples,
                                    target_rejection_rate = 0.3,
                                    rejection_rate_tolerance = 0.05,
                                    step_optimization_nsamples = 500,
                                    adjust_step_every = 100):
        """Return samples from posterior distribution over theta given data.
        """

        # wrap log likelihood function to give it to optimum_jump and
        # sample_distribution
        missing_mask_nclasses = self._missing_mask(annotations)
        prevalence = self.pi

        _llhood = self._log_likelihood_core
        def _wrap_llhood(params, _):
            theta = self._params_to_theta(params)
            lhood, _ =  _llhood(annotations, prevalence, theta,
                                missing_mask_nclasses)
            return lhood

        # compute optimal step size for given target rejection rate
        params_start = self._theta_to_params(self.theta.copy())
        params_upper = np.ones_like(params_start)
        params_lower = np.zeros_like(params_start)
        step = optimum_jump(_wrap_llhood, params_start, None,
                            params_upper, params_lower,
                            step_optimization_nsamples,
                            adjust_step_every,
                            target_rejection_rate,
                            rejection_rate_tolerance, 'Everything')

        # draw samples from posterior distribution over theta
        samples = sample_distribution(_wrap_llhood, params_start, None,
                                      step, nsamples,
                                      params_lower, params_upper,
                                      'Everything')
        theta_samples = np.empty((nsamples, self.nannotators,
                                  self.nclasses, self.nclasses))
        for i in xrange(nsamples):
            theta_samples[i,:] = self._params_to_theta(samples[i,:])
        return theta_samples


    # functions to flatten and unflatten theta parameter
    def _theta_to_params(self, theta):
        theta_part = theta[:,:,:-1]
        return theta_part.flatten()


    def _params_to_theta(self, params):
        nclasses = self.nclasses
        theta = np.empty((self.nannotators, nclasses, nclasses))
        theta_part = params.reshape((self.nannotators,
                                     nclasses, nclasses-1))
        theta[:,:,:-1] = theta_part
        theta[:,:,-1] = 1. - theta_part.sum(2)

        return theta


    ##### Posterior distributions #############################################

    def infer_labels(self, annotations):
        """Infer posterior distribution over true labels.

        Returns P( label | annotations, parameters), where parameters is the
        current point estimate of the parameters pi and theta.
        """

        category = self._compute_category(annotations,
                                            self.pi,
                                            self.theta)

        return category
