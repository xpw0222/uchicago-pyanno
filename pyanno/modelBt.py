"""Definition of model B-with-theta."""

from traits.api import HasStrictTraits, Int, Array, Bool
import numpy as np
import scipy.optimize
from pyanno.sampling import optimum_jump, sample_distribution
from pyanno.util import random_categorical, log_beta_pdf, compute_counts, annotations_frequency


# map of `n` to list of all possible triplets of `n` elements
_triplet_combinations = {}
def _get_triplet_combinations(n):
    """Return array of all possible combinations of n elements in triplets.
    """
    if not _triplet_combinations.has_key(n):
        _triplet_combinations[n] = (
            np.array([i for i in np.ndindex(n,n,n)]) )
    return _triplet_combinations[n]


# ??? use_priors = True switches optimization from ML to MAP -> refactor
# idea: have mle(annotations) optimize self.log_likelihood, and
# map(annotations) optimize self.log_likelihood + log P(theta | beta prior)
class ModelBt(HasStrictTraits):
    """
    At the moment the model assumes 1) a total of 8 annotators, and 2) each
    item is annotated by 3 annotators.
    """

    nclasses = Int
    nannotators = Int(8)
    # number of annotators rating each item in the loop design
    annotators_per_item = Int(3)
    gamma = Array(dtype=float, shape=(None,))
    theta = Array(dtype=float, shape=(None,))

    def __init__(self, nclasses, gamma, theta):
        self.nclasses = nclasses
        self.gamma = gamma
        self.theta = theta


    ##### Model and data generation methods ###################################

    @staticmethod
    def create_initial_state(nclasses, gamma=None, theta=None):
        """Factory method returning a random model.

        Input:
        nclasses -- number of categories
        gamma -- probability of each annotation value
        theta -- the parameters of P( v_i | psi ) (one for each annotator)
        """

        if gamma is None:
            gamma = ModelBt._random_gamma(nclasses)

        if theta is None:
            nannotators = 8
            theta = ModelBt._random_theta(nannotators)

        model = ModelBt(nclasses, gamma, theta)
        return model


    @staticmethod
    def _random_gamma(nclasses):
        beta = 2.*np.ones((nclasses,))
        return np.random.dirichlet(beta)


    @staticmethod
    def _random_theta(nannotators):
        return np.random.uniform(low=0.6, high=0.95,
                                 size=(nannotators,))


    def generate_labels(self, nitems):
        """Generate random labels from the model."""
        return random_categorical(self.gamma, nitems)


    def generate_annotations(self, labels):
        """Generate random annotations given labels."""
        theta = self.theta
        nannotators = self.nannotators
        nitems = labels.shape[0]
        nitems_per_loop = np.ceil(float(nitems) / nannotators)

        annotations = np.empty((nitems, nannotators), dtype=int)
        for j in xrange(nannotators):
            for i in xrange(nitems):
                distr = self._theta_to_categorical(theta[j], labels[i])
                annotations[i,j]  = random_categorical(distr, 1)

        # mask annotation value according to loop design
        for l in xrange(nannotators):
            label_idx = np.arange(l+self.annotators_per_item, l+nannotators) % 8
            annotations[l*nitems_per_loop:(l+1)*nitems_per_loop, label_idx] = -1

        return annotations


    def _theta_to_categorical(self, theta, psi):
        """Returns P( v_i = psi | theta_i ) as a distribution."""
        distr = np.empty((self.nclasses,))
        distr.fill((1.-theta)/(self.nclasses-1.))
        distr[psi] = theta
        assert np.allclose(distr.sum(), 1.)
        return distr


    ##### Parameters estimation methods #######################################

    def mle(self, annotations, estimate_gamma=True):

        # wrap log likelihood function to give it to optimize.fmin
        _llhood_counts = self._log_likelihood_counts
        def _wrap_llhood(params, counts):
            self.gamma, self.theta = self._vector_to_params(params)
            # minimize *negative* likelihood
            return - _llhood_counts(counts)

        self._parameter_estimation(_wrap_llhood, annotations,
                                   estimate_gamma=estimate_gamma)


    def map(self, annotations, estimate_gamma=True):

        # wrap log likelihood function to give it to optimize.fmin
        _llhood_counts = self._log_likelihood_counts
        _log_prior = self._log_prior
        def _wrap_llhood(params, counts):
            self.gamma, self.theta = self._vector_to_params(params)
            # minimize *negative* posterior probability of parameters
            return - (_llhood_counts(counts) + _log_prior())

        self._parameter_estimation(_wrap_llhood, annotations,
                                   estimate_gamma=estimate_gamma)


    def _parameter_estimation(self, objective, annotations,
                              estimate_gamma=True):
        counts = compute_counts(annotations, self.nclasses)

        params_start = self._random_initial_parameters(annotations,
                                                       estimate_gamma)

        # TODO: use gradient, constrained optimization
        params_best = scipy.optimize.fmin(objective,
                                          params_start,
                                          args=(counts,),
                                          xtol=1e-4, ftol=1e-4,
                                          disp=True, maxiter=1e+10,
                                          maxfun=1e+30)

        # parse arguments and update
        self.gamma, self.theta = self._vector_to_params(params_best)


    def _random_initial_parameters(self, annotations, estimate_gamma):
        if estimate_gamma:
            # estimate gamma from observed annotations
            gamma = annotations_frequency(annotations, self.nclasses)
        else:
            gamma = ModelBt._random_gamma(self.nclasses)

        theta = ModelBt._random_theta(self.nannotators)
        return self._params_to_vector(gamma, theta)


    def _params_to_vector(self, gamma, theta):
        """Convert the tuple (gamma, theta) to a parameters vector.

        Used to interface with the optimization routines.
        """
        return np.r_[gamma[:-1], theta]


    def _vector_to_params(self, params):
        """Convert a parameters vector to (gamma, theta) tuple.

        Used to interface with the optimization routines.
        """
        nclasses = self.nclasses
        gamma = np.zeros((nclasses,))
        gamma[:nclasses-1] = params[:nclasses-1]
        gamma[-1] = 1. - gamma[:nclasses-1].sum()
        theta = params[nclasses-1:]
        return gamma, theta


    ##### Model likelihood methods ############################################

    def log_likelihood(self, annotations):
        """Compute the log likelihood of annotations given the model."""
        counts = compute_counts(annotations, self.nclasses)
        return self._log_likelihood_counts(counts)


    def _log_likelihood_counts(self, counts):
        """Compute the log likelihood of annotations given the model.

        This method assumes the data is in counts format.
        """

        # TODO: check if it's possible to replace these constraints with bounded optimization
        # check boundary conditions
        if (min(min(self.gamma), min(self.theta)) < 0.
            or max(max(self.gamma), max(self.theta)) > 1.):
            #return np.inf
            return -1e20

        llhood = 0.
        # loop over the 8 combinations of annotators
        for i in range(8):
            # extract the theta parameters for this triplet
            triplet_indices = np.arange(i, i+3) % self.nannotators
            triplet_indices.sort()
            theta_triplet = self.theta[triplet_indices]

            # compute the likelihood for the triplet
            llhood += self._log_likelihood_triplet(counts[:,i],
                                                   theta_triplet)

        return llhood


    def _log_likelihood_triplet(self, counts_triplet, theta_triplet):
        """Compute the log likelihood of data for one triplet of annotators.

        Input:
        counts_triplet -- count data for one combination of annotators
        theta_triplet -- theta parameters of the current triplet
        """

        # log \prod_n P(v_{ijk}^{n} | params)
        # = \sum_n log P(v_{ijk}^{n} | params)
        # = \sum_v_{ijk}  count(v_{ijk}) log P( v_{ijk} | params )
        #
        # where n is n-th annotation of triplet {ijk}]

        # compute P( v_{ijk} | params )
        pf = self._pattern_frequencies(theta_triplet)
        l = (counts_triplet * np.log(pf)).sum()

        return l


    def _pattern_frequencies(self, theta_triplet):
        """Compute vector of P(v_{ijk}|params) for each combination of v_{ijk}.
        """

        gamma = self.gamma
        nclasses = self.nclasses
        # list of all possible combinations of v_i, v_j, v_k elements
        v_ijk_combinations = _get_triplet_combinations(nclasses)

        # P( v_{ijk} | params ) = \sum_psi P( v_{ijk} | psi, params ) P( psi )

        pf = 0.
        not_theta = (1.-theta_triplet) / (nclasses-1.)
        p_v_ijk_given_psi = np.empty_like(v_ijk_combinations, dtype=float)
        for psi in range(nclasses):
            for j in range(3):
                p_v_ijk_given_psi[:,j] = np.where(v_ijk_combinations[:,j]==psi,
                                                  theta_triplet[j],
                                                  not_theta[j])
            pf += p_v_ijk_given_psi.prod(1) * gamma[psi]
        return pf


    def _log_prior(self):
        """Compute log probability of prior on the theta parameters."""
        log_prob = log_beta_pdf(self.theta, 2., 1.).sum()
        return log_prob


    ##### Sampling posterior over parameters ##################################

    # TODO arguments for burn-in, thinning
    def sample_posterior_over_theta(self, annotations, nsamples,
                                    target_rejection_rate = 0.3,
                                    rejection_rate_tolerance = 0.05,
                                    step_optimization_nsamples = 500,
                                    adjust_step_every = 100):
        """Return samples from posterior distribution over theta given data.
        """
        # optimize step size
        counts = compute_counts(annotations, self.nclasses)

        # wrap log likelihood function to give it to optimum_jump and
        # sample_distribution
        _llhood_counts = self._log_likelihood_counts
        def _wrap_llhood(params, counts):
            self.theta = params
            # minimize *negative* likelihood
            return _llhood_counts(counts)

        # TODO this save-reset is rather ugly, refactor: create copy of
        #      model and sample over it
        # save internal parameters to reset at the end of sampling
        save_params = (self.gamma, self.theta)
        try:
            # compute optimal step size for given target rejection rate
            params_start = self.theta.copy()
            params_upper = np.ones((self.nannotators,))
            params_lower = np.zeros((self.nannotators,))
            step = optimum_jump(_wrap_llhood, params_start, counts,
                                params_upper, params_lower,
                                step_optimization_nsamples,
                                adjust_step_every,
                                target_rejection_rate,
                                rejection_rate_tolerance, 'Everything')

            # draw samples from posterior distribution over theta
            samples = sample_distribution(_wrap_llhood, params_start, counts,
                                          step, nsamples,
                                          params_lower, params_upper,
                                          'Everything')
            return samples
        finally:
            # reset parameters
            self.gamma, self.theta = save_params


    ##### Posterior distributions #############################################

    def infer_labels(self, annotations):
        """Infer posterior distribution over true labels given theta."""
        nitems = annotations.shape[0]
        gamma = self.gamma
        nclasses = self.nclasses

        # get indices of annotators active in each row
        valid_entries = (annotations > -1).nonzero()
        annotator_indices = np.reshape(valid_entries[1],
            (nitems, self.annotators_per_item))
        valid_annotations = annotations[valid_entries]
        valid_annotations = np.reshape(valid_annotations,
            (nitems, self.annotators_per_item))

        # thetas of active annotators
        theta_equal = self.theta[annotator_indices]
        theta_not_equal = (1. - theta_equal) / (nclasses - 1.)

        # compute posterior over psi
        psi_distr = np.zeros((nitems, nclasses))
        for psi in xrange(nclasses):
            tmp = np.where(valid_annotations == psi,
                           theta_equal, theta_not_equal)
            psi_distr[:,psi] = gamma[psi] * tmp.prod(1)

        # normalize distribution
        psi_distr /= psi_distr.sum(1)[:,np.newaxis]
        return psi_distr
