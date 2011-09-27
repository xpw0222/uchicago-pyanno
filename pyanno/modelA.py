"""Definition of model A."""

from collections import defaultdict
import numpy as np
import scipy.optimize
from pyanno.sampling import optimum_jump, sample_distribution
from pyanno.util import compute_counts, random_categorical


_compatibility_tables_cache = {}
def _compatibility_tables(nclasses):
    """Return a map from agreement indices to annotation patterns.

    The agreement indices are defined as in Table 3 of
    Rzhetsky et al., 2009, supplementary material:
    0=aaa, 1=aaA, 2=aAa, 3=Aaa, 4=Aa@

    The dictionary maps an agreement index to an array of annotations
    compatible with the corresponding agreement pattern.
    For example, for nclasses=3 and index=1 the array contains these
    annotations:
    0 0 1
    0 0 2
    1 1 0
    1 1 2
    2 2 0
    2 2 1
    """

    if not _compatibility_tables_cache.has_key(nclasses):
        compatibility = defaultdict(list)

        # aaa
        for psi1 in range(nclasses):
            compatibility[0].append([psi1, psi1, psi1])

        # aaA, aAa, Aaa
        for psi1 in range(nclasses):
            for psi2 in range(nclasses):
                if psi2 == psi1: continue
                compatibility[1].append([psi1, psi1, psi2])
                compatibility[2].append([psi1, psi2, psi1])
                compatibility[3].append([psi2, psi1, psi1])

        # Aa@
        for psi1 in range(nclasses):
            for psi2 in range(nclasses):
                for psi3 in range(nclasses):
                    if psi1==psi2 or psi2==psi3 or psi1==psi3: continue
                    compatibility[4].append([psi1, psi2, psi3])

        for idx in range(5):
            compatibility[idx] = np.array(compatibility[idx], dtype=int)

        _compatibility_tables_cache[nclasses] = compatibility

    return _compatibility_tables_cache[nclasses]


class ModelA(object):
    """
    At the moment the model assumes 1) a total of 8 annotators, and 2) each
    item is annotated by 3 annotators.
    """

    def __init__(self, nclasses, theta, alpha, omega,
                 use_priors=True, use_omegas=True):
        self.nclasses = nclasses
        self.nannotators = 8
        # number of annotators rating each item in the loop design
        self.annotators_per_item = 3
        self.theta = theta
        self.alpha = alpha
        self.omega = omega
        self.use_priors = use_priors
        self.use_omegas = use_omegas


    @staticmethod
    def random_model(nclasses, theta=None, alpha=None, omega=None,
                     use_priors=True, use_omegas=True):
        """Factory method returning a random model.

         Input:
         nclasses -- number of categories
         theta -- probability of annotators being correct
         alpha -- probability of drawing agreement pattern given
                  correctness pattern
         omega -- probability of drawing an annotation value
                  (assumed to be the same for all annotators)
         """

        if theta is None:
            nannotators = 8
            theta = ModelA._random_theta(nannotators)

        if alpha is None:
            alpha = ModelA._random_alpha()

        if omega is None:
            omega = ModelA._random_omega(nclasses)

        return ModelA(nclasses, theta, alpha, omega, use_priors, use_omegas)


    @staticmethod
    def _random_theta(nannotators):
        return np.random.uniform(low=0.6, high=0.95,
                                 size=(nannotators,))


    @staticmethod
    def _random_alpha():
        alpha = np.empty((7,))
        alpha[0:3] = np.random.uniform(0.6, 0.99)
        alpha[4:7] = np.random.uniform(0.1, 0.25)
        rest = 1. - alpha[4:7].sum()
        alpha[3] = np.random.uniform(0.01, rest-0.01)
        assert alpha[3:].sum() < 1.
        return alpha


    @staticmethod
    def _random_omega(nclasses):
        beta = 2.*np.ones((nclasses,))
        return np.random.dirichlet(beta)


    def _generate_incorrectness(self, n, theta_triplet):
        _rnd = np.random.rand(n, self.annotators_per_item)
        incorrect = _rnd >= theta_triplet
        return incorrect


    def _generate_agreement(self, incorrect):
        """Return indices of agreement pattern given correctness pattern.

        The indices returned correspond to agreement patterns
        as in Table 3: 0=aaa, 1=aaA, 2=aAa, 3=Aaa, 4=Aa@
        """

        # create tensor A_ijk
        # (cf. Table 3 in Rzhetsky et al., 2009, suppl. mat.)
        alpha = self.alpha
        agreement_tbl = np.array(
            [[1.,       0.,       0.,       0.,       0.],
             [0.,       1.,       0.,       0.,       0.],
             [0.,       0.,       1.,       0.,       0.],
             [0.,       0.,       0.,       1.,       0.],
             [0.,       0.,       0.,       alpha[0], 1.-alpha[0]],
             [0.,       0.,       alpha[1], 0.,       1.-alpha[1]],
             [0.,       alpha[2], 0.,       0.,       1.-alpha[2]],
             [alpha[3], alpha[4], alpha[5], alpha[6], 1.-alpha[3:].sum()]])

        # this array maps boolean correctness patterns (e.g., CCI) to
        # indices in the agreement tensor, `agreement_tbl`
        correctness_to_agreement_idx = np.array([0, 3, 2, 6, 1, 5, 4, 7])

        # convert correctness pattern to index in the A_ijk tensor
        correct_idx = correctness_to_agreement_idx[
            incorrect[:,0]*1 + incorrect[:,1]*2 + incorrect[:,2]*4]

        # the indices stored in `agreement` correspond to agreement patterns
        # as in Table 3: 0=aaa, 1=aaA, 2=aAa, 3=Aaa, 4=Aa@
        nitems_per_loop = incorrect.shape[0]
        agreement = np.empty((nitems_per_loop,), dtype=int)
        for i in xrange(nitems_per_loop):
            # generate agreement pattern according to A_ijk
            agreement[i] = random_categorical(
                agreement_tbl[correct_idx[i]], 1)

        return agreement


    def _generate_annotations(self, agreement):
        """Generate triplet annotations given agreement pattern."""
        nitems_per_loop = agreement.shape[0]
        omega = self.omega
        annotations = np.empty((nitems_per_loop, 3), dtype=int)

        for i in xrange(nitems_per_loop):
            # get all compatible annotations
            compatible = _compatibility_tables(self.nclasses)[agreement[i]]
            # compute probability of each possible annotation
            distr = omega[compatible].prod(1)
            distr /= distr.sum()
            # draw annotation
            compatible_idx = random_categorical(distr, 1)[0]
            annotations[i,:] = compatible[compatible_idx, :]
        return annotations


    def generate_annotations(self, nitems):
        """Generate random annotations given labels."""
        theta = self.theta
        nannotators = self.nannotators
        nitems_per_loop = nitems // nannotators

        annotations = np.empty((nitems, nannotators), dtype=int)
        annotations.fill(-1)

        # loop over annotator triplets (loop design)
        for j in range(nannotators):
            triplet_indices = np.arange(j, j+3) % self.nannotators

            # -- step 1: generate correct / incorrect labels

            # parameters for this triplet
            theta_triplet = self.theta[triplet_indices]
            incorrect = self._generate_incorrectness(nitems_per_loop,
                                                     theta_triplet)

            # -- step 2: generate agreement patterns given correctness
            # convert boolean correctness into combination indices
            # (indices as in Table 3)
            agreement = self._generate_agreement(incorrect)

            # -- step 3: generate annotations
            annotations[j*nitems_per_loop:(j+1)*nitems_per_loop,
                        triplet_indices] = (
                            self._generate_annotations(agreement)
                        )

        return annotations


    # ---- Parameters estimation methods

    def mle(self, annotations):
        pass
