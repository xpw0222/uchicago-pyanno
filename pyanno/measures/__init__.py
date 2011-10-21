"""Define standard reliability and agreement measures."""

# TODO: functions to compute confidence interval
# TODO: compare results with nltk

from helpers import pairwise_matrix
from covariation import pearsons_rho, spearmans_rho, cronbachs_alpha
from agreement import (scotts_pi, cohens_kappa, cohens_weighted_kappa,
                       fleiss_kappa, krippendorffs_alpha)
