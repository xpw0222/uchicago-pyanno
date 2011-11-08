# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

from __future__ import division

import numpy as np

from pyanno.measures.distances import diagonal_distance
from pyanno.measures.helpers import (compute_nclasses,
                                     chance_agreement_same_frequency,
                                     observed_agreement_frequency,
                                     chance_adjusted_agreement,
                                     chance_agreement_different_frequency,
                                     confusion_matrix, coincidence_matrix, all_invalid)
from pyanno.util import labels_frequency, is_valid, PyannoValueError

import logging
logger = logging.getLogger(__name__)


def scotts_pi(annotations1, annotations2, nclasses=None):
    """Return Scott's pi statistic for two annotators.

    Assumes that the annotators draw random annotations with the same
    frequency as the combined observed annotations.

    Allows for missing values.

    Scott 1955
    http://en.wikipedia.org/wiki/Scott%27s_Pi
    """

    if all_invalid(annotations1, annotations2):
        logger.debug('No valid annotations')
        return np.nan

    if nclasses is None:
        nclasses = compute_nclasses(annotations1, annotations2)

    chance_agreement = chance_agreement_same_frequency(annotations1,
                                                       annotations2,
                                                       nclasses)

    observed_agreement = observed_agreement_frequency(annotations1,
                                                      annotations2,
                                                      nclasses)

    return chance_adjusted_agreement(observed_agreement.sum(),
                                      chance_agreement.sum())


def cohens_kappa(annotations1, annotations2, nclasses=None):
    """Compute Cohen's kappa for two annotators.

    Assumes that the annotators draw annotations at random with different but
    constant frequencies.

    Cohen, Jacob (1960). A coefficient of agreement for nominal scales.
    Educational and Psychological Measurement, 20, 37--46.

    http://en.wikipedia.org/wiki/Cohen%27s_kappa
    """

    if all_invalid(annotations1, annotations2):
        logger.debug('No valid annotations')
        return np.nan

    if nclasses is None:
        nclasses = compute_nclasses(annotations1, annotations2)

    chance_agreement = chance_agreement_different_frequency(annotations1,
                                                            annotations2,
                                                            nclasses)

    observed_agreement = observed_agreement_frequency(annotations1,
                                                      annotations2,
                                                      nclasses)

    return chance_adjusted_agreement(observed_agreement.sum(),
                                     chance_agreement.sum())


def cohens_weighted_kappa(annotations1, annotations2,
                          weights_func = diagonal_distance,
                          nclasses=None):
    """Compute Cohen's weighted kappa for two annotators.

    Assumes that the annotators draw annotations at random with different but
    constant frequencies. Disagreements are weighted by a weights
    w_ij representing the "seriousness" of disagreement. For ordered codes,
    it is often set to the distance from the diagonal, i.e. w_ij = |i-j| .

    When w_ij is 0.0 on the diagonal and 1.0 elsewhere,
    Cohen's weighted kappa is equivalent to Cohen's kappa.


    Input:

    annotations1, annotations2 -- array of annotations; classes are
        indicated by non-negative integers, -1 indicates missing values

    weights_func -- weights function that receives two matrices of classes
        i, j and returns the matrix of weights between them.
        Default is `diagonal_distance`

    nclasses -- number of classes in the annotations. If None, `nclasses` is
        inferred from the values in the annotations


    Output:

    kappa -- Cohens' weighted kappa


    See also:
    `diagonal_distance`, `binary_distance`, `cohens_kappa`

    Cohen 1968
    http://en.wikipedia.org/wiki/Cohen%27s_kappa
    """

    if all_invalid(annotations1, annotations2):
        logger.debug('No valid annotations')
        return np.nan

    if nclasses is None:
        nclasses = compute_nclasses(annotations1, annotations2)

    # observed probability of each combination of annotations
    observed_freq = confusion_matrix(annotations1, annotations2, nclasses)
    observed_freq_sum = observed_freq.sum()
    if observed_freq_sum == 0:
        return np.nan

    observed_freq /= observed_freq_sum

    # expected probability of each combination of annotations if annotators
    # draw annotations at random with different but constant frequencies
    freq1 = labels_frequency(annotations1, nclasses)
    freq2 = labels_frequency(annotations2, nclasses)
    chance_freq = np.outer(freq1, freq2)

    # build weights matrix from weights function
    weights = np.fromfunction(weights_func, shape=(nclasses, nclasses),
                              dtype=float)

    kappa = 1. - (weights*observed_freq).sum() / (weights*chance_freq).sum()

    return kappa


def fleiss_kappa(annotations, nclasses=None):
    """Compute Fleiss' kappa.

    http://en.wikipedia.org/wiki/Fleiss%27_kappa
    """

    if all_invalid(annotations):
        logger.debug('No valid annotations')
        return np.nan

    if nclasses is None:
        nclasses = compute_nclasses(annotations)

    # transform raw annotations into the number of annotations per class
    # for each item
    nitems = annotations.shape[0]
    nannotations = np.zeros((nitems, nclasses))
    for k in range(nclasses):
        nannotations[:,k] = (annotations==k).sum(1)

    return _fleiss_kappa_nannotations(nannotations)


def _fleiss_kappa_nannotations(nannotations):
    """Compute Fleiss' kappa gien number of annotations per class format.

    This is a testable helper for fleiss_kappa.
    """

    nitems = nannotations.shape[0]

    # check that all rows are annotated by the same number of annotators
    _nanno_sum = nannotations.sum(1)
    nannotations_per_item = _nanno_sum[0]
    if not np.all(_nanno_sum == nannotations_per_item):
        raise PyannoValueError(
            'Number of annotations per item is not constant.'
        )

    # empirical frequency of categories
    freqs = nannotations.sum(0) / (nitems*nannotations_per_item)
    chance_agreement = (freqs**2.).sum()

    # annotator agreement for i-th item, relative to possible annotators pairs
    agreement_rate = (((nannotations**2.).sum(1) - nannotations_per_item)
                      / (nannotations_per_item*(nannotations_per_item-1.)))
    observed_agreement = agreement_rate.mean()

    return chance_adjusted_agreement(observed_agreement, chance_agreement)


def krippendorffs_alpha(annotations, metric_func=diagonal_distance,
                        nclasses=None):
    """Compute Krippendorff's alpha.

    Input:

    annotations -- array of annotations; classes are
        indicated by non-negative integers, -1 indicates missing values

    weights_func -- weights function that receives two matrices of classes
        i, j and returns the matrix of weights between them.
        Default is `diagonal_distance`

    nclasses -- number of classes in the annotations. If None, `nclasses` is
        inferred from the values in the annotations


    Output:

    alpha -- Krippendorff's alpha


    See also:
    `diagonal_distance`, `binary_distance`


    Klaus Krippendorff (2004). Content Analysis, an Introduction to Its
    Methodology, 2nd Edition. Thousand Oaks, CA: Sage Publications.
    In particular, Chapter 11, pages 219--250.

    http://en.wikipedia.org/wiki/Krippendorff%27s_Alpha
    """

    if all_invalid(annotations):
        logger.debug('No valid annotations')
        return np.nan

    if nclasses is None:
        nclasses = compute_nclasses(annotations)

    coincidences = coincidence_matrix(annotations, nclasses)

    nc = coincidences.sum(1)
    n = coincidences.sum()

    # ---- coincidences expected by chance
    chance_coincidences = np.empty((nclasses, nclasses), dtype=float)
    for c in range(nclasses):
        for k in range(nclasses):
            if c == k:
                chance_coincidences[c,k] = nc[c]*(nc[k]-1.) / (n-1.)
            else:
                chance_coincidences[c,k] = nc[c]*nc[k] / (n-1.)

    # build weights matrix from weights function
    weights = np.fromfunction(metric_func, shape=(nclasses, nclasses),
                              dtype=float) ** 2.

    alpha = 1. - ((weights*coincidences).sum()
                  / (weights*chance_coincidences).sum())

    return alpha


