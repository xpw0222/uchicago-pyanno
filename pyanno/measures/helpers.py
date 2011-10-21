from __future__ import division

import numpy as np
from pyanno.util import labels_count, labels_frequency, is_valid


def pairwise_matrix(pairwise_statistic, annotations, *args, **kwargs):
    """Compute the matrix of all combinations of a pairwise statistics.

    This function applies an agreement or covariation statistic that is only
    defined for pairs of annotators to all combinations of annotators pairs,
    and returns a matrix of the result.
    """

    nannotators = annotations.shape[1]

    pairwise = np.empty((nannotators, nannotators), dtype=float)
    for i in range(nannotators):
        for j in range(nannotators):
            pairwise[i,j] = pairwise_statistic(annotations[:,i],
                                               annotations[:,j],
                                               *args, **kwargs)

    return pairwise


def confusion_matrix(annotations1, annotations2, nclasses):
    """Compute confusion matrix from pairs of annotations.

    Labels are numbers between 0 and `nclasses`. Any other value is
    considered a missing value.

    Parameters
    ----------
    annotations1 : array, shape = [n_samples]
    annotations2 : array, shape = [n_samples]
    nclasses

    Returns
    -------
    conf_mat : array, shape = [nclasses, nclasses]
        confusion matrix; conf_mat[i,j] = number of observations that was
        annotated as category `i` by annotator 1 and as `j` by annotator 2

    References
    ----------
    http://en.wikipedia.org/wiki/Confusion_matrix
    """

    conf_mat = np.empty((nclasses, nclasses), dtype=float)
    for i in range(nclasses):
        for j in range(nclasses):
            conf_mat[i, j] = np.sum(np.logical_and(annotations1 == i,
                                                   annotations2 == j))

    return conf_mat


def coincidence_matrix(annotations, nclasses):
    """Build coincidence matrix."""

    # total number of annotations in row
    nannotations = is_valid(annotations).sum(1).astype(float)
    valid = nannotations > 1

    nannotations = nannotations[valid]
    annotations = annotations[valid,:]

    # number of annotations of class c in row
    nc_in_row = np.empty((nannotations.shape[0], nclasses), dtype=int)
    for c in range(nclasses):
        nc_in_row[:, c] = (annotations == c).sum(1)

    coincidences = np.empty((nclasses, nclasses), dtype=float)
    for c in range(nclasses):
        for k in range(nclasses):
            if c==k:
                nck_pairs = nc_in_row[:, c] * (nc_in_row[:, c] - 1)
            else:
                nck_pairs = nc_in_row[:, c] * nc_in_row[:, k]
            coincidences[c, k] = (nck_pairs / (nannotations - 1.)).sum()

    return coincidences


def chance_adjusted_agreement(observed_agreement, chance_agreement):
    """Return the chance-adjusted agreement given the specified agreement
    and expected agreement.

    Defined by (observed_agreement - chance_agreement)/(1.0 - chance_agreement)

    Input:
    observed_agreement -- agreement
    chance_agreement -- expected agreement
    """

    return (observed_agreement - chance_agreement) / (1. - chance_agreement)


def observed_agreement_frequency(annotations1, annotations2, nclasses):
    """Observed frequency of agreement by two annotators.

    If a category is never observed, the frequency for that category is set
    to 0.0 .

    Only count entries where both annotators responded toward observed
    frequency.
    """

    conf_mat = confusion_matrix(annotations1, annotations2, nclasses)
    conf_mat_sum = conf_mat.sum()
    if conf_mat_sum != 0:
        observed_agreement = conf_mat.diagonal() / conf_mat_sum
    else:
        observed_agreement = np.empty((nclasses,), dtype=float)
        observed_agreement.fill(np.nan)
    return observed_agreement


def chance_agreement_same_frequency(annotations1, annotations2, nclasses):
    """Expected frequency of agreement by random annotations.

    Assumes that the annotators draw random annotations with the same
    frequency as the combined observed annotations.
    """

    count1 = labels_count(annotations1, nclasses)
    count2 = labels_count(annotations2, nclasses)

    count_total = count1 + count2
    total = count_total.sum()
    chance_agreement = (count_total / total) ** 2.

    return chance_agreement


def chance_agreement_different_frequency(annotations1, annotations2, nclasses):
    """Expected frequency of agreement by random annotations.

    Assumes that the annotators draw annotations at random with different but
    constant frequencies.
    """

    freq1 = labels_frequency(annotations1, nclasses)
    freq2 = labels_frequency(annotations2, nclasses)

    chance_agreement = freq1 * freq2
    return chance_agreement


def compute_nclasses(*annotations):
    max_ = np.amax(map(np.amax, annotations))
    return max_ + 1
