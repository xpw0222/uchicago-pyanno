"""Define standard reliability measures."""

from __future__ import division
import numpy as np
from pyanno.util import benchmark, labels_count, labels_frequency

# TODO: functions to compute confidence interval


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

    conf_mat = np.empty((nclasses, nclasses), dtype=np.long)
    for i in range(nclasses):
        for j in range(nclasses):
            conf_mat[i, j] = np.sum(np.logical_and(annotations1 == i,
                                                   annotations2 == j))

    return conf_mat


def _chance_adjusted_agreement(observed_agreement, chance_agreement):
    """Return the chance-adjusted agreement given the specified agreement
    and expected agreement.

    Defined by (observed_agreement - chance_agreement)/(1.0 - chance_agreement)

    Input:
    observed_agreement -- agreement
    chance_agreement -- expected agreement
    """

    return (observed_agreement - chance_agreement) / (1. - chance_agreement)


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

    Assumes that the annotators draw random annotations with the same
    frequency as the combined observed annotations.
    """

    freq1 = labels_frequency(annotations1, nclasses)
    freq2 = labels_frequency(annotations2, nclasses)

    chance_agreement = freq1 * freq2
    return chance_agreement


def observed_agreement_frequency(annotations1, annotations2, nclasses):
    """Observed frequency of agreement by two annotators.

    If a category is never observed, the frequency for that category is set
    to 0.0 .

    Only count entries where both annotators responded toward observed
    frequency.
    """

    conf_mat = confusion_matrix(annotations1, annotations2, nclasses)
    observed_agreement = conf_mat.diagonal() / conf_mat.sum()
    return observed_agreement


def _compute_nclasses(*annotations):
    max_ = np.amax(map(np.amax, annotations))
    return max_ + 1


def scotts_pi(annotations1, annotations2, nclasses=None):
    """Return Scott's pi statistic for two annotators.

    Assumes that the annotators draw random annotations with the same
    frequency as the combined observed annotations.

    Allows for missing values.

    Scott 1955
    http://en.wikipedia.org/wiki/Scott%27s_Pi
    """

    if nclasses is None:
        nclasses = _compute_nclasses(annotations1, annotations2)

    chance_agreement = chance_agreement_same_frequency(annotations1,
                                                       annotations2,
                                                       nclasses)

    observed_agreement = observed_agreement_frequency(annotations1,
                                                      annotations2,
                                                      nclasses)

    return _chance_adjusted_agreement(observed_agreement.sum(),
                                      chance_agreement.sum())


def cohens_kappa(annotations1, annotations2, nclasses=None):
    """Compute Cohen's kappa for two annotators.

    Assumes that the annotators draw annotations at random with different but
     constant frequencies.

    Cohen 1960
    http://en.wikipedia.org/wiki/Cohen%27s_kappa
    """

    if nclasses is None:
        nclasses = _compute_nclasses(annotations1, annotations2)

    chance_agreement = chance_agreement_different_frequency(annotations1,
                                                            annotations2,
                                                            nclasses)

    observed_agreement = observed_agreement_frequency(annotations1,
                                                      annotations2,
                                                      nclasses)

    return _chance_adjusted_agreement(observed_agreement.sum(),
                                      chance_agreement.sum())

def fleiss_kappa(annotations, nclasses=None):
    """Compute Fleiss' kappa.

    http://en.wikipedia.org/wiki/Fleiss%27_kappa
    """

    if nclasses is None:
        nclasses = _compute_nclasses(annotations)

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
        raise ValueError('Number of annotations per item should be constant.')

    # empirical frequency of categories
    freqs = nannotations.sum(0) / (nitems*nannotations_per_item)
    chance_agreement = (freqs**2.).sum()

    # annotator agreement for i-th item, relative to possible annotators pairs
    agreement_rate = (((nannotations**2.).sum(1) - nannotations_per_item)
                      / (nannotations_per_item*(nannotations_per_item-1.)))
    observed_agreement = agreement_rate.mean()

    return _chance_adjusted_agreement(observed_agreement, chance_agreement)
