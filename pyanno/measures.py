"""Define standard reliability measures."""

from __future__ import division
import numpy as np
from pyanno.util import benchmark, labels_count

# TODO: functions to compute confidence interval


def confusion_matrix(annotations1, annotations2, nclasses):
    """Compute confusion matrix to evaluate the accuracy of a classification.

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


def chance_agreement_frequency(annotations1, annotations2, nclasses):
    """Excepted frequency of agreement by random annotations.

    Assumes that the annotators draw random annotations with the same
    frequency as the observed combined annotations.
    """

    count1 = labels_count(annotations1, nclasses)
    count2 = labels_count(annotations2, nclasses)

    count_total = count1 + count2
    total = count_total.sum()
    chance_agreement = (count_total / total) ** 2.

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


def _compute_nclasses(annotations):
    return annotations.max() + 1


def scotts_pi(annotations1, annotations2, nclasses=None):
    """Return Scott's pi statistic for two annotators.

    Allows for missing values.

    http://en.wikipedia.org/wiki/Scott%27s_Pi
    """

    if nclasses is None:
        nclasses = max(_compute_nclasses(annotations1),
                       _compute_nclasses(annotations2))

    chance_agreement = chance_agreement_frequency(annotations1,
                                                  annotations2,
                                                  nclasses)

    observed_agreement = observed_agreement_frequency(annotations1,
                                                      annotations2,
                                                      nclasses)

    return _chance_adjusted_agreement(observed_agreement.sum(),
                                      chance_agreement.sum())
