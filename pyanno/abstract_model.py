# Copyright (c) 2011, Enthought, Ltd.
# Authors: Pietro Berkes <pberkes@enthought.com>, Andrey Rzhetsky
# License: Modified BSD license (2-clause)

"""Defines :class:`AbstractModel`, an abstract class that specifies the
interface of pyAnno models.
"""

from traits.has_traits import HasTraits
from traits.trait_types import Int
from pyanno.util import PyannoValueError, MISSING_VALUE
import numpy as np

class AbstractModel(HasTraits):
    """Abstract class defining the interface of a pyAnno model.
    """

    # number of label classes
    nclasses = Int

    # number of annotators per item
    nannotators = Int


    @staticmethod
    def create_initial_state(nclasses):
        """Factory method returning a model with random initial parameters.

        Arguments
        ---------
        nclasses : int
            Number of label classes
        """
        raise NotImplementedError()


    def generate_annotations(self, nitems):
        """Generate a random annotation set from the model.

        Sample a random set of annotations from the probability distribution
        defined the current model parameters.

        Arguments
        ---------
        nitems : int
            Number of items to sample

        Returns
        -------
        annotations : ndarray, shape = (n_items, n_annotators)
            annotations[i,j] is the annotation of annotator j for item i
        """
        raise NotImplementedError()


    def mle(self, annotations):
        """Computes maximum likelihood estimate (MLE) of parameters.

        Estimate the model parameters from a set of observed annotations
        using maximum likelihood estimation.

        Arguments
        ---------
        annotations : ndarray, shape = (n_items, n_annotators)
            annotations[i,j] is the annotation of annotator j for item i
        """
        raise NotImplementedError()


    def map(self, annotations):
        """Computes maximum a posteriori (MAP) estimate of parameters.

        Estimate the model parameters from a set of observed annotations
        using maximum a posteriori estimation.

        Arguments
        ---------
        annotations : ndarray, shape = (n_items, n_annotators)
            annotations[i,j] is the annotation of annotator j for item i
        """
        raise NotImplementedError()


    def log_likelihood(self, annotations):
        """Compute the log likelihood of a set of annotations given the model.

        Returns log P(annotations | current model parameters).

        Arguments
        ---------
        annotations : ndarray, shape = (n_items, n_annotators)
            annotations[i,j] is the annotation of annotator j for item i

        Returns
        -------
        log_lhood : float
            log likelihood of `annotations`
        """
        raise NotImplementedError()


    def _compute_total_nsamples(self, nsamples, burn_in_samples, thin_samples):
        """Compute the total number of samples to generate in order to return
        `nsamples` samples after burn-in and thinning.

        This helper function is typically called from the implementation
        of `samples_posterior_over_accuracy`.
        """
        return nsamples*thin_samples + burn_in_samples


    def _post_process_samples(self, samples, burn_in_samples, thin_samples):
        """Eliminate samples, discarding the first `burn_in_samples`,
        and thinning the rest.

        This helper function is typically called from the implementation
        of `samples_posterior_over_accuracy`.
        """
        return samples[burn_in_samples::thin_samples,:]


    def sample_posterior_over_accuracy(self, annotations, nsamples,
                                       burn_in_samples=0, thin_samples=1):
        """Return samples from posterior over the accuracy parameters.

        Draw samples from `P(accuracy parameters | data, model parameters)`.
        The accuracy parameters control the probability of an annotator
        reporting the correct label (the exact nature of these parameters
        varies from model to model).

        Arguments
        ---------
        annotations : ndarray, shape = (n_items, n_annotators)
            annotations[i,j] is the annotation of annotator j for item i

        nsamples : int
            Number of samples to return (i.e., burn-in and thinning samples
            are not included)

        burn_in_samples : int
            Discard the first `burn_in_samples` during the initial burn-in
            phase, where the Monte Carlo chain converges to the posterior

        thin_samples : int
            Only return one every `thin_samples` samples in order to reduce
            the auto-correlation in the sampling chain. This is called
            "thinning" in MCMC parlance.

        Returns
        -------
        samples : ndarray, shape = (n_samples, ??)
            Array of samples from the posterior distribution over parameters.
        """
        raise NotImplementedError()


    def infer_labels(self, annotations):
        """Infer posterior distribution over label classes.

         Compute the posterior distribution over label classes given observed
         annotations, :math:`P( \mathbf{y} | \mathbf{x})`.

         Arguments
         ----------
         annotations : ndarray, shape = (n_items, n_annotators)
             annotations[i,j] is the annotation of annotator j for item i

         Returns
         -------
         posterior : ndarray, shape = (n_items, n_classes)
             posterior[i,k] is the posterior probability of class k given the
             annotation observed in item i.
         """
        raise NotImplementedError()


    def are_annotations_compatible(self, annotations):
        """Returns True if the annotations are compatible with the model.

        The standard implementation is: valid if the number of annotators
        is correct, if the classes are between 0 and nclasses-1,
        and if missing values are marked with :attr:`pyanno.util.MISSING_VALUE`
        """

        masked_annotations = np.ma.masked_equal(annotations, MISSING_VALUE)

        if annotations.shape[1] != self.nannotators:
            return False

        if annotations.max() >= self.nclasses:
            return False

        if masked_annotations.min() < 0:
            return False

        return True


    def _raise_if_incompatible(self, annotations):
        if not self.are_annotations_compatible(annotations):
            raise PyannoValueError('Annotations are incompatible with model '
                                   'parameters')
