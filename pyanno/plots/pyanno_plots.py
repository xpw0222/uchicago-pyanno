# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

from pyanno import measures
from pyanno.plots.matrix_plot import plot_square_matrix


def plot_pairwise_statistics(stat_func, annotations, nclasses=None, **kwargs):
    """Plot a matrix representation of a statistical function of annotations.

    The function `stat_func` is applied to the annotations of all pairs of
    annotators, and the resulting matrix is displayed in a window.

    Example:
    plot_pairwise_statistics(pyanno.measures.cohens_kappa, annotations)
    """

    if nclasses is None:
        measures.helpers.compute_nclasses(annotations)

    matrix = measures.pairwise_matrix(stat_func, annotations,
                                      nclasses=nclasses)

    kwargs_local = {'colormap_low': -1.0,
                    'colormap_high': 1.0}
    # add user kwargs, allowing for overwrite
    kwargs_local.update(kwargs)

    matrix_view = plot_square_matrix(matrix, **kwargs_local)
    return matrix_view



#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    import pyanno
    model = pyanno.ModelBt.create_initial_state(4)
    annotations = model.generate_annotations(model.generate_labels(100))
    mv = plot_pairwise_statistics(measures.cohens_kappa, annotations)
    return mv


if __name__ == '__main__':
    mv = main()
