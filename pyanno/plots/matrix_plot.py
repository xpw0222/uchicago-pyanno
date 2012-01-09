# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

"""View of 2D matrix based on Chaco."""

from chaco.array_plot_data import ArrayPlotData
from chaco.color_bar import ColorBar
from chaco.data_range_1d import DataRange1D
from chaco.default_colormaps import jet, Reds, RdBu, reverse
from chaco.linear_mapper import LinearMapper
from chaco.plot import Plot
from chaco.plot_containers import HPlotContainer
from enable.component_editor import ComponentEditor

from traits.trait_numeric import Array
from traits.trait_types import Str, Instance, Float
from traitsui.item import Item

import numpy as np
from pyanno.plots.plots_superclass import PyannoPlotContainer


class MatrixPlot(PyannoPlotContainer):

    # data to be displayed
    matrix = Array

    #### plot-related traits
    colormap_low = Float(None)
    colormap_high = Float(None)
    origin = Str('top left')

    matrix_plot_container = Instance(HPlotContainer)

    def _create_colormap(self):
        if self.colormap_low is None:
            self.colormap_low = self.matrix.min()

        if self.colormap_high is None:
            self.colormap_high = self.matrix.max()

        if self.colormap_low >= 0.0:
            colormap_factory = Reds
        else:
            colormap_factory = reverse(RdBu)

        colormap = colormap_factory(
            DataRange1D(low=self.colormap_low, high=self.colormap_high)
        )

        return colormap


    def _matrix_plot_container_default(self):
        matrix = np.nan_to_num(self.matrix)
        width = matrix.shape[0]

        # create a plot data object and give it this data
        plot_data = ArrayPlotData()
        plot_data.set_data("values", matrix)

        # create the plot
        plot = Plot(plot_data, origin=self.origin)

        img_plot = plot.img_plot("values",
                                 interpolation='nearest',

                                 xbounds=(0, width),
                                 ybounds=(0, width),
                                 colormap=self._create_colormap())[0]

        #### fix axes
        self._remove_grid_and_axes(plot)

        axis = self._create_increment_one_axis(plot, 0.5, width, 'bottom')
        self._add_value_axis(plot, axis)

        axis = self._create_increment_one_axis(
            plot, 0.5, width, 'left',
            ticks=[str(i) for i in range(width-1, -1, -1)])
        self._add_index_axis(plot, axis)

        #### tweak plot attributes
        self._set_title(plot)
        plot.aspect_ratio = 1.
        # padding [left, right, up, down]
        plot.padding = [0, 0, 25, 25]

        # create the colorbar, handing in the appropriate range and colormap
        colormap = img_plot.color_mapper
        colorbar = ColorBar(index_mapper=LinearMapper(range=colormap.range),
                            color_mapper=colormap,
                            plot=img_plot,
                            orientation='v',
                            resizable='v',
                            width=20,
                            padding=[0, 20, 0, 0])
        colorbar.padding_top = plot.padding_top
        colorbar.padding_bottom = plot.padding_bottom

        # create a container to position the plot and the colorbar side-by-side
        container = HPlotContainer(use_backbuffer=True)
        container.add(plot)
        container.add(colorbar)
        container.bgcolor = 0xFFFFFF

        self.decorate_plot(container, self.matrix)
        return container


    resizable_plot_item = Item(
        'matrix_plot_container',
        editor=ComponentEditor(),
        resizable=True,
        show_label=False,
        width = 600,
        height = 400
        )

    traits_plot_item = Instance(Item)
    def _traits_plot_item_default(self):
        return Item(
                    'matrix_plot_container',
                    editor=ComponentEditor(),
                    resizable=False,
                    show_label=False,
                    height=-200,
                    width=-200
                    )


def plot_square_matrix(matrix, **kwargs):
    """Plot a square matrix using a color code.

    Arguments
    ---------
    matrix : ndarray, shape = (n, n)
        Square array of numberical values.

    kwargs : dictionary
        Additional keyword arguments passed to the plot. The argument `title`
        sets the title of the plot. The arguments `colormap_low` and
        `colormap_high` set the lowest and highest value for the color map.
    """
    matrix_view = MatrixPlot(matrix=matrix, **kwargs)
    matrix_view.edit_traits(view='resizable_view')
    return matrix_view




#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    matrix = np.eye(5)
    matrix_view = plot_square_matrix(matrix, title='Debug plot_matrix')
    matrix_view.configure_traits()
    return matrix_view


if __name__ == '__main__':
    mv = main()
