# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Apache license
from chaco.array_plot_data import ArrayPlotData
from chaco.cmap_image_plot import CMapImagePlot
from chaco.data_range_1d import DataRange1D
from chaco.default_colormaps import jet, YlOrRd, Reds, BuPu
from chaco.label_axis import LabelAxis
from chaco.plot import Plot
from chaco.scales.scales import FixedScale
from chaco.scales_tick_generator import ScalesTickGenerator
from enable.component_editor import ComponentEditor

from tables.array import Array
from traits.trait_types import Str, Instance, Float, Any, Int
from traitsui.group import VGroup, HGroup
from traitsui.include import Include
from traitsui.item import Item
from traitsui.view import View

from pyanno.plots.plots_superclass import PyannoPlotContainer
import numpy as np


class PosteriorPlot(PyannoPlotContainer):
    # data to be displayed
    pasterior = Array

    ### plot-related traits
    plot_width = Float(250)
    plot_height = Float(200)

    colormap_low = Float(0.0)
    colormap_high = Float(1.0)

    origin = Str('top left')

    posterior_plot = Instance(Plot)

    #matrix_plot_container = Instance(HPlotContainer)

    def _create_colormap(self):
        if self.colormap_low is None:
            self.colormap_low = self.posterior.min()

        if self.colormap_high is None:
            self.colormap_high = self.posterior.max()

        colormap = jet(DataRange1D(low=self.colormap_low,
                                   high=self.colormap_high))

        return colormap


    def _create_increment_one_axis(self, plot, start, number, orientation,
                                   ticks=None):
        """Create axis with ticks at a distance of one units.

        Parameters
        ----------
        plot : Plot
            plot where the axis will be attached
        start : float
            position of first tick
        number : int
            number of ticks
        orientation: ['top', 'bottom', 'left', 'right']
            position of axis on the plot
        ticks : list of strings
            string to be displayed for each tick
        """

        ids = start + np.arange(0, number)
        if ticks is None:
            ticks = [str(idx) for idx in np.arange(0, number)]

        axis = LabelAxis(
            plot,
            orientation = orientation,
            positions = ids,
            labels = ticks,
            label_rotation = 0
        )

        # use a FixedScale tick generator with a resolution of 1
        axis.tick_generator = ScalesTickGenerator(scale=FixedScale(1.))

        return axis


    def _posterior_plot_default(self):
        data = self.posterior
        nannotations, nclasses = data.shape

        # create a plot data object
        plot_data = ArrayPlotData()
        plot_data.set_data("values", data)

        # create the plot
        plot = Plot(plot_data, origin=self.origin)

        img_plot = plot.img_plot("values",
                                 interpolation='nearest',
                                 xbounds=(0, nclasses),
                                 ybounds=(0, nannotations),
                                 colormap=self._create_colormap())[0]

        self._remove_grid_and_axes(plot)

        # create x axis for labels
        label_axis = self._create_increment_one_axis(plot, 0.5, nclasses, 'top')
        plot.index_axis = label_axis
        plot.underlays.append(label_axis)

        # create y axis for values
        value_axis_ticks = [str(id) for id in range(nannotations-1, -1, -1)]
        value_axis = self._create_increment_one_axis(plot, 0.5, nannotations,
                                                     'left', value_axis_ticks)
        plot.value_axis = value_axis
        plot.underlays.append(value_axis)

        plot.aspect_ratio = float(nclasses) / nannotations
        self.plot_height = int(self.plot_width / plot.aspect_ratio)

        self.decorate_plot(plot, self.posterior)
        return plot


    def add_markings(self, classes):
        pass


    def _create_resizable_view(self):
        # resizable_view factory, as I need to compute the height of the plot
        # from the number of annotations, and I couldn't find any other way to
        # do that

        # "touch" posterior_plot to have it initialize
        self.posterior_plot

        resizable_plot_item = (
            Item(
                'posterior_plot',
                editor=ComponentEditor(),
                resizable=True,
                show_label=False,
                width = self.plot_width,
                height = self.plot_height,
            )
        )

        resizable_view = View(
            VGroup(
                Include('instructions_group'),
                resizable_plot_item,
            ),
            width = 450,
            height = 800,
            scrollable = True,
            resizable = True
        )

        return resizable_view


#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    import numpy as np

    matrix = np.random.random(size=(200, 5))
    matrix = matrix / matrix.sum(1)[:,None]
    matrix[0,0] = 1.

    matrix_view = PosteriorPlot(posterior=matrix, title='Debug plot_posterior')
    resizable_view = matrix_view._create_resizable_view()

    matrix_view.edit_traits(view=resizable_view)

    return matrix_view, resizable_view


if __name__ == '__main__':
    mv, rv = main()
