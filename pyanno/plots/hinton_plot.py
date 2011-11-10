# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)
from chaco.color_bar import ColorBar
from chaco.data_range_1d import DataRange1D

from chaco.data_range_2d import DataRange2D
from chaco.default_colormaps import Reds
from chaco.label_axis import LabelAxis
from chaco.linear_mapper import LinearMapper
from chaco.plot_containers import HPlotContainer
from chaco.scales.scales import FixedScale
from chaco.scales_tick_generator import ScalesTickGenerator
from chaco.api import ArrayPlotData, Plot
from chaco.base import n_gon

from enable.component_editor import ComponentEditor
from traits.has_traits import on_trait_change
from traits.trait_types import ListFloat, Any
from traits.api import Instance
from traitsui.item import Item

import numpy as np
from pyanno.plots.plot_tools import get_class_color
from pyanno.plots.plots_superclass import PyannoPlotContainer


class HintonDiagramPlot(PyannoPlotContainer):
    """Defines a Hinton diagram view of a discrete probability distribution.

    A Hinton diagram displays a probability distribution as a series of
    squares, with area proportional to the probability mass of each element.

    The plot updates automatically whenever the attribute `data` is modified.
    """

    #### Traits definition ####################################################

    # data to be displayed
    data = ListFloat

    #### plot-related traits
    plot_data = Instance(ArrayPlotData)
    plot = Any


    @on_trait_change('data', post_init=True)
    def _update_data(self):
        distr_len = len(self.data)
        plot_data = self.plot_data

        # centers of the squares
        centers = [(i, 0.5) for i in xrange(1, distr_len + 1)]

        for idx, center in enumerate(centers):
            # draw square with area proportional to probability mass
            r = np.sqrt(self.data[idx] / (2.*np.pi))
            npoints = n_gon(center=center, r=r, nsides=40)
            nxarray, nyarray = np.transpose(npoints)
            # save in dataarray
            plot_data.set_data('x%d' % idx, nxarray)
            plot_data.set_data('y%d' % idx, nyarray)


    def _plot_data_default(self):
        self.plot_data = ArrayPlotData()
        self._update_data()
        return self.plot_data


    #### Plot definition ######################################################

    def _create_probability_axis(self, plot):
        """Create plot axis for probability values."""
        prob_axis = LabelAxis(
            plot,
            orientation='left',
            positions=[0.5, 0.5 + np.sqrt(0.25) / 2., 1.0],
            labels=['0', '0.25', '1']
        )
        prob_axis.tick_generator = ScalesTickGenerator(scale=FixedScale(0.001))
        return prob_axis


    def _plot_default(self):
        distr_len = len(self.data)

        # colormap for probability values
        cm = Reds(DataRange1D(low=0., high=1.))
        data_color_indices = cm.map_index(np.asarray(self.data))

        # PolygonPlot holding the circles of the Hinton diagram
        polyplot = Plot(self.plot_data)
        for idx in range(distr_len):
            p = polyplot.plot(('x%d' % idx, 'y%d' % idx),
                          type="polygon",
                          face_color=cm.color_bands[data_color_indices[idx]],
                          edge_color='black',
                          color_mapper=cm)

        self._set_title(polyplot)
        self._remove_grid_and_axes(polyplot)

        # create x axis for labels
        axis = self._create_increment_one_axis(polyplot, 1., distr_len, 'bottom')
        self._add_index_axis(polyplot, axis)

        # create y axis for probability density
        #prob_axis = self._create_probability_axis(polyplot)
        #polyplot.value_axis = prob_axis
        #polyplot.underlays.append(prob_axis)

        # tweak some of the plot properties
        range2d = DataRange2D(low=(0.5, 0.), high=(distr_len+0.5, 1.))
        polyplot.range2d = range2d
        polyplot.aspect_ratio = ((range2d.x_range.high - range2d.x_range.low)
                                 / (range2d.y_range.high - range2d.y_range.low))

        polyplot.border_visible = False
        polyplot.padding = [0, 0, 25, 25]

        # add colorbar
        colorbar = ColorBar(index_mapper = LinearMapper(range=cm.range),
                            color_mapper = cm,
                            plot = p[0],
                            orientation = 'v',
                            resizable = '',
                            width = 10,
                            height = 80)
        colorbar.padding_bottom = polyplot.padding_bottom
        colorbar.padding_top = polyplot.padding_top
        colorbar.padding_left = 40
        colorbar.padding_right = 0

        # create a container to position the plot and the colorbar side-by-side
        container = HPlotContainer(use_backbuffer=True, valign='center')
        container.add(colorbar)
        container.add(polyplot)
        container.bgcolor = 0xFFFFFF # light gray: 0xEEEEEE

        self.decorate_plot(container, self.data)
        return container


    #### View definition ######################################################

    resizable_plot_item = Item(
        'plot',
        editor=ComponentEditor(),
        resizable=True,
        show_label=False,
        width = 600,
        height = 400
        )

    traits_plot_item = Item(
        'plot',
        editor=ComponentEditor(),
        resizable=False,
        show_label=False,
        height=-110,
        )


def plot_hinton_diagram(data, **kwargs):
    """Create and display a Chaco plot of a Hinton diagram.

    The component allows saving the plot (with Ctrl-S), and copying the matrix
    data to the clipboard (with Ctrl-C).

    Keyword arguments:
    title -- title for the resulting plot
    """
    hinton_diagram = HintonDiagramPlot(data=data, **kwargs)
    hinton_diagram.edit_traits(view='resizable_view')
    return hinton_diagram



#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    data = np.random.random(5)
    data /= data.sum()
    hinton_view = plot_hinton_diagram(data.tolist(),
                                      title='Debug plot_hinton_diagram')
    hinton_view.configure_traits()
    return hinton_view


if __name__ == '__main__':
    hv = main()
