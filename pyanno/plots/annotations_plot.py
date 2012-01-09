# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

from chaco.array_plot_data import ArrayPlotData
from chaco.color_bar import ColorBar
from chaco.data_range_1d import DataRange1D
from chaco.default_colormaps import Reds
from chaco.linear_mapper import LinearMapper
from chaco.plot import Plot
from chaco.plot_containers import HPlotContainer
from chaco.tools.pan_tool import PanTool
from enable.component_editor import ComponentEditor
from traits.trait_numeric import Array

from traits.trait_types import Str, Instance, Float
from traitsui.group import VGroup, HGroup
from traitsui.include import Include
from traitsui.item import Item, Spring
from traitsui.view import View

from pyanno.ui.appbase.wx_utils import is_display_small
from pyanno.plots.plots_superclass import PyannoPlotContainer
from pyanno.util import is_valid

import numpy as np


class PosteriorPlot(PyannoPlotContainer):
    # data to be displayed
    posterior = Array

    ### plot-related traits
    plot_width = Float
    def _plot_width_default(self):
        return 450 if is_display_small() else 500

    plot_height = Float
    def _plot_height_default(self):
        return 600 if is_display_small() else 750

    colormap_low = Float(0.0)
    colormap_high = Float(1.0)

    origin = Str('top left')

    plot_container = Instance(HPlotContainer)
    plot_posterior = Instance(Plot)

    def _create_colormap(self):
        if self.colormap_low is None:
            self.colormap_low = self.posterior.min()

        if self.colormap_high is None:
            self.colormap_high = self.posterior.max()

        colormap = Reds(DataRange1D(low=self.colormap_low,
                                   high=self.colormap_high))

        return colormap


    def _plot_container_default(self):
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
        ndisp = 55
        img_plot.y_mapper.range.high = ndisp
        img_plot.y_mapper.domain_limits=((0, nannotations))

        self._set_title(plot)
        plot.padding_top = 80

        # create x axis for labels
        label_axis = self._create_increment_one_axis(plot, 0.5, nclasses, 'top')
        label_axis.title = 'classes'
        self._add_index_axis(plot, label_axis)

        plot.y_axis.title = 'items'

        # tweak plot aspect
        goal_aspect_ratio = 2.0
        plot_width = (goal_aspect_ratio * self.plot_height
                      * nclasses / ndisp)
        self.plot_width = min(max(plot_width, 200), 400)
        plot.aspect_ratio = self.plot_width / self.plot_height

        # add colorbar
        colormap = img_plot.color_mapper
        colorbar = ColorBar(index_mapper = LinearMapper(range=colormap.range),
                            color_mapper = colormap,
                            plot = img_plot,
                            orientation = 'v',
                            resizable = '',
                            width = 15,
                            height = 250)
        colorbar.padding_top = plot.padding_top
        colorbar.padding_bottom = int(self.plot_height - colorbar.height -
                                      plot.padding_top)
        colorbar.padding_left = 0
        colorbar.padding_right = 30


        # create a container to position the plot and the colorbar side-by-side
        container = HPlotContainer(use_backbuffer=True)
        container.add(plot)
        container.add(colorbar)
        container.bgcolor = 0xFFFFFF # light gray: 0xEEEEEE

        # add pan tools
        img_plot.tools.append(PanTool(img_plot, constrain=True,
                                      constrain_direction="y", speed=7.))

        self.decorate_plot(container, self.posterior)
        self.plot_posterior = plot
        return container


    def add_markings(self, mark_classes, mark_name, marker_shape,
                     delta_x, delta_y, marker_size=5, line_width=1.,
                     marker_color='white'):
        plot = self.plot_posterior
        nannotations = plot.data.arrays['values'].shape[0]

        y_name = mark_name + '_y'
        x_name = mark_name + '_x'

        valid = is_valid(mark_classes)
        y_values = np.arange(nannotations)[valid] + delta_y + 0.5
        x_values = mark_classes[valid].astype(float) + delta_x + 0.5

        plot.data.set_data(y_name, y_values)
        plot.data.set_data(x_name, x_values)

        plot.plot((x_name, y_name), type='scatter', name=mark_name,
                  marker=marker_shape, marker_size=marker_size,
                  color='transparent',
                  outline_color=marker_color, line_width=line_width)


    def remove_markings(self, mark_name):
        self.plot_posterior.delplot(mark_name)


    def _create_resizable_view(self):
        # resizable_view factory, as I need to compute the height of the plot
        # from the number of annotations, and I couldn't find any other way to
        # do that

        # "touch" posterior_plot to have it initialize
        self.plot_container

        if is_display_small():
            height = 760
        else:
            height = 800

        resizable_plot_item = (
            Item(
                'plot_container',
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
            height = height,
            resizable = True
        )

        return resizable_view


    def traits_view(self):
        return self._create_resizable_view()


    pan_instructions = Str
    def _pan_instructions_default(self):
        return 'Left-click and drag to navigate items'

    instructions_group = VGroup(
        HGroup(
            Spring(),
            Item('instructions', style='readonly', show_label=False),
            Spring()
        ),
        HGroup(
            Spring(),
            Item('pan_instructions', style='readonly', show_label=False),
            Spring()
        )
    )


def plot_posterior(posterior, show_maximum=False, **kwargs):
    """Display a plot of the posterior distribution over classes.

    This function is used together with the `infer_labels` method offered by
    all models, e.g.: ::

    >>> from pyanno.models import ModelB
    >>> from pyanno.plots import plot_posterior
    >>> # create a new model with 3 classes and 6 annotators
    >>> model = ModelB.create_initial_state(3, 6)
    >>> annotations = model.generate_annotations(100)
    >>> # compute the posterior distribution over class labels
    >>> posterior = model.infer_labels(annotations)
    >>> # plot the distribution in a window
    >>> plot_posterior(posterior)

    Arguments
    ---------
    posterior : ndarray, shape=(n_annotations, n_classes)
        posterior[i,:] is the posterior distribution over classes for the
        i-th annotation.

    show_maximum : bool
        if True, indicate the position of the maxima with white circles

    kwargs : dictionary
        Additional keyword arguments passed to the plot. The argument 'title'
        sets the title of the plot.
    """
    post_view = PosteriorPlot(posterior=posterior, **kwargs)
    resizable_view = post_view._create_resizable_view()
    post_view.edit_traits(view=resizable_view)

    if show_maximum:
        maximum = posterior.argmax(1)
        post_view.add_markings(maximum, 'maximum',
                               'circle', 0., 0., marker_size=4,
                               marker_color='blue')

    return post_view


#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    import numpy as np

    matrix = np.random.random(size=(20000, 5))
    matrix = matrix / matrix.sum(1)[:,None]
    matrix[0,0] = 1.

    matrix_view = plot_posterior(matrix, show_maximum=True, title='TEST')
    matrix_view.configure_traits()
    return matrix_view


if __name__ == '__main__':
    mv = main()
