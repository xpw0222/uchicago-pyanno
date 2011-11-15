# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

from chaco.label_axis import LabelAxis
from chaco.scales.scales import FixedScale
from chaco.scales_tick_generator import ScalesTickGenerator
import numpy as np

from traits.api import HasTraits, Str
from traitsui.api import HGroup, Spring, Item, View, VGroup, Include
from pyanno.plots.plot_tools import SaveToolPlus, CopyDataToClipboardTool


class PyannoPlotContainer(HasTraits):

    #### plot-related traits
    title = Str

    instructions = Str
    def _instructions_default(self):
        import sys
        if sys.platform == 'darwin':
            return 'Cmd-S: Save, Cmd-C: Copy data'
        else:
            return 'Ctrl-S: Save,  Ctrl-C: Copy data'


    def decorate_plot(self, plot, data):
        """Add Copy and Save tools."""

        save_tool = SaveToolPlus(component=plot)
        copy_tool = CopyDataToClipboardTool(component=plot, data=data)

        plot.tools.append(save_tool)
        plot.tools.append(copy_tool)


    def _set_title(self, plot):
        if self.title is not None:
            plot.title = self.title


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


    def _remove_grid_and_axes(self, plot):
        """Remove grids and axes."""
        plot.underlays = []


    def _add_index_axis(self, plot, axis):
        plot.index_axis = axis
        plot.underlays.append(axis)


    def _add_value_axis(self, plot, axis):
        plot.value_axis = axis
        plot.underlays.append(axis)


    #### View definitions #####################################################

    instructions_group = HGroup(
        Spring(),
        Item('instructions', style='readonly', show_label=False),
        Spring()
    )

    resizable_view = View(
        VGroup(
            Include('resizable_plot_item'),
            Include('instructions_group'),
            ),
        resizable = True
    )

    def traits_view(self):
        traits_view = View(
            VGroup(
                self.traits_plot_item,
                Include('instructions_group'),
                ),
            resizable = True
        )
        return traits_view
