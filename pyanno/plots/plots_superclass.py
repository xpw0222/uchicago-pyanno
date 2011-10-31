# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Apache license

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
        """Add title and Copy and Save tools."""

        if self.title is not None:
            plot.title = self.title

        save_tool = SaveToolPlus(component=plot)
        copy_tool = CopyDataToClipboardTool(component=plot, data=data)

        plot.tools.append(save_tool)
        plot.tools.append(copy_tool)


    def _remove_grid_and_axes(self, plot):
        # remove grids and axes
        plot.underlays = []


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

    traits_view = View(
        VGroup(
            Include('traits_plot_item'),
            Include('instructions_group'),
        ),
        resizable = True
    )
