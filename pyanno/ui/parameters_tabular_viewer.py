# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

from traits.api import HasTraits, List, Str
from traits.trait_types import Int
from traitsui.api import View, Group, TabularEditor, OKButton, Item
from pyanno.ui.arrayview import Array2DAdapter


class ParametersTabularView(HasTraits):
    """Tabular view of a set of parameters (not editable).
    """

    # 2D data to be displayed
    data = List

    # title of the view window
    title = Str

    # format of displayed data
    format = Str('%.4f')

    # height of view
    height = Int(200)

    # width of view
    width = Int(500)

    def traits_view(self):
        return View(
            Group(Item('data',
                       editor=TabularEditor(
                           adapter=Array2DAdapter(ncolumns=len(self.data[0]),
                                                  format=self.format,
                                                  show_index=False),
                           editable=False
                       ),
                       show_label=False),
                  group_theme = 'white_theme.png',
            ),
            title     = self.title,
            width     = self.width,
            height    = self.height,
            resizable = True,
            buttons   = [OKButton]
            )
