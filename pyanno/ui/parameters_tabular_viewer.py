# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

from traits.api import HasTraits, List, Str
from traits.trait_types import Int
from traitsui.api import View, Group, TabularEditor, OKButton, Item
from pyanno.ui.arrayview import Array2DAdapter


WIDTH_CELL = 80
HEIGHT_CELL = 20
MAX_WIDTH = 800
MAX_HEIGHT = 800
W_MARGIN = 50
H_MARGIN = 150

class ParametersTabularView(HasTraits):
    """Tabular view of a set of parameters (not editable).
    """

    # 2D data to be displayed
    data = List

    # title of the view window
    title = Str

    # format of displayed data
    format = Str('%.4f')

    def traits_view(self):
        ncolumns = len(self.data[0])
        nrows = len(self.data)
        w_table = WIDTH_CELL * ncolumns
        h_table = HEIGHT_CELL * nrows
        w_view = min(w_table + W_MARGIN, MAX_WIDTH)
        h_view = min(h_table + H_MARGIN, MAX_HEIGHT)
        return View(
            Group(Item('data',
                       editor=TabularEditor(
                           adapter=Array2DAdapter(ncolumns=ncolumns,
                                                  format=self.format,
                                                  show_index=False,
                                                  count_from_one=False),
                           editable=False
                       ),
                       width = w_table,
                       height = h_table,
                       padding = 10,
                       show_label=False),
            ),
            title     = self.title,
            width     = w_view,
            height    = h_view,
            resizable = True,
            buttons   = [OKButton]
            )
