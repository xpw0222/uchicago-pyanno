# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Apache license

from traits.api import HasTraits, List, Str
from traitsui.api import View, Group, TabularEditor, OKButton, Item
from pyanno.ui.arrayview import Array2DAdapter


class ParametersTabularView(HasTraits):
    """Tabular view of a set of parameters (not editable).
    """

    # 2D data to be displayed
    data = List

    # title of the view window
    title = Str

    def traits_view(self):
        return View(
            Group(Item('data',
                       editor=TabularEditor(
                           adapter=Array2DAdapter(ncolumns=len(self.data[0]),
                                                  format='%.4f',
                                                  show_index=False),
                           editable=False
                       ),
                       show_label=False)),
            title     = self.title,
            width     = 500,
            height    = 200,
            resizable = True,
            buttons   = [OKButton]
            )
