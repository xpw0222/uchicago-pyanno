# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Apache license

"""Plot tools to save plots and copy their data inside to the clipboard.
"""

# Major library imports
import os.path

# Enthought library imports
from chaco.plot_graphics_context import PlotGraphicsContext
from chaco.tools.save_tool import SaveTool
from enable.base_tool import BaseTool
from traits.trait_types import Int, Any
from pyface.api import clipboard

import numpy as np

class SaveToolPlus(SaveTool):
    """Subclass of SaveTool that requests a filename and dpi before saving."""

    dpi = Int(300)

    def normal_key_pressed(self, event):
        """ Handles a key-press when the tool is in the 'normal' state.

        Saves an image of the plot if the keys pressed are Control and S.
        """

        if self.component is None: return

        if ((event.character == "s" or event.character == u'\x13')
            and event.control_down):
            # TODO: request name of file, dpi
            print 'triggered!!'

            if os.path.splitext(self.filename)[-1] == ".pdf":
                self._save_pdf()
            else:
                self._save_raster()
            event.handled = True

    def _save_raster(self):
        """ Saves an image of the component."""
        self.component.do_layout(force=True)
        # NOTE saving only works properly when dpi is a multiple of 72
        gc = PlotGraphicsContext((int(self.component.outer_width),
                                  int(self.component.outer_height)),
                                 dpi=np.ceil(self.dpi / 72.0)*72)
        gc.render_component(self.component)
        gc.save(self.filename)


class CopyDataToClipboardTool(BaseTool):
    """Tool that copies the plot's data to the clipboard."""

    # TODO allow apple-c key combination

    data = Any

    def normal_key_pressed(self, event):
        """ Handles a key-press when the tool is in the 'normal' state.

        Saves an image of the plot if the keys pressed are Control and S.
        """

        if self.component is None: return

        if ((event.character == "c" or event.character == u'\x03')
            and event.control_down):
            clipboard.data = repr(self.data)

            print 'triggered copy!!'

            event.handled = True
