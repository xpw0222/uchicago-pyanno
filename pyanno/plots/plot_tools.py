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
from enthought.etsconfig.api import ETSConfig

import numpy as np


def _is_control_down(key_event):
    """Return true if the Ctrl or Cmd key is down."""

    is_control_down = key_event.control_down

    if ETSConfig.toolkit == 'wx':
        # workaround for the fact that wxPython does not return True in
        # KeyEvent.ContrlDown() when the Cmd key is pressed on a Mac,
        # which is not what wx does (see
        # http://docs.wxwidgets.org/2.9.2/classwx_keyboard_state.html)

        # note that qt already does the right thing (i.e.,
        # control_down is true also for Mac's cmd key)
        is_control_down = (key_event.event.ControlDown()
                           or key_event.event.CmdDown())

    return is_control_down


class SaveToolPlus(SaveTool):
    """Subclass of SaveTool that requests a filename and dpi before saving."""

    dpi = Int(300)

    def normal_key_pressed(self, event):
        """ Handles a key-press when the tool is in the 'normal' state.

        Saves an image of the plot if the keys pressed are Control and S.
        """

        if self.component is None: return

        if ((event.character == "s" or event.character == u'\x13')
            and _is_control_down(event)):
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

    data = Any

    def normal_key_pressed(self, event):
        """ Handles a key-press when the tool is in the 'normal' state.

        Saves an image of the plot if the keys pressed are Control and S.
        """

        if self.component is None: return

        if ((event.character == "c" or event.character == u'\x03')
            and _is_control_down(event)):
            clipboard.data = repr(self.data)

            print 'triggered copy!!'

            event.handled = True
