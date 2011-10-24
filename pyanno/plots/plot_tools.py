# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Apache license

"""Plot tools to save plots and copy their data inside to the clipboard.
"""

from chaco.plot_graphics_context import PlotGraphicsContext
from chaco.tools.save_tool import SaveTool
from enable.base_tool import BaseTool
from traits.has_traits import HasTraits
from traits.trait_types import Int, Any, File
from pyface.api import clipboard
from enthought.etsconfig.api import ETSConfig

from traitsui.editors.file_editor import FileEditor
from traitsui.group import VGroup
from traitsui.item import Item
from traitsui.menu import OKCancelButtons
from traitsui.view import View

import os.path
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


class SaveFileDialog(HasTraits):
    save_file = File(exists=False)
    dpi = Int(300)

    traits_view = View(
        VGroup(
            Item('save_file', label='Save to:',
                 editor=FileEditor(allow_dir=False,
                                   dialog_style='save',
                                   entries=0),
                 style='simple'),
            Item('dpi', label='Resolution (dpi):')
        ),
        title = 'Save plot',
        width = 400,
        resizable=True,
        buttons=OKCancelButtons
    )


class SaveToolPlus(SaveTool):
    """Subclass of SaveTool that requests a filename and dpi before saving."""

    def normal_key_released(self, event):
        """Handles a key-release when the tool is in the 'normal' state.

        Saves an image of the plot when the user presses Ctrl-S.
        """

        if self.component is None: return

        if ((event.character == "s" or event.character == u'\x13')
            and _is_control_down(event)):

            dialog = SaveFileDialog()
            dialog.edit_traits(kind='modal')

            self.filename = dialog.save_file

            if self.filename != '':
                if os.path.splitext(self.filename)[-1] == ".pdf":
                    self._save_pdf()
                else:
                    self._save_raster(dpi=dialog.dpi)

            event.handled = True

    def _save_raster(self, dpi=300):
        """ Saves an image of the component."""
        self.component.do_layout(force=True)
        # NOTE saving only works properly when dpi is a multiple of 72
        gc = PlotGraphicsContext((int(self.component.outer_width),
                                  int(self.component.outer_height)),
                                 dpi=np.ceil(dpi / 72.0)*72)
        gc.render_component(self.component)
        gc.save(self.filename)


class CopyDataToClipboardTool(BaseTool):
    """Tool that copies the plot's data to the clipboard."""

    data = Any

    def normal_key_released(self, event):
        """Handles a key-release when the tool is in the 'normal' state.

        Copy `data` to the keyboard when the user presses Ctrl-C.
        """

        if self.component is None: return

        if ((event.character == "c" or event.character == u'\x03')
            and _is_control_down(event)):
            clipboard.data = repr(self.data)

            print 'triggered copy!!'

            event.handled = True
