# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

"""Entry point for pyanno UI application."""

from enthought.etsconfig.api import ETSConfig
from pyanno.ui.pyanno_ui_application import pyanno_application

import logging

ETSConfig.toolkit = 'wx'

# set the default color of windows to white
import wx
import traitsui.wx.constants
traitsui.wx.constants.WindowColor = wx.WHITE
traitsui.wx.constants.BorderedGroupColor = wx.WHITE

def main():
    """Create and start the application."""
    with pyanno_application(logging_level=logging.DEBUG) as app:
        app.open()

if __name__ == '__main__':
    main()
