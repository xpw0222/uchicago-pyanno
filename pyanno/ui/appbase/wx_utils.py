# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

"""Utility functions for dealing with wx backend."""

import wx
import traitsui.wx.constants

def is_display_small():
    """Return True if screen is smaller than 1300x850."""
    size = wx.GetDisplaySize()
    if size is not None:
        w, h = size
        return w < 1300 or h < 850
    return False


def set_background_color_to_white():
    """Set the default color of windows to white."""
    traitsui.wx.constants.WindowColor = wx.WHITE
    traitsui.wx.constants.BorderedGroupColor = wx.WHITE
