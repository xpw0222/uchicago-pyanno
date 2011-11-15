# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

"""Utility functions for dealing with wx backend."""

import wx

def is_display_small():
    """Return True if screen is smaller than 1300x850."""
    size = wx.GetDisplaySize()
    if size is not None:
        w, h = size
        return w < 1300 or h < 850
    return False
