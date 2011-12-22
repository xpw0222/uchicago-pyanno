# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

"""Entry point for pyanno UI application.

At present, the application is based on the wx backend of the traitsui library.
It also supports 2 screen formats:

* for large displays (larger than 1300x850), the main window will be
  1300 x 850 pixels large

* for small displays it will be 1024x768
"""


from traits.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'wx'

import pyanno.ui.appbase.wx_utils as wx_utils
wx_utils.set_background_color_to_white()

from pyanno.ui.pyanno_ui_application import pyanno_application
import numpy

import logging


def main():
    """Create and start the application."""

    # deactivate warnings for operations like log(0.) and log(-inf), which
    # are handled correctly by pyanno
    numpy.seterr(divide='ignore', invalid='ignore')

    with pyanno_application(logging_level=logging.INFO) as app:
        app.open()

if __name__ == '__main__':
    main()
