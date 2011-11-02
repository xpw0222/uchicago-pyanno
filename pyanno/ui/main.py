# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Apache license

"""Entry point for pyanno UI application."""

from enthought.etsconfig.api import ETSConfig
ETSConfig.toolkit = 'wx'

from pyanno.modelBt import ModelBt
from pyanno.ui.model_bt_view import ModelBtView
from pyanno.ui.model_data_view import ModelDataView

import logging


def setup_logging():
    logging.basicConfig(level=logging.DEBUG)


def main():
    """Create and start the application."""

    setup_logging()

    # create initial model
    model = ModelBt.create_initial_state(5)

    # create main window
    model_data_view = ModelDataView(model=model,
                                    model_view=ModelBtView(model=model))

    # display main window
    model_data_view.configure_traits()


if __name__ == '__main__':
    main()
