# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (3-clause)

"""Entry point for pyanno UI application."""

from pyanno.modelBt import ModelBt
from pyanno.ui.model_bt_view import ModelBtView
from pyanno.ui.model_data_view import ModelDataView


def main():
    """Create and start the application."""

    model = ModelBt.create_initial_state(5)
    model_data_view = ModelDataView(model=model,
                                    model_view=ModelBtView(model=model))
    model_data_view.configure_traits()


if __name__ == '__main__':
    main()
