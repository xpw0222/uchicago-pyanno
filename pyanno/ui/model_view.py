# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Apache license

"""Defines super class for all traits UI views of annotations models."""
from traits.has_traits import HasTraits
from traits.trait_types import Event, Str, Instance
from traitsui.group import Group
from traitsui.handler import ModelView
from traitsui.include import Include
from traitsui.menu import OKCancelButtons
from traitsui.view import View


class NewModelDialog(HasTraits):
    """Create a dialog requesting the parameters to create a pyAnno model."""

    model_name = Str
    parameters_group = Instance(Group)

    def traits_view(self):
        traits_view = View(
            Include('parameters_group'),
            buttons=OKCancelButtons,
            title='Create new ' + self.model_name,
            kind='modal'
        )
        return traits_view


class PyannoModelView(ModelView):
    """Superclass for all pyAnno model views"""

    #### Class attributes (*not* traits)

    # name of the model
    model_name = 'pyAnno model'

    # subclass of NewModelDialog
    new_model_dialog_class = None


    #### Model traits

    # raised when model is updated
    model_updated = Event


    #### Class methods

    @classmethod
    def _create_model_from_dialog(cls, dialog):
        raise NotImplementedError()

    @classmethod
    def create_model_dialog(cls):
        """Open a dialog to create a new model and model view."""

        dialog = cls.new_model_dialog_class()
        dialog_ui = dialog.edit_traits()
        if dialog_ui.result:
            # user pressed 'Ok'
            # create model and update view
            model = cls._create_model_from_dialog(dialog)
            model_view = cls(model=model)
            return model, model_view
        else:
            return None, None
