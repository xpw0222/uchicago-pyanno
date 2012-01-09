# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

"""Defines super class for all traits UI views of annotations models."""
from traits.has_traits import HasTraits
from traits.trait_types import Event, Str, Instance
from traitsui.group import Group, VGroup, VGrid
from traitsui.handler import ModelView
from traitsui.include import Include
from traitsui.item import Item, Label
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
            #kind='modal'
        )
        return traits_view


class PyannoModelView(ModelView):
    """Superclass for all pyAnno model views"""

    #### Class attributes (*not* traits)

    # subclass of NewModelDialog
    new_model_dialog_class = None


    #### Model traits

    # name of the model
    model_name = Str('pyAnno model')

    # raised when model is updated
    model_updated = Event


    #### Class methods

    @classmethod
    def _create_model_from_dialog(cls, dialog):
        raise NotImplementedError()

    @classmethod
    def create_model_dialog(cls, parent, **kwargs):
        """Open a dialog to create a new model.

        All keyword arguments are passed to the model dialog class constructor.
        """

        dialog = cls.new_model_dialog_class(**kwargs)
        dialog_ui = dialog.edit_traits(kind="livemodal",
                                       parent=parent)
        if dialog_ui.result:
            # user pressed 'Ok'
            # create model and update view
            model = cls._create_model_from_dialog(dialog)
            return model
        else:
            return None


    info_group = VGroup(
        Item('model_name',
             label='Model name:',
             style='readonly'),
        VGrid(
            Item('model.nclasses',
                 label='Number of classes:',
                 style='readonly'),
            Label(' '),
            Item('model.nannotators',
                 label='Number of annotators:',
                 style='readonly'),
            Label(' '),
            padding=0
        ),
        padding=0
    )
