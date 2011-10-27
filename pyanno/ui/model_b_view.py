# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Apache license
from traits.has_traits import on_trait_change, HasTraits
from traits.trait_types import Instance, Event, Bool, Str, Range
from traitsui.group import VGroup

from traitsui.handler import ModelView
from traitsui.item import Item, Label
from traitsui.menu import OKButton, OKCancelButtons
from traitsui.view import View
from pyanno.modelB import ModelB
from pyanno.plots.hinton_plot import HintonDiagramPlot
from pyanno.plots.matrix_plot import MatrixPlot


MODEL_B_NAME = 'Model B (full model)'


class NewModelBDialog(HasTraits):
    """Create a dialog requesting the parameters to create Model B."""

    model_name = Str(MODEL_B_NAME)
    nclasses = Range(low=3, high=20, value=5)
    nannotators = Range(low=2, high=20, value=8)

    def traits_view(self):
        traits_view = View(
            VGroup(
                Item(name='nclasses',
                     label='Number of annotation classes:'),
                Item(name='nannotators',
                     label='Number of annotators:')
            ),
            buttons=OKCancelButtons,
            title='Create new ' + self.model_name,
            kind='modal'
        )
        return traits_view


class ModelBView(ModelView):
    """ Traits UI Model/View for 'ModelB' objects.
    """

    model_name = Str(MODEL_B_NAME)
    @staticmethod
    def create_model_dialog():
        """Open a dialog to create a new model and model view."""

        dialog = NewModelBDialog()
        dialog_ui = dialog.edit_traits()
        if dialog_ui.result:
            # user pressed 'Ok'
            # create model and update view
            model = ModelB.create_initial_state(dialog.nclasses,
                                                dialog.nannotators)
            model_view = ModelBView(model=model)
            return model, model_view
        else:
            return None, None

    # raised when model is updated
    model_updated = Event

    @on_trait_change('model,model_updated')
    def update_from_model(self):
        """Update view parameters to the ones in the model."""
        self.pi_hinton_diagram = HintonDiagramPlot(
            data = self.model.pi.tolist(),
            title = 'Pi parameters, P(label=k)')
        self.theta_matrix_plot = MatrixPlot(
            matrix = self.model.theta[0,:,:],
            colormap_low = 0., colormap_high = 1.,
            title = 'Theta[0,:,:]'
        )


    #### UI traits
    pi_hinton_diagram = Instance(HintonDiagramPlot)
    theta_matrix_plot = Instance(MatrixPlot)

    info_group = VGroup(
        Item('model_name',
             label='Model name:',
             style='readonly',
             emphasized=True),
        Item('model.nclasses',
             label='Number of labels',
             style='readonly',
             emphasized=True),
        Item('model.nannotators',
             label='Number of annotators',
             style='readonly',
             emphasized=True),
    )

    parameters_group = VGroup(
        Item('handler.pi_hinton_diagram',
             style='custom',
             resizable=False,
             show_label=False),
        Item('handler.theta_matrix_plot',
             style='custom',
             resizable=False,
             show_label=False)
    )

    body = VGroup(
        info_group,
        parameters_group
    )

    traits_view = View(body, buttons=[OKButton], resizable=True)


#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno import ModelB

    model = ModelB.create_initial_state(4, 5)
    model_view = ModelBView(model=model)
    model_view.configure_traits(view='traits_view')

    return model, model_view


if __name__ == '__main__':
    m, mv = main()
