# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Apache license

from traits.has_traits import on_trait_change
from traits.trait_types import Instance, Str, Range
from traitsui.group import VGroup
from traitsui.item import Item
from traitsui.menu import OKButton
from traitsui.view import View

from pyanno.modelB import ModelB
from pyanno.plots.hinton_plot import HintonDiagramPlot
from pyanno.plots.matrix_plot import MatrixPlot
from pyanno.ui.model_view import PyannoModelView, NewModelDialog


MODEL_B_NAME = 'Model B (full model)'


class NewModelBDialog(NewModelDialog):
    """Create a dialog requesting the parameters to create Model B."""

    model_name = Str(MODEL_B_NAME)
    nclasses = Range(low=3, high=20, value=5)
    nannotators = Range(low=2, high=20, value=8)

    parameters_group = VGroup(
        Item(name='nclasses',
             label='Number of annotation classes:'),
        Item(name='nannotators',
             label='Number of annotators:')
    )


class ModelBView(PyannoModelView):
    """ Traits UI Model/View for 'ModelB' objects.
    """

    model_name = MODEL_B_NAME
    new_model_dialog_class = NewModelBDialog

    @classmethod
    def _create_model_from_dialog(cls, dialog):
        return ModelB.create_initial_state(dialog.nclasses, dialog.nannotators)


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
