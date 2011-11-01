# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Apache license

from traits.has_traits import on_trait_change, HasTraits
from traits.trait_numeric import Array
from traits.trait_types import Instance, Str, Range, Button, Int
from traits.traits import Property
from traitsui.editors.range_editor import RangeEditor
from traitsui.group import VGroup, VGrid
from traitsui.include import Include
from traitsui.item import Item
from traitsui.menu import OKButton
from traitsui.view import View

from pyanno.modelB import ModelB
from pyanno.plots.hinton_plot import HintonDiagramPlot
from pyanno.plots.matrix_plot import MatrixPlot
from pyanno.ui.model_view import PyannoModelView, NewModelDialog
from pyanno.ui.parameters_tabular_viewer import ParametersTabularView


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


class ModelA_ThetaView(HasTraits):
    """Tabular view for the parameters theta in Model B.

    Includes a spin box to select the parameters for each annotator.
    """

    @staticmethod
    def show(theta):
        """Create a window that with a ThetaView inside."""
        tv = ModelA_ThetaView(theta=theta)
        tv.edit_traits()

    # 3D tensor to be displayed
    theta = Array

    # title of the view window
    title = "Model B, parameters theta"

    # annotator number
    annotator_idx = Int(0)

    # tabular view of theta for annotator j
    theta_j_view = Instance(ParametersTabularView)

    def _theta_j_view_default(self):
        return ParametersTabularView(
            data = self.theta[self.annotator_idx,:,:].tolist()
        )

    @on_trait_change('annotator_idx')
    def _theta_j_update(self):
        self.theta_j_view.data = self.theta[self.annotator_idx,:,:].tolist()

    def traits_view(self):

        traits_view = View(
            VGroup(
                Item('annotator_idx',
                     label='Annotator index',
                     editor=RangeEditor(mode='spinner',
                                        low=0, high=self.theta.shape[0]-1,
                                        ),
                ),
                VGroup(
                    Item('theta_j_view', style='custom', show_label=False)
                )
            ),
            width = 600,
            height = 400,
            resizable = True
        )
        return traits_view


class ModelBView(PyannoModelView):
    """ Traits UI Model/View for 'ModelB' objects.
    """

    # name of the model (inherited from PyannoModelView)
    model_name = MODEL_B_NAME

    # dialog to instantiated when creating a new model
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


    #### Actions

    view_pi = Button(label='View...')

    view_theta = Button(label='View...')


    def _view_pi_fired(self):
        """Create viewer for parameters pi."""
        pi_view = ParametersTabularView(
            title = 'Model B, parameters pi',
            data = [self.model.pi.tolist()]
        )
        pi_view.edit_traits()


    def _view_theta_fired(self):
        """Create viewer for parameters theta."""
        ModelA_ThetaView.show(self.model.theta)


    #### Traits UI view #########

    parameters_group = VGrid(
        Item('handler.pi_hinton_diagram',
             style='custom',
             resizable=False,
             show_label=False),
        Item('handler.view_pi', show_label=False),
        Item('handler.theta_matrix_plot',
             style='custom',
             resizable=False,
             show_label=False),
        Item('handler.view_theta', show_label=False),
    )

    body = VGroup(
        Include('info_group'),
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
