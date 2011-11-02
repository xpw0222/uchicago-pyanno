# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Apache license

from traits.has_traits import on_trait_change
from traits.trait_types import Instance, Str, Range, Button
from traitsui.group import VGroup, VGrid
from traitsui.include import Include
from traitsui.item import Item
from traitsui.menu import OKButton
from traitsui.view import View

from pyanno.modelA import ModelA
from pyanno.plots.hinton_plot import HintonDiagramPlot
from pyanno.plots.matrix_plot import MatrixPlot
from pyanno.plots.theta_plot import ThetaPlot
from pyanno.ui.model_view import PyannoModelView, NewModelDialog
from pyanno.ui.parameters_tabular_viewer import ParametersTabularView


MODEL_A_NAME = 'Model A (full model)'


class NewModelADialog(NewModelDialog):
    """Create a dialog requesting the parameters to create Model A."""

    model_name = Str(MODEL_A_NAME)
    nclasses = Range(low=3, high=20, value=5)

    parameters_group = VGroup(
        Item(name='nclasses',
             label='Number of annotation classes:'),
    )


class ModelAView(PyannoModelView):
    """ Traits UI Model/View for 'ModelA' objects.
    """

    # name of the model (inherited from PyannoModelView)
    model_name = MODEL_A_NAME

    # dialog to instantiated when creating a new model
    new_model_dialog_class = NewModelADialog


    @classmethod
    def _create_model_from_dialog(cls, dialog):
        return ModelA.create_initial_state(dialog.nclasses)


    @on_trait_change('model,model_updated')
    def update_from_model(self):
        """Recreate plots."""
        self.theta_plot = ThetaPlot(
            model=self.model,
            title = 'Theta parameters, P(annotator[k] is correct)'
        )
        #self.theta_plot._update_plot_data()

        self.omega_hinton_diagram = HintonDiagramPlot(
            data = self.model.omega.tolist(),
            title = 'Omega parameters, P(label = k)'
        )


    def plot_theta_samples(self, theta_samples):
        self.theta_plot.theta_samples = theta_samples
        self.theta_plot.theta_samples_valid = True
        self.theta_plot.redraw = True


    #### UI traits

    theta_plot = Instance(ThetaPlot)

    omega_hinton_diagram = Instance(HintonDiagramPlot)


    #### Actions

    view_omega = Button(label='View...')

    view_theta = Button(label='View...')


    def _view_theta_fired(self):
        """Create viewer for theta parameters."""
        theta_view = ParametersTabularView(
            title = 'Model A, parameters theta',
            data = [self.model.theta.tolist()]
        )
        theta_view.edit_traits()


    def _view_omega_fired(self):
        """Create viewer for parameters omega."""
        omega_view = ParametersTabularView(
            title = 'Model A, parameters omega',
            data = [self.model.omega.tolist()]
        )
        omega_view.edit_traits()


    #### Traits UI view #########

    parameters_group = VGrid(
        Item('handler.omega_hinton_diagram',
             style='custom',
             resizable=False,
             show_label=False),
        Item('handler.view_omega', show_label=False),
        Item('handler.theta_plot',
             style='custom',
             resizable=False,
             show_label=False,
        ),
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

    from pyanno import ModelA

    model = ModelA.create_initial_state(4)
    model_view = ModelAView(model=model)
    model_view.configure_traits(view='traits_view')

    return model, model_view


if __name__ == '__main__':
    m, mv = main()
