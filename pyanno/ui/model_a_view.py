# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

from traits.has_traits import on_trait_change
from traits.trait_types import Instance, Str, Range, Button, Int, Enum
from traitsui.editors.range_editor import RangeEditor
from traitsui.group import VGroup, VGrid, HGroup
from traitsui.include import Include
from traitsui.item import Item, Spring, UItem
from traitsui.menu import OKButton
from traitsui.view import View

from pyanno.modelA import ModelA
from pyanno.plots.hinton_plot import HintonDiagramPlot
from pyanno.plots.plots_superclass import PyannoPlotContainer
from pyanno.plots.theta_plot import ThetaScatterPlot, ThetaDistrPlot
from pyanno.ui.appbase.wx_utils import is_display_small
from pyanno.ui.model_view import PyannoModelView, NewModelDialog
from pyanno.ui.parameters_tabular_viewer import ParametersTabularView


MODEL_A_NAME = 'Model A (full model)'


class NewModelADialog(NewModelDialog):
    """Create a dialog requesting the parameters to create Model A."""

    model_name = Str(MODEL_A_NAME)
    nclasses = Int(5)

    parameters_group = VGroup(
        Item(name='nclasses',
             editor=RangeEditor(mode='spinner', low=3, high=1000),
             label='Number of annotation classes:',
             width=100),
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
 
        self.omega_hinton_diagram = HintonDiagramPlot(
            data = self.model.omega.tolist(),
            title = 'Omega parameters, P(label = k)'
        )

        self.theta_distribution_plot = ThetaDistrPlot(theta=self.model.theta)
        self.theta_scatter_plot = ThetaScatterPlot(model=self.model)

        self._theta_view_update()


    def _theta_view_default(self):
        return self.theta_distribution_plot


    @on_trait_change('theta_views')
    def _theta_view_update(self):
        if self.theta_views.startswith('Distr'):
            self.theta_view = self.theta_distribution_plot
        else:
            self.theta_view = self.theta_scatter_plot


    def plot_theta_samples(self, theta_samples):
        self.theta_distribution_plot = ThetaDistrPlot(
            theta = self.model.theta,
            theta_samples = theta_samples
        )

        self.theta_scatter_plot = ThetaScatterPlot(model = self.model)
        self.theta_scatter_plot.theta_samples = theta_samples
        self.theta_scatter_plot.theta_samples_valid = True

        self._theta_view_update()


    #### UI traits

    omega_hinton_diagram = Instance(HintonDiagramPlot)

    theta_scatter_plot = Instance(ThetaScatterPlot)

    theta_distribution_plot = Instance(ThetaDistrPlot)

    theta_views = Enum('Distribution plot',
                       'Scatter plot')

    theta_view = Instance(PyannoPlotContainer)


    #### Actions

    view_omega = Button(label='View Omega...')

    view_theta = Button(label='View Theta...')


    def _view_theta_fired(self):
        """Create viewer for theta parameters."""
        theta_view = ParametersTabularView(
            title = 'Model A, parameters Theta',
            data = [self.model.theta.tolist()]
        )
        theta_view.edit_traits()


    def _view_omega_fired(self):
        """Create viewer for parameters omega."""
        omega_view = ParametersTabularView(
            title = 'Model A, parameters Omega',
            data = [self.model.omega.tolist()]
        )
        omega_view.edit_traits()


    #### Traits UI view #########

    def traits_view(self):
        w_view = 350 if is_display_small() else 480

        parameters_group = VGroup(
            Item('_'),

            HGroup(
                VGroup(
                    Spring(),
                    Item('handler.omega_hinton_diagram',
                         style='custom',
                         resizable=False,
                         show_label=False,
                         width=w_view
                    ),
                    Spring()
                ),
                Spring(),
                VGroup(
                    Spring(),
                    Item('handler.view_omega', show_label=False),
                    Spring()
                )
            ),

            Spring(),
            Item('_'),
            Spring(),

            HGroup(
                VGroup(
                    UItem('handler.theta_views'),
                    UItem('handler.theta_view',
                         style='custom',
                         resizable=False,
                         width=w_view
                    ),
                    Spring()
                ),
                Spring(),
                VGroup(
                    Spring(),
                    UItem('handler.view_theta'),
                    Spring()
                )
            )

        )

        body = VGroup(
            Include('info_group'),
            parameters_group
        )

        traits_view = View(body, buttons=[OKButton], resizable=True)
        return traits_view


#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno.models import ModelA

    model = ModelA.create_initial_state(4)
    model_view = ModelAView(model=model)
    model_view.configure_traits(view='traits_view')

    return model, model_view


if __name__ == '__main__':
    m, mv = main()
