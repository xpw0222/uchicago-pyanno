# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

from traits.has_traits import on_trait_change
from traits.trait_types import Button, List, CFloat, Str, Range, Int, Enum, Any
from traitsui.api import View, Item, VGroup
from traitsui.editors.range_editor import RangeEditor
from traitsui.group import VGrid, HGroup
from traitsui.handler import Handler
from traitsui.include import Include
from traitsui.item import Spring, UItem
from traitsui.menu import OKButton
from traits.api import Instance
import numpy as np
from pyanno.modelBt_loopdesign import ModelBtLoopDesign

from pyanno.plots.hinton_plot import HintonDiagramPlot
from pyanno.plots.plots_superclass import PyannoPlotContainer
from pyanno.plots.theta_plot import ThetaScatterPlot, ThetaDistrPlot
from pyanno.ui.appbase.wx_utils import is_display_small
from pyanno.ui.model_view import PyannoModelView, NewModelDialog
from pyanno.ui.parameters_tabular_viewer import ParametersTabularView


MODEL_BT_NAME = 'Model B-with-theta (optimized for loop design)'

class NewModelBtLoopDesignDialog(NewModelDialog):
    model_name = Str(MODEL_BT_NAME)
    nclasses = Int(5)

    parameters_group = VGroup(
        Item(name='nclasses',
             editor=RangeEditor(mode='spinner', low=3, high=1000),
             label='Number of annotation classes:',
             width=100),
    )


class ModelBtLoopDesignView(PyannoModelView):
    """ Traits UI Model/View for 'ModelBtLoopDesign' objects.
    """

    model_name = Str(MODEL_BT_NAME)
    new_model_dialog_class = NewModelBtLoopDesignDialog

    @classmethod
    def _create_model_from_dialog(cls, dialog):
        return ModelBtLoopDesign.create_initial_state(dialog.nclasses)


    #### Model properties
    gamma = List(CFloat)


    @on_trait_change('model,model_updated')
    def update_from_model(self):
        """Update view parameters to the ones in the model."""
        self.gamma = self.model.gamma.tolist()

        self.theta_distribution_plot = ThetaDistrPlot(theta=self.model.theta)
        self.theta_scatter_plot = ThetaScatterPlot(model=self.model)

        self._theta_view_update()


    def _gamma_default(self):
        return self.model.gamma.tolist()


    @on_trait_change('gamma[]', post_init=True)
    def _update_gamma(self):
        self.model.gamma = np.asarray(self.gamma)
        self.gamma_hinton.data = self.gamma


    #### UI-related traits

    gamma_hinton = Instance(HintonDiagramPlot)

    theta_scatter_plot = Instance(ThetaScatterPlot)

    theta_distribution_plot = Instance(ThetaDistrPlot)

    theta_views = Enum('Distribution plot',
                       'Scatter plot')

    theta_view = Instance(PyannoPlotContainer)

    def _gamma_hinton_default(self):
        return HintonDiagramPlot(data = self.gamma,
                                 title='Gamma parameters, P(label=k)')


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


    #### Actions

    view_gamma = Button(label='View Gamma...')

    view_theta = Button(label='View Theta...')


    def _view_gamma_fired(self):
        """Create viewer for gamma parameters."""
        gamma_view = ParametersTabularView(
            title = 'Model B-with-Theta, parameters Gamma',
            data=[self.gamma]
        )
        gamma_view.edit_traits()


    def _view_theta_fired(self):
        """Create viewer for theta parameters."""
        theta_view = ParametersTabularView(
            title = 'Model B-with-Theta, parameters Theta',
            data=[self.model.theta.tolist()]
        )
        theta_view.edit_traits()


    #### Traits UI view #########

    def traits_view(self):
        if is_display_small():
            w_view = 350
        else:
            w_view = 480

        parameters_group = VGroup(
            Item('_'),

            HGroup(
                VGroup(
                    Spring(),
                    Item('handler.gamma_hinton',
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
                    Item('handler.view_gamma', show_label=False),
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
            parameters_group,
        )

        traits_view = View(body, buttons=[OKButton], resizable=True)
        return traits_view


#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno.modelBt_loopdesign import ModelBtLoopDesign

    model = ModelBtLoopDesign.create_initial_state(5)
    anno = model.generate_annotations(100)
    samples = model.sample_posterior_over_accuracy(anno, 50,
                                                   step_optimization_nsamples=3)

    model_view = ModelBtLoopDesignView(model=model)
    model_view.plot_theta_samples(samples)
    model_view.configure_traits(view='traits_view')

    return model, model_view


if __name__ == '__main__':
    model, model_view = main()
