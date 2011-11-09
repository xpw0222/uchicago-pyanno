# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

from traits.has_traits import on_trait_change
from traits.trait_types import Button, List, CFloat, Str, Range, Int, Enum, Any
from traitsui.api import View, Item, VGroup
from traitsui.editors.range_editor import RangeEditor
from traitsui.group import VGrid, HGroup
from traitsui.include import Include
from traitsui.item import Spring
from traitsui.menu import OKButton
from traits.api import Instance
import numpy as np
from pyanno.modelBt import ModelBt

from pyanno.plots.hinton_plot import HintonDiagramPlot
from pyanno.plots.theta_plot import ThetaPlot
from pyanno.ui.model_view import PyannoModelView, NewModelDialog
from pyanno.ui.parameters_tabular_viewer import ParametersTabularView


MODEL_BT_NAME = 'Model B-with-theta'


class NewModelBtDialog(NewModelDialog):
    model_name = Str(MODEL_BT_NAME)
    nclasses = Int(5)

    parameters_group = VGroup(
        Item(name='nclasses',
             editor=RangeEditor(mode='spinner', low=3, high=1000),
             label='Number of annotation classes:',
             width=100)
    )


class ModelBtView(PyannoModelView):
    """ Traits UI Model/View for 'ModelBt' objects.
    """

    model_name = Str(MODEL_BT_NAME)
    new_model_dialog_class = NewModelBtDialog

    @classmethod
    def _create_model_from_dialog(cls, dialog):
        return ModelBt.create_initial_state(dialog.nclasses)


    #### Model properties
    gamma = List(CFloat)


    @on_trait_change('model_updated')
    def update_from_model(self):
        """Update view parameters to the ones in the model."""
        self.gamma = self.model.gamma.tolist()
        self.theta_view.theta_samples_valid = False
        self.theta_view.redraw = True


    def _gamma_default(self):
        return self.model.gamma.tolist()


    @on_trait_change('gamma[]', post_init=True)
    def _update_gamma(self):
        self.model.gamma = np.asarray(self.gamma)
        self.gamma_hinton.data = self.gamma


    #### UI-related traits

    gamma_hinton = Instance(HintonDiagramPlot)

    theta_view = Instance(ThetaPlot)


    def _gamma_hinton_default(self):
        return HintonDiagramPlot(data = self.gamma,
                                 title='Gamma parameters, P(label=k)')


    def _theta_view_default(self):
        self.theta_view = ThetaPlot(model=self.model)
        self.theta_view._update_plot_data()
        return self.theta_view


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


    def plot_theta_samples(self, theta_samples):
        self.theta_view.theta_samples = theta_samples
        self.theta_view.theta_samples_valid = True
        self.theta_view.redraw = True


    #### Traits UI view #########

    parameters_group = VGroup(
        Item('_'),

        HGroup(
            VGroup(
                Spring(),
                Item('handler.gamma_hinton',
                     style='custom',
                     resizable=False,
                     show_label=False,
                     width=550
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
                Spring(),
                Item('handler.theta_view',
                     style='custom',
                     resizable=False,
                     show_label=False,
                     width=550
                ),
                Spring()
            ),
            Spring(),
            VGroup(
                Spring(),
                Item('handler.view_theta', show_label=False),
                Spring()
            )
        )
    )

    body = VGroup(
        Include('info_group'),
        Include('parameters_group')
    )

    traits_view = View(body, buttons=[OKButton], resizable=True)


#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno import ModelBt

    model = ModelBt.create_initial_state(5)
    model_view = ModelBtView(model=model)
    model_view.configure_traits(view='traits_view')

    return model, model_view


if __name__ == '__main__':
    model, model_view = main()
