# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Apache license

from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import Button, List, CFloat, Str, Range
from traitsui.api import View, Item, Group, VGroup, HGroup
from traitsui.editors.tabular_editor import TabularEditor
from traitsui.include import Include
from traitsui.item import  Spring
from traitsui.menu import OKButton, OKCancelButtons
from traits.api import Instance
import numpy as np
from pyanno.modelBt import ModelBt

from pyanno.ui.arrayview import Array2DAdapter
from pyanno.plots.hinton_plot import HintonDiagramPlot
from pyanno.plots.theta_plot import ThetaPlot
from pyanno.ui.model_view import PyannoModelView, NewModelDialog


MODEL_BT_NAME = 'Model B-with-theta'


class NewModelBtDialog(NewModelDialog):
    model_name = Str(MODEL_BT_NAME)
    nclasses = Range(low=3, high=50, value=5)

    parameters_group = VGroup(
        Item(name='nclasses',
             label='Number of annotation classes:')
    )


class GammaView(HasTraits):
    data = List

    def traits_view(self):
        return View(
            Group(Item('data',
                       editor=TabularEditor
                           (
                           adapter=Array2DAdapter(ncolumns=len(self.data[0]),
                                                  show_index=False)),
                       show_label=False)),
            title     = 'Model B-with-Theta, gamma parameters',
            width     = 500,
            height    = 200,
            resizable = True,
            buttons   = OKCancelButtons
            )


def vcenter(item):
    return VGroup(Spring(),item,Spring())

class ModelBtView(PyannoModelView):
    """ Traits UI Model/View for 'ModelBt' objects.
    """

    model_name = Str(MODEL_BT_NAME)
    new_model_dialog_class = NewModelBtDialog

    @classmethod
    def _create_model_from_dialog(cls, dialog):
        return ModelBt.create_initial_state(dialog.nclasses)


    #### Model properties #######
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


    def plot_theta_samples(self, theta_samples):
        self.theta_view.theta_samples = theta_samples
        self.theta_view.theta_samples_valid = True


    #### Traits UI view #########
    gamma_hinton = Instance(HintonDiagramPlot)

    theta_view = Instance(ThetaPlot)

    ## Actions ##
    edit_gamma = Button(label='Edit...')
    edit_theta = Button(label='Edit...')
    generate_data = Button(label='Generate annotations...')

    def _edit_gamma_fired(self):
        """Create editor for gamma parameters"""
        gamma_view = GammaView(data=[self.gamma])
        gamma_view.edit_traits(kind='livemodal')

    body = VGroup(
        Include('info_group'),
        HGroup(
            Item('handler.gamma_hinton',
                 style='custom',
                 resizable=False,
                 show_label=False,
            ),
            vcenter(Item('handler.edit_gamma', show_label=False)),
        ),
        HGroup(
            Spring(),
            Item('handler.theta_view',
                 style='custom',
                 resizable=False,
                 show_label=False,
            ),
            Spring(),
            Item('handler.edit_theta',
                         show_label=False,
                         enabled_when='False'),
        )
        #Item('model.theta', label="Theta[j] = P(annotation_j=k | label=k)"),
    )

    traits_view = View(body, buttons=[OKButton], resizable=True)


    def _gamma_hinton_default(self):
        return HintonDiagramPlot(data = self.gamma,
                                 title='Gamma parameters, P(label=k)')

    def _theta_view_default(self):
        self.theta_view = ThetaPlot(model=self.model)
        self.theta_view._update_plot_data()
        return self.theta_view


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
