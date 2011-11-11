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
from pyanno.modelBt import ModelBt

from pyanno.plots.hinton_plot import HintonDiagramPlot
from pyanno.plots.plots_superclass import PyannoPlotContainer
from pyanno.plots.theta_plot import ThetaScatterPlot, ThetaDistrPlot
from pyanno.ui.model_btloop_view import ModelBtLoopDesignView
from pyanno.ui.model_view import PyannoModelView, NewModelDialog
from pyanno.ui.parameters_tabular_viewer import ParametersTabularView


MODEL_BT_NAME = 'Model B-with-theta'

class NewModelBtDialog(NewModelDialog):
    model_name = Str(MODEL_BT_NAME)
    nclasses = Int(5)
    nannotators = Int(8)

    parameters_group = VGroup(
        Item(name='nclasses',
             editor=RangeEditor(mode='spinner', low=3, high=1000),
             label='Number of annotation classes:',
             width=100),
        Item(name='nannotators',
             editor=RangeEditor(mode='spinner', low=2, high=1000),
             label='Number of annotators:',
             width=100),
        group_theme = 'white_theme.png'
    )


ModelBtView = ModelBtLoopDesignView

class ModelBtView(ModelBtLoopDesignView):
    """ Traits UI Model/View for 'ModelBt' objects.
    """

    model_name = Str(MODEL_BT_NAME)
    new_model_dialog_class = NewModelBtDialog

    @classmethod
    def _create_model_from_dialog(cls, dialog):
        return ModelBt.create_initial_state(
            dialog.nclasses, dialog.nannotators)


#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno import ModelBt

    model = ModelBt.create_initial_state(5, 12)
    anno = model.generate_annotations(100)
    samples = model.sample_posterior_over_accuracy(
        anno, 50,
        step_optimization_nsamples=3)

    model_view = ModelBtView(model=model)
    model_view.plot_theta_samples(samples)
    model_view.configure_traits(view='traits_view')

    return model, model_view


if __name__ == '__main__':
    model, model_view = main()
