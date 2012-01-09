# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

from traits.trait_types import    Str, Int
from traitsui.api import  Item, VGroup
from traitsui.editors.range_editor import RangeEditor
from pyanno.modelBt import ModelBt

from pyanno.ui.model_btloop_view import ModelBtLoopDesignView
from pyanno.ui.model_view import  NewModelDialog


MODEL_BT_NAME = 'Model B-with-theta'

class NewModelBtDialog(NewModelDialog):
    model_name = Str(MODEL_BT_NAME)
    nclasses = Int(5)
    nannotators = Int(8)

    parameters_group = VGroup(
        Item(name='nclasses',
             editor=RangeEditor(mode='spinner', low=2, high=1000),
             label='Number of annotation classes:',
             width=100),
        Item(name='nannotators',
             editor=RangeEditor(mode='spinner', low=2, high=1000),
             label='Number of annotators:',
             width=100),
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

    from pyanno.models import ModelBt

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
