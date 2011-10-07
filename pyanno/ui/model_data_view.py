"""View for model and data pair."""
from traits.has_traits import HasTraits
from traits.trait_numeric import Array
from traits.trait_types import Any, File, Instance, Button
from traitsui.editors.file_editor import FileEditor
from traitsui.editors.instance_editor import InstanceEditor
from traitsui.group import HGroup, VGroup
from traitsui.item import Item, Spring
from traitsui.view import View
from pyanno.ui.model_bt_view import ModelBtView

import numpy as np

class ModelDataView(HasTraits):

    model = Any
    model_view = Instance(ModelBtView)

    annotations = Array(dtype=int, shape=(None, None))
    annotations_file = File

    ### Actions ##############################################################

    ml_estimate = Button(label='ML estimate...')
    map_estimate = Button(label='MAP estimate...')
    sample_theta_posterior = Button(label='Sample theta...')
    estimate_labels = Button(label='Estimate labels...')

    def _ml_estimate_fired(self):
        """Request data file and run ML estimation of parameters."""
        print 'Estimate...'
        model.mle(self.annotations, estimate_gamma=True)
        #self.update_from_model()


    ### Views ################################################################

    def traits_view(self):
        model_create_group = (
            Item(label='choose model'),
            Item(label='new button')
        )

        model_group = (
            VGroup (
                model_create_group,
                Item('model_view', style='custom'),
                Spring(),
                show_border = True
            )
        )

        data_create_group = VGroup(
            Item('data_file', style='simple', label='Annotations file', width=400)
        )

        data_group = (
            HGroup (
                data_create_group,
                Spring(),
                show_border = True
            )
        )

        model_data_group = (
            HGroup(
                Item('ml_estimate', enabled_when='annotations.shape != (1,1)')
            )
        )

        full_view = View(
            VGroup(
                HGroup(
                    model_group,
                    data_group
                ),
                model_data_group
            ),
            width = 800,
            height = 600,
            resizable = True
        )

        return full_view



#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno.modelBt import ModelBt

    model = ModelBt.create_initial_state(5)
    model_data_view = ModelDataView(model=model,
                                    model_view=ModelBtView(model=model),
                                    annotations=np.random.randint(4,
                                                                  size=(10,
                                                                        4)))
    model_data_view.configure_traits(view='traits_view')

    return model, model_data_view


if __name__ == '__main__':
    model, model_view = main()
