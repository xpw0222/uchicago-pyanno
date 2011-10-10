"""View for model and data pair."""
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_numeric import Array
from traits.trait_types import Any, File, Instance, Button, Enum, Str, Range, Bool, Float, Event, List
from traitsui.editors.file_editor import FileEditor
from traitsui.editors.instance_editor import InstanceEditor
from traitsui.editors.tabular_editor import TabularEditor
from traitsui.group import HGroup, VGroup
from traitsui.item import Item, Spring
from traitsui.menu import OKCancelButtons
from traitsui.view import View
from pyanno.modelBt import ModelBt
from pyanno.ui.arrayview import Array2DAdapter
from pyanno.ui.model_bt_view import ModelBtView

import numpy as np

ANNOTATIONS_INFO_STR = """Annotations file {}
Number of annotations: {}
Number of annotators: {}
Labels: {}"""


# TODO fix size, scroll bar on second line
# TODO remember last setting of parameters
class NewModelDialog(HasTraits):
    model_name = Str
    nclasses = Range(low=3, high=50)

    def traits_view(self):
        traits_view = View(
            VGroup(
                Item(name='nclasses',
                     label='Number of annotation classes:',
                     id='test')
            ),
            buttons=OKCancelButtons,
            title='Create new ' + self.model_name,
            kind='modal'
        )
        return traits_view


class DataView(HasTraits):
    data = Array

    def traits_view(self):
        return View(
            VGroup(Item('data',
                        editor=TabularEditor
                            (
                            adapter=Array2DAdapter(ncolumns=len(self.data[0]),
                                                   format='%d',
                                                   show_index=False)),
                        show_label=False)),
            title='Model B-with-Theta, gamma parameters',
            width=500,
            height=800,
            resizable=True,
            buttons=OKCancelButtons
        )


class ModelDataView(HasTraits):

    model = Any
    model_view = Instance(ModelBtView)
    model_updated = Event
    model_update_suspended = Bool(False)

    annotations = Array(dtype=int, shape=(None, None))
    annotations_file = File
    annotations_updated = Event
    annotations_are_defined = Bool(False)

    annotations_info_str = Str


    @on_trait_change('annotations_updated,model_updated')
    def _update_info_str(self):
        if not self.annotations_are_defined:
            self.info_string = ('Please define an annotations list.')
        else:
            self.info_string = ('Model and annotations are defined.')

    @on_trait_change('annotations_updated')
    @on_trait_change('model,model:theta,model:gamma')
    def _fire_model_updated(self):
        if not self.model_update_suspended:
            self.model_updated = True


    ### Actions ##############################################################

    ml_estimate = Button(label='ML estimate...')
    map_estimate = Button(label='MAP estimate...')
    sample_theta_posterior = Button(label='Sample theta...')
    estimate_labels = Button(label='Estimate labels...')

    def _ml_estimate_fired(self):
        """Request data file and run ML estimation of parameters."""
        print 'Estimate...'
        model.mle(self.annotations, estimate_gamma=True)
        self.model_view.update_from_model()

    @on_trait_change('annotations_file')
    def _update_annotations_file(self):
        print 'file'


    ### Views ################################################################

    def traits_view(self):
        model_create_group = (
            HGroup(
                Item(label='choose model'),
                Item(label='new button'),
                label = 'Model creation',
                show_border=True
            )
        )

        model_group = (
            VGroup (
                model_create_group,
                Item('model_view', style='custom', show_label=False),
                show_border = True,
                label = 'Model view'
            )
        )

        data_create_group = VGroup(
            Item('annotations_file', style='simple', label='Annotations file',
                 width=400),
            show_border = True,
            label = 'Data creation'
        )

        data_group = (
            HGroup (
                data_create_group,
                Spring(),
                show_border = True,
                label = 'Data view'
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
            width = 1200,
            height = 800,
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
                                    annotations=model.generate_annotations
                                        (model.generate_labels(50*8)))
    model_data_view.configure_traits(view='traits_view')

    return model, model_data_view


if __name__ == '__main__':
    model, model_data_view = main()
