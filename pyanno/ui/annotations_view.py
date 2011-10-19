from traits.has_traits import HasTraits, HasStrictTraits, on_trait_change
from traits.trait_numeric import Array
from traits.trait_types import Instance, Int, List, ListFloat, Str, Button, Event
from traits.traits import Property
from traitsui.api import ModelView, View, VGroup
from traitsui.editors.tabular_editor import TabularEditor
from traitsui.group import HGroup
from traitsui.item import Item, Spring
from traitsui.menu import OKCancelButtons
from pyanno.ui.arrayview import Array2DAdapter
from pyanno.ui.hinton_plot import HintonDiagramPlot
from pyanno.util import labels_frequency, is_valid

import numpy as np


ANNOTATIONS_INFO_STR = """Annotations file {}
Number of annotations: {}
Number of annotators: {}
Labels: {}"""


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
            title='Annotations (-1 is missing value)',
            width=500,
            height=800,
            resizable=True,
            buttons=OKCancelButtons
        )


class AnnotationsView(HasStrictTraits):
    """ Traits UI Model/View for annotations."""

    ### Model-related traits ###

    annotations = Array
    nclasses = Int(1)
    annotations_name = Str

    frequency = ListFloat

    @on_trait_change('annotations,annotations_updated,nclasses')
    def _update_frequency(self):
        nclasses = max(self.nclasses, self.annotations.max())
        self.frequency =  labels_frequency(self.annotations,
                                                nclasses).tolist()
        self.frequency_plot.data = self.frequency


    ### Traits UI definitions ###

    # event raised when annotations are updated
    annotations_updated = Event

    ## annotations info string definition
    annotations_info_str = Str

    @on_trait_change('annotations,annotations_updated')
    def _update_annotations_info_str(self):
        valid = is_valid(self.annotations)
        classes = str(np.unique(self.annotations[valid]))
        self.annotations_info_str = ANNOTATIONS_INFO_STR.format(
            self.annotations_name,
            self.annotations.shape[0],
            self.annotations.shape[1],
            classes)

    ## frequency plot definition
    frequency_plot = Instance(HintonDiagramPlot)

    def _frequency_plot_default(self):
        return HintonDiagramPlot(data=self.frequency,
                                 title='Observed label frequencies')

    ## edit data button opens annotations editor
    edit_data = Button(label='Edit annotations...')

    def _edit_data_fired(self):
        data_view = DataView(data=self.annotations)
        data_view.edit_traits(kind='modal')
        self.annotations_updated = True


    ### View definition ###
    body = VGroup(
        Item('annotations_info_str',
             show_label=False,
             style='readonly',
             height=80
        ),
        Item('frequency_plot',
             style='custom',
             resizable=False,
             show_label=False
        ),
        HGroup(
            Item('edit_data',
                 enabled_when='annotations_are_defined',
                 show_label=False),
            Spring()
        ),
    )

    traits_view = View(body)



#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno.modelBt import ModelBt
    model = ModelBt.create_initial_state(5)
    annotations = model.generate_annotations(model.generate_labels(2))
    model_view = AnnotationsView(annotations=annotations, annotations_name='blah')
    model_view.configure_traits()
    return model, annotations, model_view


if __name__ == '__main__':
    m, a, mv = main()
