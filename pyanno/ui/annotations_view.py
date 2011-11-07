# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

from traits.has_traits import HasTraits, on_trait_change
from traits.trait_numeric import Array
from traits.trait_types import Instance, Int, ListFloat, Str, Button, Event
from traitsui.api import  View, VGroup
from traitsui.editors.tabular_editor import TabularEditor
from traitsui.group import HGroup
from traitsui.item import Item, Spring
from traitsui.menu import OKCancelButtons
from pyanno.annotations import AnnotationsContainer
from pyanno.ui.arrayview import Array2DAdapter
from pyanno.plots.hinton_plot import HintonDiagramPlot
from pyanno.util import labels_frequency, is_valid


ANNOTATIONS_INFO_STR = """Annotations file {}
Number of annotations: {}
Number of annotators: {}
Number of classes: {}
Labels: {}"""


class DataView(HasTraits):
    data = Array(dtype=object)

    def traits_view(self):
        return View(
            VGroup(Item('data',
                        editor=TabularEditor
                            (
                            adapter=Array2DAdapter(ncolumns=len(self.data[0]),
                                                   format='%s',
                                                   show_index=False)),
                        show_label=False)),
            title='Annotations (-1 is missing value)',
            width=500,
            height=800,
            resizable=True,
            buttons=OKCancelButtons
        )


class AnnotationsView(HasTraits):
    """ Traits UI Model/View for annotations."""

    ### Model-related traits ###

    # container for annotations and their metadata
    annotations_container = Instance(AnnotationsContainer)

    # this can be set by the current model (could be different from the
    # number of classes in the annotations themselves)
    nclasses = Int(1)

    frequency = ListFloat

    @on_trait_change('annotations_container,annotations_updated,nclasses')
    def _update_frequency(self):
        nclasses = max(self.nclasses, self.annotations_container.nclasses)
        self.frequency =  labels_frequency(
            self.annotations_container.annotations,
            nclasses).tolist()
        self.frequency_plot = HintonDiagramPlot(
            data=self.frequency,
            title='Observed label frequencies')


    ### Traits UI definitions ###

    # event raised when annotations are updated
    annotations_updated = Event

    ## annotations info string definition
    annotations_info_str = Str

    @on_trait_change('annotations_container,annotations_updated')
    def _update_annotations_info_str(self):
        valid = is_valid(self.annotations_container)
        classes = str(self.annotations_container.labels)
        self.annotations_info_str = ANNOTATIONS_INFO_STR.format(
            self.annotations_container.name,
            self.annotations_container.nitems,
            self.annotations_container.nannotators,
            self.annotations_container.nclasses,
            classes)

    ## frequency plot definition
    frequency_plot = Instance(HintonDiagramPlot)

    ## edit data button opens annotations editor
    edit_data = Button(label='Edit annotations...')

    def _edit_data_fired(self):
        data_view = DataView(data=self.annotations_container.raw_annotations)
        data_view.edit_traits(kind='modal')
        self.annotations_container = AnnotationsContainer.from_array(
            data_view.data,
            name = self.annotations_container.name
        )
        self.annotations_updated = True


    ### View definition ###
    body = VGroup(
        Item('annotations_info_str',
             show_label=False,
             style='readonly',
             height=100
        ),
        HGroup(
            Item('edit_data',
                 enabled_when='annotations_are_defined',
                 show_label=False),
            Spring()
        ),
        Item('frequency_plot',
             style='custom',
             resizable=False,
             show_label=False
        ),
    )

    traits_view = View(body)



#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno import ModelBt
    model = ModelBt.create_initial_state(5)
    annotations = model.generate_annotations(model.generate_labels(2))

    anno = AnnotationsContainer.from_array(annotations, name='blah')
    model_view = AnnotationsView(annotations_container=anno)
    model_view.configure_traits()
    return model, annotations, model_view


if __name__ == '__main__':
    m, a, mv = main()
