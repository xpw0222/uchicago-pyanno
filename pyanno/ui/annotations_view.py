# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

from traits.has_traits import HasTraits, on_trait_change
from traits.trait_numeric import Array
from traits.trait_types import Instance, Int, ListFloat, Str, Button, Event, File, Any
from traitsui.api import  View, VGroup
from traitsui.editors.file_editor import FileEditor
from traitsui.editors.range_editor import RangeEditor
from traitsui.editors.tabular_editor import TabularEditor
from traitsui.file_dialog import save_file
from traitsui.group import HGroup
from traitsui.item import Item, Spring, Label
from traitsui.menu import OKCancelButtons
from pyanno.annotations import AnnotationsContainer
from pyanno.ui.arrayview import Array2DAdapter
from pyanno.plots.hinton_plot import HintonDiagramPlot
from pyanno.util import labels_frequency, is_valid, MISSING_VALUE, PyannoValueError
import numpy as np

import logging
logger = logging.getLogger(__name__)


ANNOTATIONS_INFO_STR = """Annotations file {}
Number of annotations: {}
Number of annotators: {}
Number of classes: {}
Labels: {}"""


class DataView(HasTraits):
    data = Array(dtype=object)

    def traits_view(self):
        return View(
            VGroup(
                #HGroup(
                #    Spring(),
                #    Label('Annotators'),
                #    Spring()
                #),
                Item('data',
                    editor=TabularEditor
                        (
                        adapter=Array2DAdapter(ncolumns=len(self.data[0]),
                                               format='%s',
                                               show_index=True)),
                    show_label=False,
                    width=600)
            ),
            title='Annotations',
            width=500,
            height=800,
            resizable=True,
            buttons=OKCancelButtons
        )


class AnnotationsView(HasTraits):
    """ Traits UI Model/View for annotations."""

    # reference to main application
    Application = Any

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
        try:
            frequency =  labels_frequency(
                self.annotations_container.annotations,
                nclasses).tolist()
        except PyannoValueError as e:
            logger.info(e)
            frequency = np.zeros((nclasses,)).tolist()

        self.frequency = frequency
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

    # save current annotations
    save_data = Button(label='Save...')

    def _edit_data_fired(self):
        data_view = DataView(data=self.annotations_container.raw_annotations)
        data_view.edit_traits(kind='modal')
        self.annotations_container = AnnotationsContainer.from_array(
            data_view.data,
            name = self.annotations_container.name
        )
        if self.application is not None:
            self.application.main_window.set_annotations(
                self.annotations_container)


    def _save_data_fired(self):
        save_filename = SaveAnnotationsDialog.open()
        if save_filename is not None:
            self.annotations_container.save_to(save_filename, set_name=True)
            if self.application is not None:
                self.application.main_window.set_annotations(
                    self.annotations_container)


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
            Item('save_data',
                 enabled_when='annotations_are_defined',
                 show_label=False),
            Spring()
        ),
        HGroup(
            Item('frequency_plot',
                 style='custom',
                 resizable=False,
                 show_label=False
            ),
        )
    )

    traits_view = View(body)


class SaveAnnotationsDialog(HasTraits):

    filename = File

    def _filename_default(self):
        import os
        home = os.getenv('HOME')
        return  os.path.join(home, 'annotations.txt')

    @staticmethod
    def open():
        dialog = SaveAnnotationsDialog()
        dialog_ui = dialog.edit_traits(kind='modal')
        if dialog_ui.result:
            # user presser 'OK'
            return dialog.filename
        else:
            return None

    traits_view = View(
        Item('filename', label='Save to:',
             editor=FileEditor(allow_dir=False,
                               dialog_style='save',
                               entries=0),
             style='simple'),
        width = 400,
        resizable = True,
        buttons = ['OK', 'Cancel']
    )

class CreateNewAnnotationsDialog(HasTraits):

    nannotators = Int(8)
    nitems = Int(100)


    @staticmethod
    def create_annotations_dialog():
        dialog = CreateNewAnnotationsDialog()
        dialog_ui = dialog.edit_traits(kind='modal')
        if dialog_ui.result:
            # user pressed 'Ok'
            annotations = np.empty((dialog.nitems, dialog.nannotators),
                                   dtype=int)
            annotations.fill(MISSING_VALUE)
            return annotations
        else:
            return None


    def traits_view(self):
        view = View(
            VGroup(
                Item(
                    'nannotators',
                    editor=RangeEditor(mode='spinner', low=3, high=1000),
                    label='Number of annotators:'
                ),
                Item(
                    'nitems',
                    editor=RangeEditor(mode='spinner', low=2, high=1000000),
                    label='Number of items'
                )
            ),
            buttons = ['OK', 'Cancel']
        )
        return view


#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno import ModelBt
    model = ModelBt.create_initial_state(5)
    annotations = model.generate_annotations(2)

    anno = AnnotationsContainer.from_array(annotations, name='blah')
    model_view = AnnotationsView(annotations_container=anno)
    model_view.configure_traits()
    return model, annotations, model_view


if __name__ == '__main__':
    m, a, mv = main()
