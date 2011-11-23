# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

from traits.has_traits import HasTraits, on_trait_change
from traits.trait_numeric import Array
from traits.trait_types import (Instance, Int, ListFloat, Button, Event, File,
                                Any)
from traits.traits import Property
from traitsui.api import  View, VGroup
from traitsui.editors.file_editor import FileEditor
from traitsui.editors.range_editor import RangeEditor
from traitsui.editors.tabular_editor import TabularEditor
from traitsui.group import HGroup, VGrid, Group
from traitsui.handler import ModelView
from traitsui.item import Item, Spring, Label
from traitsui.menu import OKCancelButtons
from pyanno.annotations import AnnotationsContainer
from pyanno.ui.appbase.wx_utils import is_display_small
from pyanno.ui.arrayview import Array2DAdapter
from pyanno.plots.hinton_plot import HintonDiagramPlot
from pyanno.util import labels_frequency, MISSING_VALUE, PyannoValueError
import numpy as np

import logging
logger = logging.getLogger(__name__)


WIDTH_CELL = 60
MAX_WIDTH = 1000
W_MARGIN = 150

class DataView(HasTraits):
    data = Array(dtype=object)

    def traits_view(self):
        ncolumns = len(self.data[0])
        w_table = min(WIDTH_CELL * ncolumns, MAX_WIDTH)
        w_view = min(w_table + W_MARGIN, MAX_WIDTH)
        return View(
            Group(
                Item('data',
                    editor=TabularEditor
                        (
                        adapter=Array2DAdapter(ncolumns=ncolumns,
                                               format='%s',
                                               show_index=True)),
                    show_label=False,
                    width=w_table,
                    padding=10),
            ),
            title='Annotations',
            width=w_view,
            height=800,
            resizable=True,
            buttons=OKCancelButtons
        )


class AnnotationsView(ModelView):
    """ Traits UI Model/View for annotations."""

    # reference to main application
    application = Any

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
            logger.debug(e)
            frequency = np.zeros((nclasses,)).tolist()

        self.frequency = frequency
        self.frequency_plot = HintonDiagramPlot(
            data=self.frequency,
            title='Observed label frequencies')


    ### Traits UI definitions ###

    # event raised when annotations are updated
    annotations_updated = Event

    ## frequency plot definition
    frequency_plot = Instance(HintonDiagramPlot)

    ## edit data button opens annotations editor
    edit_data = Button(label='Edit annotations...')

    # save current annotations
    save_data = Button(label='Save annotations...')

    def _edit_data_fired(self):
        data_view = DataView(data=self.annotations_container.raw_annotations)
        data_view.edit_traits(kind='livemodal', parent=self.info.ui.control)
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

    _name = Property
    def _get__name(self):
        return self.annotations_container.name

    _nitems = Property
    def _get__nitems(self):
        return self.annotations_container.nitems

    _nclasses = Property
    def _get__nclasses(self):
        return self.annotations_container.nclasses

    _labels = Property
    def _get__labels(self):
        return str(self.annotations_container.labels)

    _nannotators = Property
    def _get__nannotators(self):
        return str(self.annotations_container.nannotators)

    def traits_view(self):
        if is_display_small():
            w_view = 350
        else:
            w_view = 450

        info_group = VGroup(
            Item('_name',
                 label='Annotations name:',
                 style='readonly',
                 padding=0),
            VGrid(
                Item('_nclasses',
                     label='Number of classes:',
                     style='readonly',
                     width=10),
                Item('_labels',
                     label='Labels:',
                     style='readonly'),
                Item('_nannotators',
                     label='Number of annotators:',
                     style='readonly', width=10),
                Item('_nitems',
                     label='Number of items:',
                     style='readonly'),
                padding=0
            ),
            padding=0
        )


        body = VGroup(
            info_group,

            Item('_'),

            HGroup(
                VGroup(
                    Spring(),
                    Item('frequency_plot',
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
                    Item('edit_data',
                         enabled_when='annotations_are_defined',
                         show_label=False),
                    Item('save_data',
                         enabled_when='annotations_are_defined',
                         show_label=False),
                    Spring()
                )
            ),

            Spring(),
            Item('_'),

        )

        traits_view = View(body)
        return traits_view


class SaveAnnotationsDialog(HasTraits):

    filename = File

    def _filename_default(self):
        import os
        home = os.getenv('HOME') or os.getenv('HOMEPATH')
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
             style='simple'
        ),
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
                ),
            ),
            buttons = ['OK', 'Cancel']
        )
        return view


#### Testing and debugging ####################################################

def main():
    """ Entry point for standalone testing/debugging. """

    from pyanno.modelBt_loopdesign import ModelBtLoopDesign
    model = ModelBtLoopDesign.create_initial_state(5)
    annotations = model.generate_annotations(2)

    anno = AnnotationsContainer.from_array(annotations, name='blah')
    model_view = AnnotationsView(annotations_container=anno, model=HasTraits())
    model_view.configure_traits()
    return model, annotations, model_view


if __name__ == '__main__':
    m, a, mv = main()
