# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import List, Instance, Event, Str, Float, Button, Int
from traitsui.editors.table_editor import TableEditor
from traitsui.group import Group, VGroup, HGroup
from traitsui.item import Item, Spring
from traitsui.table_column import ObjectColumn
from traitsui.table_filter import TableFilter
from traitsui.view import View
from pyanno.database import PyannoDatabase
from pyanno.ui.model_data_view import ModelDataView

#import logging
#logger = logging.getLogger(__name__)


class DBEntry(HasTraits):
    data_id = Str
    model_name = Str
    value = Str
    idx = Int

    @staticmethod
    def from_result(idx, data_id, result):
        return DBEntry(
            idx = idx,
            data_id = data_id,
            model_name = str(result.model.__class__),
            value = '%.4f' % result.value
        )


class SearchIDFilter(TableFilter):
    name = 'Search data ID filter'

    id_substr = Str('')

    def filter(self, entry):
        return self.id_substr in entry.data_id

    filter_view = Group(
        Item('id_substr', label='Search annotation IDs')
    )



class DatabaseView(HasTraits):

    # reference to pyanno database
    database = Instance(PyannoDatabase)

    # reference to main window
    model_data_view = Instance(ModelDataView)

    # trait displayed in the TableView, it is synchronized with the actual db
    results_table = List(DBEntry)


    # currently selected entry
    current_selection = Instance(DBEntry)


    # event raised when the database entries changed
    db_updated = Event

    # button that sends the currently selected row to the main window
    open_to_main = Button('Open results...')


    @on_trait_change('db_updated,database')
    def _create_results_table(self):
        db = self.database.database

        table = []
        for data_id in db.keys():
            results = db[data_id]
            for idx, result in enumerate(results):
                table.append(DBEntry.from_result(idx, data_id, result))

        # sort entries by hand as the sortable columns in wx TableEditor do
        # not work
        table.sort(key=lambda x: (x.data_id, x.value))

        self.results_table = table


    def find_database_record(self, entry):
        """Given an entry in the results_table, find the corresponding
        database record."""
        return self.database.retrieve_id(entry.data_id)[entry.idx]


    @on_trait_change('open_to_main')
    def forward_to_model_data_view(self):
        # called on double click, it sets the model and the annotations in
        # the main window

        mdv = self.model_data_view
        if mdv is not None:
            entry = self.current_selection
            record = self.find_database_record(entry)
            mdv.set_from_database_record(record)


    def track_selection(self, entry):
        # update current selection each time the selection is changes
        self.current_selection = entry


    def traits_view(self):
        db_table_editor = TableEditor(
            columns=[
                ObjectColumn(name='data_id', label='Annotations ID',
                             editable=False, width=0.20),
                ObjectColumn(name='model_name',
                             editable=False, width=0.20),
                ObjectColumn(name='value', label='Log likelihood',
                             editable=False, width=0.10)
            ],
            search      = SearchIDFilter(),
            editable    = True,
            deletable   = True,
            configurable = False,
            sortable    = True,
            sort_model  = False,
            auto_size   = False,
            orientation = 'vertical',
            show_toolbar = True,
            selection_mode = 'row',
            on_dclick   = self.forward_to_model_data_view,
            on_select   = self.track_selection
        )

        traits_view = View(
            VGroup(
                Item('results_table',
                     editor = db_table_editor,
                     show_label = False,
                ),
                HGroup(
                    Item('open_to_main',
                         show_label=False,
                         width=100),
                    Spring()
                )
            ),
            title     = 'pyAnno results database',
            width     = 500,
            height    = 400,
            resizable = True,
            #buttons   = [ 'OK' ],
            kind      = 'live'
        )

        return traits_view



#### Testing and debugging ####################################################

from pyanno.modelA import ModelA
from pyanno.modelB import ModelB
from pyanno.annotations import AnnotationsContainer

def _create_new_entry(model, annotations, id, db):
    value = model.log_likelihood(annotations)
    ac = AnnotationsContainer.from_array(annotations)
    db.store_result(id, ac, model, value)


def main():
    """ Entry point for standalone testing/debugging. """

    from tempfile import mktemp
    from contextlib import closing

    # database filename
    tmp_filename = mktemp(prefix='tmp_pyanno_db_')
    with closing(PyannoDatabase(tmp_filename)) as db:
        # populate database
        model = ModelA.create_initial_state(5)
        annotations = model.generate_annotations(100)
        _create_new_entry(model, annotations, 'test_id', db)

        modelb = ModelB.create_initial_state(5, 8)
        _create_new_entry(modelb, annotations, 'test_id', db)

        annotations = model.generate_annotations(100)
        _create_new_entry(modelb, annotations, 'test_id2', db)

        # create view
        model_data_view = DatabaseView(database=db)
        model_data_view.configure_traits(view='traits_view')

    return model, model_data_view


if __name__ == '__main__':
    m, mdv = main()
