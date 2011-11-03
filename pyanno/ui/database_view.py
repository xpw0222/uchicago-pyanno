# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)
from traits.has_traits import HasTraits, on_trait_change
from traits.trait_types import List, Instance, Event, Str, Float
from traitsui.editors.table_editor import TableEditor
from traitsui.group import Group
from traitsui.handler import ModelView
from traitsui.item import Item
from traitsui.table_column import ObjectColumn
from traitsui.table_filter import TableFilter
from traitsui.view import View
from pyanno.database import PyannoDatabase

class DBEntry(HasTraits):
    data_id = Str
    model_name = Str
    value = Str

    @staticmethod
    def from_result(data_id, result):
        return DBEntry(
            data_id = data_id,
            model_name = str(result.model.__class__),
            value = '%.4f' % result.value
        )

    traits_view = View(
        Item('data_id'),
        Item('model_name'),
        Item('value')
    )


class SearchIDFilter(TableFilter):
    name = 'Search data ID filter'

    id_substr = Str('')

    def filter(self, entry):
        return self.id_substr in entry.data_id

    filter_view = Group(
        Item('id_substr', label='Search annotation IDs')
    )


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
)



class DatabaseView(HasTraits):

    database = Instance(PyannoDatabase)

    # trait displayed in the TableView, it is synchronized with the actual db
    results_table = List(DBEntry)

    # event raised when the database entries changed
    db_updated = Event


    @on_trait_change('db_updated,database')
    def _create_results_table(self):
        db = self.database.database

        table = []
        for data_id in db.keys():
            results = db[data_id]
            for result in results:
                table.append(DBEntry.from_result(data_id, result))

        # sort entries by hand as the sortable columns in wx TableEditor do
        # not work
        table.sort(key=lambda x: (x.data_id, x.value))

        self.results_table = table


    traits_view = View(
        Item('results_table',
             editor = db_table_editor,
             show_label = False,
        ),
        title     = 'pyAnno results database',
        width     = 800,
        height    = 400,
        resizable = True,
        #buttons   = [ 'OK' ],
        kind      = 'live'
    )


def main():
    """ Entry point for standalone testing/debugging. """

    from tempfile import mktemp
    from contextlib import closing

    # database filename
    tmp_filename = mktemp(prefix='tmp_pyanno_db_')
    with closing(PyannoDatabase(tmp_filename)) as db:
        from pyanno.modelA import ModelA
        from pyanno.modelB import ModelB
        model = ModelA.create_initial_state(5)
        annotations = model.generate_annotations(100)
        value = model.log_likelihood(annotations)
        db.store_result('test_id', annotations, model, value)


        modelb = ModelB.create_initial_state(5, 8)
        value = modelb.log_likelihood(annotations)
        db.store_result('test_id', annotations, modelb, value)


        annotations = model.generate_annotations(100)
        value = model.log_likelihood(annotations)
        db.store_result('test_id2', annotations, model, value)

        model_data_view = DatabaseView(database=db)
        model_data_view.configure_traits(view='traits_view')

    return model, model_data_view


if __name__ == '__main__':
    m, mdv = main()
