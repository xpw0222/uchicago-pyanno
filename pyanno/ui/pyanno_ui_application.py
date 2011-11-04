# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)

from contextlib import contextmanager
from traits.has_traits import HasTraits
from traits.trait_types import Instance, Bool
import unicodedata
from traitsui.ui import UI
from pyanno.database import PyannoDatabase
from pyanno.modelBt import ModelBt
from pyanno.ui.database_view import DatabaseView
from pyanno.ui.model_data_view import ModelDataView

import logging
logger = logging.getLogger(__name__)


class PyannoApplication(HasTraits):

    database = Instance(PyannoDatabase)

    main_window = Instance(ModelDataView)

    database_window = Instance(DatabaseView)

    database_ui = Instance(UI)


    def open(self):
        self._open_pyanno_database()
        self._open_main_window()


    def close(self):
        self.database.close()


    def _open_main_window(self):
        self.main_window = ModelDataView(application=self)

        model = ModelBt.create_initial_state(5)
        self.main_window.set_model(model)

        self.main_window.configure_traits()


    def _open_pyanno_database(self):
        pass


    def open_database_window(self):
        if (self.database_ui is None
            or self.database_ui.control is None):
                # window was closed or not existent
                logger.debug('Open database window')
                database_window = DatabaseView(database=self.database,
                                               application=self)
                database_ui = database_window.edit_traits(kind='live')

                self.database_window = database_window
                self.database_ui = database_ui
                self.db_window_open = True

        if (self.database_ui is not None
            and self.database_ui.control is not None):
            # windows exists, raise
            self.database_ui.control.Raise()


    def close_database_window(self):
        # wx specific
        self.database_ui.control.Close()
        self.db_window_open = False


    def update_window_from_database_record(self, record):
        """Update main window from pyanno database record.
        """
        self.main_window.set_from_database_record(record)


    def add_current_state_to_database(self):
        mdv = self.main_window

        # file name may contain unicode characters
        data_id = mdv.annotations_view.annotations_container.name
        if data_id is '':
            data_id = 'anonymous_annotations'
        elif type(data_id) is unicode:
            u_data_id = unicodedata.normalize('NFKD', data_id)
            data_id = u_data_id.encode('ascii','ignore')

        self.database.store_result(
            data_id,
            mdv.annotations_view.annotations_container,
            mdv.model,
            mdv.log_likelihood
        )

        if self.db_window_open:
            self.database_window.db_updated = True


    def _create_debug_database(self):
        """Create and populate a test database in a temporary file.
        """

        from tempfile import mktemp
        from pyanno.modelA import ModelA
        from pyanno.modelB import ModelB
        from pyanno.annotations import AnnotationsContainer

        # database filename
        tmp_filename = mktemp(prefix='tmp_pyanno_db_')
        db = PyannoDatabase(tmp_filename)

        def _create_new_entry(model, annotations, id):
            value = model.log_likelihood(annotations)
            ac = AnnotationsContainer.from_array(annotations, name=id)
            db.store_result(id, ac, model, value)

        # populate database
        model = ModelA.create_initial_state(5)
        annotations = model.generate_annotations(100)
        _create_new_entry(model, annotations, 'test_id')

        modelb = ModelB.create_initial_state(5, 8)
        _create_new_entry(modelb, annotations, 'test_id')

        annotations = model.generate_annotations(100)
        _create_new_entry(modelb, annotations, 'test_id2')

        self.database = db


@contextmanager
def pyanno_application():
    app = PyannoApplication()
    yield app
    app.close()



def main():
    """ Entry point for standalone testing/debugging. """

    app = PyannoApplication()
    app._create_debug_database()
    app._open_main_window()
    #app.open_database_window()
    app.close()

    return app

if __name__ == '__main__':
    app = main()
