# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)
from contextlib import closing
import os
from shelve import Shelf
import shelve
from tempfile import TemporaryFile, tempdir, mktemp, gettempdir

import unittest
import pyanno
from pyanno.database import PyannoDatabase, PyannoResult
from pyanno.util import PyannoValueError
import numpy as np


class TestDatabase(unittest.TestCase):

    def setUp(self):
        self.tmp_filename = mktemp(prefix='tmp_pyanno_db_')

        # fixtures
        self.model1 = pyanno.ModelA.create_initial_state(4)
        self.annotations1 = self.model1.generate_annotations(100)
        self.value1 = self.model1.log_likelihood(self.annotations1)
        self.data_id1 = 'bogus.txt'

        self.model2 = pyanno.ModelA.create_initial_state(4)
        self.annotations2 = self.model2.generate_annotations(100)
        self.value2 = self.model2.log_likelihood(self.annotations2)
        self.data_id2 = 'bogus2.txt'



    def tearDown(self):
        # remove temp file
        try:
            os.remove(self.tmp_filename)
        except OSError:
            # ignore, the file was not created by the test
            pass


    def test_database_create_and_close(self):
        with closing(PyannoDatabase(self.tmp_filename)) as db:
            self.assertEqual(len(db.database), 0)
            self.assertEqual(db.db_filename, self.tmp_filename)
            self.assertFalse(db.closed)

        self.assertTrue(db.closed)

        # make sure database is closed
        with self.assertRaisesRegexp(ValueError, 'closed shelf'):
            db.database['a'] = 2


    def test_storage(self):
        with closing(PyannoDatabase(self.tmp_filename)) as db:
            db.store_result(self.data_id1, self.annotations1,
                            self.model1, self.value1)
            self.assertEqual(len(db.database), 1)

            results = db.retrieve_id(self.data_id1)
            self.assert_(isinstance(results, list))
            self.assertEqual(len(results), 1)
            self.assert_(isinstance(results[0], PyannoResult))

            np.testing.assert_equal(results[0].annotations, self.annotations1)
            self.assertEqual(results[0].model.nclasses, self.model1.nclasses)
            self.assertEqual(results[0].value, self.value1)
            self.assert_(isinstance(results[0].model, self.model1.__class__))

            # store new entry for different annotations
            db.store_result(self.data_id2, self.annotations2,
                            self.model2, self.value2)
            self.assertEqual(len(db.database), 2)

            # add new result for same annotations
            db.store_result(self.data_id1, self.annotations1,
                            self.model2, self.value2)
            results = db.retrieve_id(self.data_id1)
            self.assertEqual(len(results), 2)
            self.assertEqual(results[1].value, self.value2)


    def test_conflict(self):
        # behavior: one cannot store two different annotations with same id
        with closing(PyannoDatabase(self.tmp_filename)) as db:
            db.store_result(self.data_id1, self.annotations1,
                            self.model1, self.value1)

            with self.assertRaises(PyannoValueError):
                db.store_result(self.data_id1, self.annotations2,
                                self.model1, self.value1)

                
    def test_persistence(self):
        with closing(PyannoDatabase(self.tmp_filename)) as db:
            db.store_result(self.data_id1, self.annotations1,
                            self.model1, self.value1)

        with closing(PyannoDatabase(self.tmp_filename)) as db:
            results = db.retrieve_id(self.data_id1)
            np.testing.assert_equal(results[0].annotations, self.annotations1)


if __name__ == '__main__':
    unittest.main()