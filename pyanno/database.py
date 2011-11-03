# Copyright (c) 2011, Enthought, Ltd.
# Author: Pietro Berkes <pberkes@enthought.com>
# License: Modified BSD license (2-clause)
from collections import namedtuple
from contextlib import contextmanager
import shelve
from traits.has_traits import HasStrictTraits
from traits.trait_numeric import Array
from traits.trait_types import Str, Any, Float, Instance
from pyanno.annotations import AnnotationsContainer
from pyanno.util import PyannoValueError
import numpy as np

class PyannoResult(HasStrictTraits):
    anno_container = Instance(AnnotationsContainer)
    # TODO change into Instance(AbstractModel)
    model = Any
    value = Float


class PyannoDatabase(object):

    def __init__(self, filename):
        self.db_filename = filename
        self.database = shelve.open(filename, flag='c', protocol=2)
        self.closed = False


    def store_result(self, data_id, anno_container, model, value):
        """Store a pyAnno result in the database.

        The `data_id` must be a **unique** identifier for an annotations
        set.

        Parameters
        ----------
        data_id : string
            Readable **unique** identifier for the annotations set (e.g.,
            the file name where the annotations are stored).

        anno_container : AnnotationsContainer
            An annotations container (see :class:`~AnnotationsContainer).

        model : object
            pyAnno model object instance

        value : float
            value of the objective function for the model-annotations pair,
            typically the log likelihood of the annotations given the model
        """

        entry = PyannoResult(anno_container=anno_container,
                             model=model, value=value)
        self._check_consistency(data_id, anno_container)

        # NOTE shelves to not automatically handle changing mutable values,
        # we need to take care of it manually
        if not self.database.has_key(data_id):
            temp = []
        else:
            temp = self.database[data_id]

        temp.append(entry)

        self.database[data_id] = temp


    def retrieve_id(self, data_id):
        return self.database[data_id]


    def close(self):
        """Close database."""
        self.database.close()
        self.closed = True


    def _check_consistency(self, data_id, anno_container):
        """Make sure that all entries with same ID have the same annotations.
        """
        if self.database.has_key(data_id):
            previous = self.database[data_id][0]
            # check that the new annotations are the same as the previous ones
            if not np.all(previous.anno_container.annotations ==
                          anno_container.annotations):
                msg = ('Conflicting annotations with same ID. Please '
                       'rename the new entry.')
                raise PyannoValueError(msg)
