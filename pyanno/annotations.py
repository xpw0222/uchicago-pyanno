"""Defines objects to create and manipulate raw annotations."""

from traits.has_traits import HasStrictTraits
from traits.trait_numeric import Array
from traits.trait_types import Str, List, Int
from traits.traits import Property

import numpy as np


class Annotations(HasStrictTraits):

    DEFAULT_MISSING_VALUES = ['-1', 'NA', 'None', '*']

    # raw annotations, as they are imported from file or array
    raw_annotations = List(List(Str))

    # name of file or array from which the annotations were imported
    name = Str

    # list of all labels found in file/array
    labels = List(Str)

    # labels corresponding to a missing value
    missing_values = List(Str)

    # names of the annotators if defined
    annotator_names = List(Str)

    # number of classes found in the annotations
    nclasses = Property(Int, depends_on='labels')
    def _get_nclasses(self):
        return len(self.labels)

    # number of annotators
    nannotators = Property(Int, depends_on='raw_annotations')
    def _get_nannotators(self):
        return len(self.raw_annotations[0])

    # annotations
    annotations = Property(Array, depends_on='raw_annotations')
    def _get_annotations(self):
        nitems, nannotators = len(self.raw_annotations), self.nannotators
        anno = np.empty((nitems, nannotators), dtype=int)

        # build map from labels and missing values to annotation values
        raw2val = dict(zip(self.labels, range(self.nclasses)))
        raw2val.update([(mv, -1) for mv in self.missing_values])

        # translate
        for i, row in enumerate(self.raw_annotations):
            for j, lbl in enumerate(row):
                anno[i,j] = raw2val[lbl]

        return anno


    @staticmethod
    def _from_file_object(fobj, missing_values=None, name=''):
        """Useful for testing, as it can be called using a StringIO object.
        """

        if missing_values is None:
            missing_values = Annotations.DEFAULT_MISSING_VALUES

        missing_set = set(missing_values)

        raw_annotations = []
        labels_set = set()
        nannotators = None
        for n, line in enumerate(fobj):
            # remove commas and split in individual tokens
            line = line.strip().replace(',', ' ')

            # ignore empty lines
            if len(line) == 0: continue

            labels = line.split()

            # verify that number of lines is consistent in the whole file
            if nannotators is None: nannotators = len(labels)
            else:
                if len(labels) != nannotators:
                    raise ValueError('File has inconsistent number of entries '
                                     'on separate lines (line {})'.format(n))

            raw_annotations.append(labels)
            labels_set.update(labels)

        # remove missing values from set of labels
        missing_values = sorted(list(missing_set & labels_set))
        all_labels =  sorted(list(labels_set - missing_set))

        # create annotations object
        anno = Annotations(
            raw_annotations = raw_annotations,
            labels = all_labels,
            missing_values = missing_values,
            name = name
        )

        return anno


    @staticmethod
    def from_file(filename, missing_values=None):
        """Load annotations from a file.

        The file is a text file with a columns separated by spaces and/or
        commas, and rows on different lines.

        Input:
        filename -- file name
        missing_values -- list of labels that are considered missing values.
           Default is ['-1', 'NA', 'None', '*']

        """

        if missing_values is None:
            missing_values = Annotations.DEFAULT_MISSING_VALUES

        with open(filename) as fh:
            anno = Annotations._from_file_object(fh,
                                                 missing_values=missing_values,
                                                 name=filename)

        return anno
