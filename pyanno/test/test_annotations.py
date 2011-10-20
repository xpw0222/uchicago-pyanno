from cStringIO import StringIO
import numpy as np
import unittest
from pyanno.annotations import Annotations
from pyanno.util import MISSING_VALUE as MV


class TestAnnotations(unittest.TestCase):

    def test_from_file_string_labels(self):
        s = """
        D B A A B
        C A * C A
        B B D D *
        """

        buffer = StringIO(s)
        anno = Annotations._from_file_object(buffer)

        self.assertEqual(anno.nclasses, 4)
        self.assertEqual(anno.nannotators, 5)
        # labels should be sorted
        self.assertEqual(anno.labels, ['A', 'B', 'C', 'D'])
        self.assertEqual(anno.missing_values, ['*'])

        expected = np.array([
            [3, 1, 0, 0, 1],
            [2, 0, MV, 2, 0],
            [1, 1, 3, 3, MV]
        ], dtype=int)
        np.testing.assert_equal(anno.annotations, expected)


    def test_from_file_numerical_labels(self):
        s = """
         1 -1  2
         2  1  3
        -1 -1  3
        """

        buffer = StringIO(s)
        anno = Annotations._from_file_object(buffer)

        expected = np.array([
            [0, MV, 1],
            [1, 0, 2],
            [MV, MV, 2]
        ])
        np.testing.assert_equal(anno.annotations, expected)

        self.assertEqual(anno.nclasses, 3)
        self.assertEqual(anno.labels, ['1', '2', '3'])
        self.assertEqual(anno.missing_values, ['-1'])


    def test_from_file_with_commas(self):
        s = """
        1,2,3,
        -1, -1, 1,
        1, 2, 3
        """

        buffer = StringIO(s)
        anno = Annotations._from_file_object(buffer)

        expected = np.array([
            [0, 1, 2],
            [MV, MV, 0],
            [0, 1, 2]
        ])
        np.testing.assert_equal(anno.annotations, expected)


    def test_from_file_inconsistent_nannotators(self):
        s = """
        1 2 3
        1 2
        1 2 3
        """

        buffer = StringIO(s)
        self.assertRaises(ValueError, Annotations._from_file_object, buffer)

if __name__ == '__main__':
    unittest.main()
