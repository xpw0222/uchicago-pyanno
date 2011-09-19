from pyanno.kappa import *
from pyanno.multinom import *
from pyanno.util import *

import unittest


class TestUtil(unittest.TestCase):
    def test_vec_copy(self):
        c = []
        d = []
        vec_copy(c, d)
        self.assertEquals(c, d)
        a = [1, 2, 3]
        b = [4, 5, 6]
        vec_copy(a, b)
        self.assertEquals(a, b)

    def test_fill_vec(self):
        a = []
        fill_vec(a, 0)
        self.assertEquals([], a)
        b = [1]
        fill_vec(b, 3)
        self.assertEquals([3], b)
        c = [1, 2, 3]
        fill_vec(c, 5)
        self.assertEquals([5, 5, 5], c)

    def test_fill_mat(self):
        a = [[]]
        fill_mat(a, 0)
        self.assertEquals([[]], a)
        b = [[1, 2, 3]]
        fill_mat(b, 5)
        self.assertEquals([[5, 5, 5]], b)
        c = [[1, 2], [3, 4]]
        fill_mat(c, 6)
        self.assertEquals([[6, 6], [6, 6]], c)

    def test_fill_tens(self):
        a = [[[]]]
        fill_tens(a, 0)
        self.assertEquals([[[]]], a)
        b = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        fill_tens(b, 10)
        self.assertEquals([[[10, 10], [10, 10]], [[10, 10], [10, 10]]], b)

    def test_alloc_vec(self):
        a = alloc_vec(0)
        self.assertEquals([], a)
        b = alloc_vec(1)
        self.assertEquals([0.0], b)
        c = alloc_vec(2, 9.0)
        self.assertEquals([9.0, 9.0], c)

    def test_alloc_mat(self):
        a = alloc_mat(1, 0)
        self.assertEquals([[]], a)
        b = alloc_mat(1, 1, 3.0)
        self.assertEquals([[3.0]], b)
        c = alloc_mat(2, 3, 5.0)
        self.assertEquals([[5.0, 5.0, 5.0], [5.0, 5.0, 5.0]], c)

    def test_alloc_tens(self):
        a = alloc_tens(1, 1, 0)
        self.assertEquals([[[]]], a)
        b = alloc_tens(1, 1, 1)
        self.assertEquals([[[0.0]]], b)
        c = alloc_tens(2, 3, 4, 9)
        self.assertEquals([[[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]],
            [[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]]], c)

    def test_alloc_tens4(self):
        a = alloc_tens4(1, 1, 1, 0)
        self.assertEquals([[[[]]]], a)
        b = alloc_tens4(1, 1, 1, 1)
        self.assertEquals([[[[0.0]]]], b)
        c = alloc_tens4(1, 2, 3, 4, 9)
        self.assertEquals([[[[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]],
            [[9, 9, 9, 9], [9, 9, 9, 9], [9, 9, 9, 9]]]], c)

    def test_vec_sum(self):
        self.assertEquals(0, vec_sum([]))
        self.assertEquals(1, vec_sum([1]))
        self.assertEquals(3, vec_sum([1, 2]))

    def test_mat_sum(self):
        self.assertEquals(0, mat_sum([[]]))
        self.assertEquals(1, mat_sum([[1]]))
        self.assertEquals(3, mat_sum([[1, 2]]))
        self.assertEquals(21, mat_sum([[1, 2, 3], [4, 5, 6]]))

    def test_prob_norm(self):
        theta = [0.2]
        prob_norm(theta)
        self.assert_prob_normed(theta)

    def assert_prob_normed(self, theta):
        self.assert_(len(theta) > 0)
        for theta_i in theta:
            self.assert_(theta_i >= 0.0)
            self.assert_(theta_i <= 1.0)
        self.assertAlmostEqual(1.0, vec_sum(theta), 3)


class TestMultinom(unittest.TestCase):
    def testbase2(self):
        self.assertEquals(2, 2)


if __name__ == '__main__':
    unittest.main()
