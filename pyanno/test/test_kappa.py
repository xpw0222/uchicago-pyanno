import unittest
from pyanno.kappa import agr, s, pi, kappa, global_prevalence, K, chance_adj_agr
from numpy.testing import assert_array_almost_equal


class TestKappa(unittest.TestCase):
    def test_agr(self):
        cm1 = [[1]]
        self.assertEquals(1.0, agr(cm1))
        cm2 = [[41, 3], [9, 47]]
        self.assertAlmostEqual((41.0 + 47.0) / 100.0, agr(cm2))
        cm3 = [[44, 6], [6, 44]]
        self.assertAlmostEqual(0.88, agr(cm3))

    def test_s(self):
        cm1 = [[44, 6], [6, 44]]
        self.assertAlmostEqual((0.88 - 0.5) / (1.0 - 0.5), s(cm1))
        cm2 = [[44, 6, 0, 0], [6, 44, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]]
        self.assertAlmostEqual(0.84, s(cm2))
        self.assertAlmostEqual(0.88, agr(cm2))

    def test_pi(self):
        cm1 = [[44, 6, 0], [6, 44, 0], [0, 0, 0]]
        self.assertAlmostEqual(0.88, agr(cm1))
        self.assertAlmostEqual(0.82, s(cm1))
        self.assertAlmostEqual(0.76, pi(cm1))
        cm2 = [[77, 1, 2], [1, 6, 3], [2, 3, 5]]
        self.assertAlmostEqual(0.88, agr(cm2))
        self.assertAlmostEqual(0.82, s(cm2))
        self.assertAlmostEqual((0.88 - 0.66) / (1.0 - 0.66), pi(cm2))
        cm3 = [[990, 5], [5, 0]]
        self.assertAlmostEqual(0.99, agr(cm3))
        self.assertAlmostEqual(0.98, s(cm3))
        self.assertAlmostEqual((0.99 - 0.99005) / (1.0 - 0.99005), pi(cm3))

    def test_kappa(self):
        cm1 = [[8, 1],
            [4, 2]]
        ce = ((12.0 * 9.0) + (6.0 * 3.0)) / (15.0 * 15.0)
        ag = 10.0 / 15.0
        k = (ag - ce) / (1.0 - ce)
        self.assertAlmostEqual(k, kappa(cm1))
        cm2 = [[7, 5, 0],
            [1, 9, 2],
            [0, 8, 4]]
        ce2 = (8.0 * 12.0 + 22.0 * 12.0 + 6.0 * 12.0) / (36.0 * 36.0)
        ag2 = 20.0 / 36.0
        k2 = (ag2 - ce2) / (1.0 - ce2)
        self.assertAlmostEqual(k2, kappa(cm2))

    def test_global_prevalence(self):
        item = [0, 0]
        label = [0, 1]
        theta_hat = global_prevalence(item, label)
        assert_array_almost_equal([0.5, 0.5], theta_hat)
        item2 = [0, 0, 0, 1, 1]
        label2 = [0, 0, 1, 1, 2]
        theta_hat2 = global_prevalence(item2, label2)
        assert_array_almost_equal([4.0 / 12.0, 5.0 / 12.0, 3.0 / 12.0],
                                  theta_hat2)

    def test_K(self):
        item = [0, 0, 0]
        anno = [0, 1, 2]
        lab = [0, 0, 1]
        ag = 1.0 / 3.0
        ag_exp = (1.0 * 1.0) / (3.0 * 3.0) + (2.0 * 2.0) / (3.0 * 3.0)
        Ke = (ag - ag_exp) / (1.0 - ag_exp)
        self.assertAlmostEqual(Ke, K(item, anno, lab))

    def test_K_2(self):
        item = [0, 0, 0, 1, 1, 2]
        anno = [0, 1, 2, 0, 1, 0]
        lab = [0, 0, 1, 1, 1, 0]
        ag = 2.0 / 4.0
        ag_exp = (5.0 / 9.0) ** 2 + (4.0 / 9.0) ** 2
        Ke = (ag - ag_exp) / (1.0 - ag_exp)
        self.assertAlmostEqual(Ke, K(item, anno, lab))

    def test_K_3(self):
        item = [0, 0, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2]
        anno = [0, 1, 2, 0, 1, 0, 1, 2, 3, 4, 5, 6]
        lab = [0, 0, 1, 1, 1, 0, 1, 2, 1, 0, 1, 2]
        ag = (1.0 + 1.0 + 1.0 + 3.0 + 1.0) / (3.0 + 1.0 + 21.0)
        ag_exp = (20.0 / 63.0) ** 2 + (37.0 / 63.0) ** 2 + (6.0 / 63.0) ** 2
        Ke = (ag - ag_exp) / (1.0 - ag_exp)
        self.assertAlmostEqual(Ke, K(item, anno, lab))

    def test_chance_adj_agr(self):
        self.assertAlmostEqual((0.9 - 0.5) / (1.0 - 0.5),
                               chance_adj_agr(0.9, 0.5))


if __name__ == '__main__':
    unittest.main()
