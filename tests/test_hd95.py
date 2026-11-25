import unittest

import numpy as np

from ctta.metrics import hd95


class HD95Tests(unittest.TestCase):
    def test_identical_masks(self):
        mask = np.zeros((8, 8), dtype=bool)
        mask[1:7, 1:7] = True
        self.assertEqual(hd95(mask, mask), 0.0)

    def test_displaced_points_use_euclidean_distance(self):
        prediction = np.zeros((8, 8), dtype=bool)
        target = np.zeros_like(prediction)
        prediction[0, 0] = target[3, 4] = True
        self.assertEqual(hd95(prediction, target), 5.0)

    def test_maximum_of_the_two_directional_percentiles(self):
        target = np.zeros((2, 11), dtype=bool)
        target[0, 0] = True
        prediction = target.copy()
        prediction[0, 10] = True
        self.assertAlmostEqual(hd95(prediction, target), 9.5)
        self.assertAlmostEqual(hd95(target, prediction), 9.5)

    def test_empty_masks_use_the_valid_grid_diagonal(self):
        empty = np.zeros((4, 5), dtype=bool)
        foreground = np.ones_like(empty)
        self.assertEqual(hd95(empty, foreground), 5.0)
        self.assertEqual(hd95(foreground, empty), 5.0)
        self.assertEqual(hd95(empty, empty), 0.0)
