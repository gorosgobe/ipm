import unittest
import numpy as np

from lib.controller import TruePixelROI


class FakePixelEstimator(object):
    def __init__(self, pixel):
        self.pixel = pixel

    def compute_pixel_position(self, handle):
        return self.pixel, None


class FakeObjectWithHandle(object):
    def get_handle(self):
        return -1


class TruePixelRoiTest(unittest.TestCase):
    FAKE_IMAGE = np.array(
        [
            [[1], [2], [3], [4], [5], [6]], # real image would have three channels
            [[7], [8], [9], [10], [11], [12]],
            [[13], [14], [15], [16], [17], [18]],
            [[19], [20], [21], [22], [23], [24]],
        ]
    )

    def test_centered_crop_returns_correctly_cropped_section(self):
        fpe = FakePixelEstimator((2, 3))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[9], [10], [11]],
            [[15], [16], [17]],
            [[21], [22], [23]]
        ]))

    def test_low_crop_compensates_height(self):
        fpe = FakePixelEstimator((3, 3))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[9], [10], [11]],
            [[15], [16], [17]],
            [[21], [22], [23]]
        ]))

    def test_low_crop_compensates_extreme_height(self):
        fpe = FakePixelEstimator((80, 3))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[9], [10], [11]],
            [[15], [16], [17]],
            [[21], [22], [23]]
        ]))

    def test_high_crop_compensates_height(self):
        fpe = FakePixelEstimator((-1, 3))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[3], [4], [5]],
            [[9], [10], [11]],
            [[15], [16], [17]]
        ]))

    def test_high_crop_compensates_extreme_height(self):
        fpe = FakePixelEstimator((-153, 3))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[3], [4], [5]],
            [[9], [10], [11]],
            [[15], [16], [17]]
        ]))

    def test_right_crop_compensates_width(self):
        fpe = FakePixelEstimator((2, 5))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[10], [11], [12]],
            [[16], [17], [18]],
            [[22], [23], [24]]
        ]))

    def test_right_crop_compensates_extreme_width(self):
        fpe = FakePixelEstimator((2, 56))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[10], [11], [12]],
            [[16], [17], [18]],
            [[22], [23], [24]]
        ]))

    def test_left_crop_compensates_width(self):
        fpe = FakePixelEstimator((2, -1))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[7], [8], [9]],
            [[13], [14], [15]],
            [[19], [20], [21]]
        ]))

    def test_left_crop_compensates_extreme_width(self):
        fpe = FakePixelEstimator((2, -37))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[7], [8], [9]],
            [[13], [14], [15]],
            [[19], [20], [21]]
        ]))

    def test_low_left_corner_compensates_both_width_and_height(self):
        fpe = FakePixelEstimator((4, -1))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[7], [8], [9]],
            [[13], [14], [15]],
            [[19], [20], [21]]
        ]))

    def test_top_right_corner_compensates_both_width_and_height(self):
        fpe = FakePixelEstimator((-4, 8))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[4], [5], [6]],
            [[10], [11], [12]],
            [[16], [17], [18]]
        ]))

if __name__ == '__main__':
    unittest.main()
