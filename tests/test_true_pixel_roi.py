import unittest
import numpy as np

from cv.controller import TruePixelROI


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
            [[1], [2], [3], [4], [5], [6]],  # real image would have three channels
            [[7], [8], [9], [10], [11], [12]],
            [[13], [14], [15], [16], [17], [18]],
            [[19], [20], [21], [22], [23], [24]],
        ]
    )

    CENTER_16_3x3_PIXELS = [(3, 2), (2, 1), (4, 1), (2, 3), (4, 3)]
    CENTER_10_3x3_PIXELS = [(3, 1), (2, 0), (4, 0), (2, 2), (4, 2)]
    CENTER_17_3x3_PIXELS = [(4, 2), (3, 1), (5, 1), (3, 3), (5, 3)]
    CENTER_14_3x3_PIXELS = [(1, 2), (0, 1), (2, 1), (0, 3), (2, 3)]
    CENTER_11_3x3_PIXELS = [(4, 1), (3, 0), (5, 0), (3, 2), (5, 2)]
    EVEN_START_9_2x2_PIXELS = [(2, 1), (2, 1), (3, 1), (2, 2), (3, 2)]
    ODD_WIDTH_EVEN_HEIGHT_2x3_PIXELS = [(2, 1), (1, 1), (3, 1), (1, 2), (3, 2)]
    EVEN_WIDTH_ODD_HEIGHT_3x2_PIXELS = [(2, 1), (2, 0), (3, 0), (2, 2), (3, 2)]

    def test_centered_crop_returns_correctly_odd_cropped_section(self):
        fpe = FakePixelEstimator((3, 2))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop, pixels = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[9], [10], [11]],
            [[15], [16], [17]],
            [[21], [22], [23]]
        ]))
        self.assertEqual(pixels, self.CENTER_16_3x3_PIXELS)

    def test_centered_crop_returns_correctly_even_cropped_section(self):
        fpe = FakePixelEstimator((2, 1))
        roi_estimator = TruePixelROI(2, 2, fpe, FakeObjectWithHandle())
        crop, pixels = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[9], [10]],
            [[15], [16]]
        ]))
        self.assertEqual(pixels, self.EVEN_START_9_2x2_PIXELS)

    def test_centered_crop_returns_correctly_odd_width_even_height_section(self):
        fpe = FakePixelEstimator((2, 1))
        roi_estimator = TruePixelROI(2, 3, fpe, FakeObjectWithHandle())
        crop, pixels = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[8], [9], [10]],
            [[14], [15], [16]]
        ]))
        self.assertEqual(pixels, self.ODD_WIDTH_EVEN_HEIGHT_2x3_PIXELS)

    def test_centered_crop_returns_correctly_even_width_odd_height_section(self):
        fpe = FakePixelEstimator((2, 1))
        roi_estimator = TruePixelROI(3, 2, fpe, FakeObjectWithHandle())
        crop, pixels = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[3], [4]],
            [[9], [10]],
            [[15], [16]]
        ]))
        self.assertEqual(pixels, self.EVEN_WIDTH_ODD_HEIGHT_3x2_PIXELS)

    def test_low_crop_compensates_height(self):
        fpe = FakePixelEstimator((3, 3))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop, pixels = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[9], [10], [11]],
            [[15], [16], [17]],
            [[21], [22], [23]]
        ]))
        self.assertEqual(pixels, self.CENTER_16_3x3_PIXELS)

    def test_low_crop_compensates_extreme_height(self):
        fpe = FakePixelEstimator((3, 80))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop, pixels = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[9], [10], [11]],
            [[15], [16], [17]],
            [[21], [22], [23]]
        ]))
        self.assertEqual(pixels, self.CENTER_16_3x3_PIXELS)

    def test_high_crop_compensates_height(self):
        fpe = FakePixelEstimator((3, -1))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop, pixels = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[3], [4], [5]],
            [[9], [10], [11]],
            [[15], [16], [17]]
        ]))
        self.assertEqual(pixels, self.CENTER_10_3x3_PIXELS)

    def test_high_crop_compensates_extreme_height(self):
        fpe = FakePixelEstimator((3, -153))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop, pixels = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[3], [4], [5]],
            [[9], [10], [11]],
            [[15], [16], [17]]
        ]))
        self.assertEqual(pixels, self.CENTER_10_3x3_PIXELS)

    def test_right_crop_compensates_width(self):
        fpe = FakePixelEstimator((5, 2))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop, pixels = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[10], [11], [12]],
            [[16], [17], [18]],
            [[22], [23], [24]]
        ]))
        self.assertEqual(pixels, self.CENTER_17_3x3_PIXELS)

    def test_right_crop_compensates_extreme_width(self):
        fpe = FakePixelEstimator((56, 2))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop, pixels = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[10], [11], [12]],
            [[16], [17], [18]],
            [[22], [23], [24]]
        ]))
        self.assertEqual(pixels, self.CENTER_17_3x3_PIXELS)

    def test_left_crop_compensates_width(self):
        fpe = FakePixelEstimator((-1, 2))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop, pixels = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[7], [8], [9]],
            [[13], [14], [15]],
            [[19], [20], [21]]
        ]))
        self.assertEqual(pixels, self.CENTER_14_3x3_PIXELS)

    def test_left_crop_compensates_extreme_width(self):
        fpe = FakePixelEstimator((-37, 2))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop, pixels = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[7], [8], [9]],
            [[13], [14], [15]],
            [[19], [20], [21]]
        ]))
        self.assertEqual(pixels, self.CENTER_14_3x3_PIXELS)

    def test_low_left_corner_compensates_both_width_and_height(self):
        fpe = FakePixelEstimator((-1, 4))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop, pixels = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[7], [8], [9]],
            [[13], [14], [15]],
            [[19], [20], [21]]
        ]))
        self.assertEqual(pixels, self.CENTER_14_3x3_PIXELS)

    def test_top_right_corner_compensates_both_width_and_height(self):
        fpe = FakePixelEstimator((8, -4))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle())
        crop, pixels = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_array_equal(crop, np.array([
            [[4], [5], [6]],
            [[10], [11], [12]],
            [[16], [17], [18]]
        ]))
        self.assertEqual(pixels, self.CENTER_11_3x3_PIXELS)

    def test_spatial_dimensions_are_added(self):
        fpe = FakePixelEstimator((3, 2))
        roi_estimator = TruePixelROI(3, 3, fpe, FakeObjectWithHandle(), add_spatial_maps=True)
        crop, pixels = roi_estimator.crop(self.FAKE_IMAGE)
        np.testing.assert_allclose(crop, np.array([
            [[9, 2 / 6, 1 / 4], [10, 3 / 6, 1 / 4], [11, 4 / 6, 1 / 4]],
            [[15, 2 / 6, 2 / 4], [16, 3 / 6, 2 / 4], [17, 4 / 6, 2 / 4]],
            [[21, 2 / 6, 3 / 4], [22, 3 / 6, 3 / 4], [23, 4 / 6, 3 / 4]]
        ]))
        self.assertEqual(pixels, self.CENTER_16_3x3_PIXELS)


if __name__ == '__main__':
    unittest.main()
