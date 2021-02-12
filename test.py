import unittest
from tensorcheck import *

import numpy as np

class TestTensorChecker(unittest.TestCase):

    def test_supports_args_and_kwargs(self):
        try:
            @tensorcheck({
                "a": {"type": np.ndarray},
                "b": {"type": np.ndarray},
            })
            def inference(a, b):
                return

            x = np.random.randn(1, 1, 3, 2)
            y = np.random.randn(1, 1, 2, 2)
            inference(x, b=y)

        except AnnotationException:
            self.fail()


    def test_annotated_key_must_be_parameter(self):
        with self.assertRaises(AnnotationException):
            @tensorcheck({
                "b": {"type": np.ndarray},
            })
            def inference(a):
                return

            x = np.random.randn(1, 1, 3, 2)
            y = np.random.randn(1, 1, 2, 2)
            inference(x)


    def test_dtype_success(self):
        try:
            @tensorcheck({
                "a": {"type": np.ndarray, "dtype": np.float},
                "b": {"type": np.ndarray, "dtype": np.uint8}
            })
            def inference(a, b):
                return

            x = np.random.randn(1, 1, 3, 2)
            img = np.random.uniform(0, 255, size=[1, 3, 5, 5]).astype(np.uint8)
            inference(x, img)

        except DataTypeException:
            self.fail()


    def test_dtype_fail(self):
        with self.assertRaises(DataTypeException):
            @tensorcheck({
                "a": {"type": np.ndarray, "dtype": np.int8}
            })
            def inference(a):
                return

            x = np.random.randn(1, 1, 3, 2)
            inference(x)


    def test_shape_success(self):
        try:
            @tensorcheck({
                "a": {"type": np.ndarray, "shape": [1, 1, 3, 2]},
            })
            def inference(a):
                return

            x = np.random.randn(1, 1, 3, 2)
            inference(x)

        except ShapeException:
            self.fail()


    def test_shape_fail(self):
        with self.assertRaises(ShapeException):
            @tensorcheck({
                "a": {"type": np.ndarray, "shape": [1, 1, 2, 2]},
            })
            def inference(a):
                return

            x = np.random.randn(1, 1, 3, 2)
            inference(x)


    def test_generic_shape_success(self):
        with self.assertRaises(ShapeException):
            @tensorcheck({
                "a": {"type": np.ndarray, "dtype": np.float64, "shape": [1, 1, 2, "W"], "range": [-5, 5] },
                "b": {"type": np.ndarray, "dtype": np.float64, "shape": [1, 1, 3, "W"], "range": [-5, 5] },
            })
            def inference(a, b):
                return

            x = np.random.randn(1, 1, 3, 2)
            y = np.random.randn(1, 1, 2, 2)
            inference(x, y)


    def test_generic_shape_fail(self):
        with self.assertRaises(ShapeException):
            @tensorcheck({
                "a": {"type": np.ndarray, "dtype": np.float64, "shape": [1, 1, "H", 2], "range": [-5, 5] },
                "b": {"type": np.ndarray, "dtype": np.float64, "shape": [1, 1, "H", 2], "range": [-5, 5] },
            })
            def inference(a, b):
                return

            x = np.random.randn(1, 1, 3, 2)
            y = np.random.randn(1, 1, 2, 2)
            inference(x, y)


    def test_rgb_image_and_alpha_mask(self):
        try:
            @tensorcheck({
                "img":  {"type": np.ndarray, "dtype": np.uint8, "shape": [1, 3, "H", "W"], "range": [0, 255] },
                "mask": {"type": np.ndarray, "dtype": np.float, "shape": [1, 1, "H", "W"], "range": [0, 1] },
            })
            def inference(img, mask):
                return

            img = np.random.uniform(0, 255, size=[1, 3, 5, 5]).astype(np.uint8)
            mask = np.random.uniform(size=[1, 1, 5, 5])
            inference(img, mask)

        except:
            self.fail()


if __name__ == '__main__':
    unittest.main(verbosity=2)
