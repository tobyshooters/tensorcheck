import unittest
from tensorcheck import *

import numpy as np

class TestTensorChecker(unittest.TestCase):

    def test_supports_args_and_kwargs(self):
        try:
            @tensorcheck({
                "a": {},
                "b": {},
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
                "b": {},
            })
            def inference(a):
                return

            x = np.random.randn(1, 1, 3, 2)
            y = np.random.randn(1, 1, 2, 2)
            inference(x)


    def test_dtype_success(self):
        try:
            @tensorcheck({
                "a": { "dtype": np.float, "dtype": np.float64 },
                "b": { "dtype": np.uint8    },
                "c": { "dtype": torch.float },
            })
            def inference(a, b, c):
                return

            x = np.random.randn(1, 1, 3, 2)
            img = np.random.uniform(0, 255, size=[1, 3, 5, 5]).astype(np.uint8)
            t = torch.randn(1, 1, 3, 3)

            inference(x, img, t)

        except DataTypeException:
            self.fail()


    def test_dtype_fail_numpy(self):
        with self.assertRaises(DataTypeException):
            @tensorcheck({
                "a": { "dtype": np.int8 }
            })
            def inference(a):
                return

            x = np.random.randn(1, 1, 3, 2)
            inference(x)

    def test_dtype_fail_torch(self):
        with self.assertRaises(DataTypeException):
            @tensorcheck({
                "a": { "dtype": torch.int }
            })
            def inference(a):
                return

            x = torch.randn(1, 1, 3, 2)
            inference(x)


    def test_shape_success(self):
        try:
            @tensorcheck({
                "a": { "shape": [1, 1, 3, 2] },
                "b": { "shape": [1, 1, 3, 2] },
            })
            def inference(a, b):
                return

            x = np.random.randn(1, 1, 3, 2)
            t = torch.randn(1, 1, 3, 2)
            inference(x, t)

        except ShapeException:
            self.fail()


    def test_shape_fail_numpy(self):
        with self.assertRaises(ShapeException):
            @tensorcheck({
                "a": { "shape": [1, 1, 2, 2] },
            })
            def inference(a):
                return

            x = np.random.randn(1, 1, 3, 2)
            inference(x)

    def test_shape_fail_torch(self):
        with self.assertRaises(ShapeException):
            @tensorcheck({
                "a": { "shape": [1, 1, 2, 2] },
            })
            def inference(a):
                return

            x = torch.randn(1, 1, 3, 2)
            inference(x)


    def test_generic_shape_success(self):
        try:
            @tensorcheck({
                "a": { "shape": [1, 1, 3, "W"] },
                "b": { "shape": [1, 1, 2, "W"] },
            })
            def inference(a, b):
                return

            x = np.random.randn(1, 1, 3, 2)
            y = np.random.randn(1, 1, 2, 2)
            inference(x, y)

        except:
            self.fail()


    def test_generic_shape_success_across_torch_and_numpy(self):
        try:
            @tensorcheck({
                "a": { "dtype": np.float,    "shape": [1, 1, 2, "W"] },
                "b": { "shape": torch.float, "shape": [1, 1, 3, "W"] },
            })
            def inference(a, b):
                return

            x = np.random.randn(1, 1, 2, 2)
            y = torch.randn(1, 1, 3, 2)
            inference(x, y)

        except:
            self.fail()


    def test_generic_shape_fail(self):
        with self.assertRaises(ShapeException):
            @tensorcheck({
                "a": { "shape": [1, 1, "H", 2] },
                "b": { "shape": [1, 1, "H", 2] },
            })
            def inference(a, b):
                return

            x = np.random.randn(1, 1, 3, 2)
            y = np.random.randn(1, 1, 2, 2)
            inference(x, y)


    def test_range_success(self):
        try:
            @tensorcheck({
                "a": { "dtype": np.float,    "range": [0, 1] },
                "b": { "dtype": torch.float, "range": [-10, 10] },
            })
            def inference(a, b):
                return

            x = np.random.uniform(size=[1, 1, 2, 2])
            y = torch.randn(1, 1, 3, 2)
            inference(x, y)

        except:
            self.fail()


    def test_upperbound_fail_numpy(self):
        with self.assertRaises(UpperBoundException):
            @tensorcheck({
                "a": { "dtype": np.float, "range": [0, 1] },
            })
            def inference(a):
                return

            x = 2 * np.ones([1, 1, 2, 2])
            inference(x)


    def test_lowerbound_fail_numpy(self):
        with self.assertRaises(LowerBoundException):
            @tensorcheck({
                "a": { "dtype": np.float, "range": [0, 1] },
            })
            def inference(a):
                return

            x = -1 * np.ones([1, 1, 2, 2])
            inference(x)


    def test_upperbound_fail_torch(self):
        with self.assertRaises(UpperBoundException):
            @tensorcheck({
                "a": { "dtype": torch.float, "range": [0, 1] },
            })
            def inference(a):
                return

            x = 2 * torch.ones(1, 1, 2, 2)
            inference(x)


    def test_lowerbound_fail_torch(self):
        with self.assertRaises(LowerBoundException):
            @tensorcheck({
                "a": { "dtype": torch.float, "range": [0, 1] },
            })
            def inference(a):
                return

            x = -1 * torch.ones(1, 1, 2, 2)
            inference(x)


    def test_rgb_image_and_alpha_mask(self):
        try:
            @tensorcheck({
                "img":  { "dtype": np.uint8, "shape": [1, 3, "H", "W"], "range": [0, 255] },
                "mask": { "dtype": np.float, "shape": [1, 1, "H", "W"], "range": [0, 1] },
            })
            def inference(img, mask):
                return

            img = np.random.uniform(0, 255, size=[1, 3, 5, 5]).astype(np.uint8)
            mask = np.random.uniform(size=[1, 1, 5, 5])
            inference(img, mask)

        except:
            self.fail()


    def test_return_with_outer_product_shape(self):
        try:
            @tensorcheck({
                "a":      { "shape": [2]    },
                "b":      { "shape": [3]    },
                "return": { "shape": [2, 3] },
            })
            def inference(a, b):
                return np.outer(a, b)

            inference(np.array([-3, 2]), np.array([2, 5, -1]))

        except:
            self.fail()


    def test_return_with_type_cast(self):
        try:
            @tensorcheck({
                "a":      { "dtype": np.float },
                "return": { "dtype": np.uint8 },
            })
            def cast_to_int(a):
                return a.astype(np.uint8)

            img = np.random.uniform(0, 255, size=[1, 3, 3])
            cast_to_int(img)

        except:
            self.fail()


    def test_return_fail_with_type(self):
        with self.assertRaises(DataTypeException):
            @tensorcheck({
                "a":      { "dtype": np.float },
                "return": { "dtype": np.uint8 },
            })
            def identity(a):
                return a

            img = np.random.uniform(0, 255, size=[1, 3, 3])
            identity(img)



if __name__ == '__main__':
    unittest.main(verbosity=2)
