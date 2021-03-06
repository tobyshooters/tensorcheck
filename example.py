import numpy as np
import torch
from tensorcheck import tensorcheck

@tensorcheck({
    "img":  {
        "dtype": np.uint8,
        "shape": [1, 3, "H", "W"],
        "range": [0, 255]
    },
    "mask": {
        "dtype": torch.float32,
        "shape": [1, 1, "H", "W"],
        "range": [0, 1]
    },
    "return": {
        "dtype": np.float32,
        "shape": [1, 3, "H", "W"],
        "range": [0, 255]
    },
})
def apply_mask(img, mask):
    # ...do compute
    return img * mask.numpy()


x = np.random.uniform(0, 255, size=[1, 3, 10, 8]).astype(np.uint8)
y = torch.rand(1, 1, 10, 7)
apply_mask(x, y)
# > tensorcheck.ShapeException: /mask/ dim 3 of torch.Size([1, 1, 10, 7]) is not W=8

x = np.random.uniform(0, 255, size=[1, 3, 10, 8]).astype(np.uint8)
y = 2 * torch.rand(1, 1, 10, 8)
apply_mask(x, y)
# > tensorcheck.UpperBoundException: /mask/ max value 1.9982... is greater than 1

x = np.random.uniform(0, 255, size=[1, 3, 10, 8]).astype(np.float)
y = torch.rand(1, 1, 10, 8)
apply_mask(x, y)
# > tensorcheck.DataTypeException: /img/ dtype float64 is not <class 'numpy.uint8'>

x = np.random.uniform(0, 255, size=[1, 3, 10, 8]).astype(np.uint8)
y = torch.rand(1, 1, 10, 8).int()
apply_mask(x, y)
# > tensorcheck.DataTypeException: /mask/ dtype torch.int32 is not torch.float32

x = np.random.uniform(0, 255, size=[1, 3, 10, 8]).astype(np.uint8)
y = torch.rand(1, 1, 10, 8)
apply_mask(x, y)
# > Success
