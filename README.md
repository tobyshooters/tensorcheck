# TensorCheck

Run-time validation of tensors for machine-learning systems.

Supports:
1. Tensor dtypes validation for `np.ndarray` and `torch.Tensor`
2. Shape validation, including generic shape variables
3. Range of input validation

## Example Usage

```python
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
})
def inference(img, mask):
    # ...do compute
    return


x = np.random.uniform(0, 255, size=[1, 3, 10, 8]).astype(np.uint8)
y = torch.rand(1, 1, 10, 7)
inference(x, y)
# > tensorcheck.ShapeException: /mask/ dim 3 of torch.Size([1, 1, 10, 7]) is not W=8

x = np.random.uniform(0, 255, size=[1, 3, 10, 8]).astype(np.uint8)
y = 2 * torch.rand(1, 1, 10, 8)
inference(x, y)
# > tensorcheck.UpperBoundException: /mask/ max value 1.9982... is greater than 1

x = np.random.uniform(0, 255, size=[1, 3, 10, 8]).astype(np.float)
y = torch.rand(1, 1, 10, 8)
inference(x, y)
# # > tensorcheck.DataTypeException: /img/ dtype float64 is not <class 'numpy.uint8'>

x = np.random.uniform(0, 255, size=[1, 3, 10, 8]).astype(np.uint8)
y = torch.rand(1, 1, 10, 8).int()
inference(x, y)
# > tensorcheck.DataTypeException: /mask/ dtype torch.int32 is not torch.float32

x = np.random.uniform(0, 255, size=[1, 3, 10, 8]).astype(np.uint8)
y = torch.rand(1, 1, 10, 8)
inference(x, y)
# > Success
```

## Future Work
- [ ] Return value checking
