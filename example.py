import numpy as np
from tensorcheck import tensorcheck

@tensorcheck({
    "img":  {
        "type": np.ndarray,
        "dtype": np.uint8,
        "shape": [1, 3, "H", "W"],
        "range": [0, 255]
    },
    "mask": {
        "type": np.ndarray,
        "dtype": np.float,
        "shape": [1, 1, "H", "W"],
        "range": [0, 1]
    },
})
def inference(img, mask):
    # ...do compute
    return


x = np.random.uniform(0, 255, size=[1, 3, 10, 8]).astype(np.uint8)
y = np.random.uniform(0,   1, size=[1, 1, 10, 7])
inference(x, y)
# > tensorcheck.ShapeException: /mask/ shape (1, 1, 10, 7) at dim 3 != W=8

x = np.random.uniform(0, 255, size=[1, 3, 10, 8]).astype(np.uint8)
y = np.random.uniform(0,   2, size=[1, 1, 10, 8])
inference(x, y)
# > tensorcheck.UpperBoundException: /mask/ max value 1.9893163253748984 is greater than 1

x = np.random.uniform(0, 255, size=[1, 3, 10, 8])
y = np.random.uniform(0,   1, size=[1, 1, 10, 8])
inference(x, y)
# > tensorcheck.DataTypeException: /img/ dtype float64 is not <class 'numpy.uint8'>

x = np.random.uniform(0, 255, size=[1, 3, 10, 8]).astype(np.uint8)
y = np.random.uniform(0,   1, size=[1, 1, 10, 8])
inference(x, y)
# > Success
