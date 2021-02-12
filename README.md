# TensorCheck

Run-time validation of tensors for machine-learning systems.

```python
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

img = np.random.uniform(0, 255, size=[1, 3, 10, 5]).astype(np.uint8)
mask = np.random.uniform(size=[1, 1, 10, 5])
inference(img, mask)
```

Supports:
1. `np.ndarray` dtypes
2. Shape validation, including generic shape variables
3. Range of input variables

To Do:
- [ ] Support for `torch.Tensor`
- [ ] Return value checking
