# reproduce_pytorch_lightning_memory_issues

Reproducing issues I am having with my custom Dataset when using PyTorch Lightning multi GPU training with DDP.

When the dataset stores items as a numpy array, then the more num_workers are used in the data loader, the higher the memory usage.
In contrast, when using PyTorch tensors instead of numpy arrays, memory usage is much lower.

For example compare the run like this:

```python
python minimal.py --num_workers 10
# Uses around 5GB of RAM
python minimal.py --numpy --num_workers 10
# Same amount of data, but using numpy, now takes >30GB of RAM -> more than 6 times the amount
```
