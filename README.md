# PyTorch multiprocessing spawn seems slow

When using multiprocessing spawn, I get much longer training times - especially when using a list of tensors in my dataset.

I found that when using a list of tensors, a file pointer is created for each entry of the list, so you may need to increase your ulimit: `ulimit -n 3333`.

Here are the speed results when running with 2 GPUs:

```bash
python custom.py --use_spawn                         # Training time: 17 seconds
python custom.py --use_spawn --use_lists             # Training time: 72 seconds (!)

# Instead of using spawn, start each process independently:
python custom.py --rank 0               # Training time: 14 seconds
python custom.py --rank 1

python custom.py --rank 0 --use_lists   # Training time: 14 seconds
python custom.py --rank 1 --use_lists
```

As you can see, spawn is slower, but especially much slower when using a list of tensors.

This minimal example only has ~100 lines of code including the model and dataset.

See related [PyTorch issue](https://github.com/pytorch/pytorch/issues/39443)
