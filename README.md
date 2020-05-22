# Speed analysis using distributed data parallel on 4 GPUs (single machine)

When using PyTorch lightning for multi GPU training instead of a self-built solution, I am encountering severe speed problems.
I built this repo to replicate these issues.

What I found is that this mostly depends on the data structure which is used in the Dataset. 
We are caching larger amounts of data to memory and are using a list of tensors to do so as the tensors are all of different size (here in this example they have the same size for simplicity).
As you can see here, using a list of tensors does not work at all when using the PyTorch Lightning solution as the speed impairments are severe.
When using a self-built solution which just runs 4 independent processes and not using PyTorch multiprocessing, these impairments are not observed.

PyTorch Lightning is using PyTorch multiprocessing which uses shared memory. When using lists, it opens a file memory pointer for each tensor entry, so you probably need to increase your file limits: `ulimit -n 99999`.

Here are the speed results:

```bash
python minimal.py --gpus 4                            # Does not use a list of tensors, but one big tensor. Training time: 105 seconds.
python minimal.py --gpus 4 --use_list                 # Uses a list of tensors.                             Training time: 310 seconds, so 3x slower.

# For the custom implementation, you need to start the run 4 times, once for each GPU:
python custom.py --world_size 4 --rank 0              # Does not use a list of tensors, but one big tensor. Training time: 98  seconds.
python custom.py --world_size 4 --rank 1
python custom.py --world_size 4 --rank 2
python custom.py --world_size 4 --rank 3

python custom.py --world_size 4 --rank 0 --use_list   # Uses a list of tensors.                             Training time: 97  seconds.
python custom.py --world_size 4 --rank 1 --use_list
python custom.py --world_size 4 --rank 2 --use_list
python custom.py --world_size 4 --rank 3 --use_list
```

As you can see, the custom implementation has a comparable runtime for both scenarios, but the list of tensors approach is not feasible in the multiprocessing setup.
