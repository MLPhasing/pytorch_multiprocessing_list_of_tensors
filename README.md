# pytorch_lightning_multi_gpu_speed_analysis

Analyzing the speedup when going from 1 GPU to 4 GPUs

```bash
python minimal.py --gpus 1 # Takes around 213 seconds, data setup is 17 seconds, so 196 seconds training time
python minimal.py --gpus 4 # Takes around 87 seconds, data setup is 17 seconds, so 70 seconds training time
```

The speedup I see from 1 to 4 GPUs is only 2.8 times faster. I would expect at least 3x speedup, but even closer to 4x.
