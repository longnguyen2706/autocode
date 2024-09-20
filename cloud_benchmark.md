# Benchmark
## Setting
torch.set_float32_matmul_precision('high'), bf16

## Runpod
3090 - 24GB -> 0.22$/ hr
- batch_size = 16 -> GPU memory 84%
- token per sec = 61000 (on good machine) - 38000 (on bad machine)

A100 PCIe - 80GB (300W) -> 1.19$/ hr
- batch_size = 80 -> GPU memory 97%
- token per sec = 152000

4060 - 8GB (80W) 
- batch_size = 4 
- token per sec = 22000

A6000 - 48GB (good network)->  0.49S/ hr
- batch_size = 32 -> GPU memory  40.2GB/49GB 
- token per sec = 74500
