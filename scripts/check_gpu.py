from __future__ import annotations

import torch

print("torch_version:", torch.__version__)
print("torch_cuda_build:", torch.version.cuda)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())

if torch.cuda.is_available():
    print("device_name:", torch.cuda.get_device_name(0))
    x = torch.randn(2048, 2048, device="cuda")
    y = x @ x
    print("gpu_matmul_mean:", float(y.mean().item()))
    print("status: GPU_READY")
else:
    print("status: CUDA_NOT_AVAILABLE")
