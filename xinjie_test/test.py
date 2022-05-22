import imp
import os
import torch
import torch_xla
import torch_xla.core.xla_model as xm

os.environ['GPU_NUM_DEVICES']="1"

d=xm.xla_device()

t1 = torch.randn(2, 2, device=d)

t2 = torch.randn(2, 2, device=d)

t3 = torch.nn.functional.relu(t1)

t4 = torch.abs(t2)

t = t3 + t4

xm.mark_step()

print(t)