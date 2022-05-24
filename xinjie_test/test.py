import imp
import os
import torch
import torch_xla
import torch_xla.core.xla_model as xm

os.environ['XRT_DEVICE_MAP']="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
os.environ['XRT_WORKERS']="localservice:0;grpc://localhost:40934"
# os.environ['GPU_NUM_DEVICES']="1"

print('------------------------------------------------------------------')
d=xm.xla_device(0)
print('------------------------------------------------------------------')
t1 = torch.randn(2, 2, device=d)
print('------------------------------------------------------------------')
t2 = torch.randn(2, 2, device=d)
print('------------------------------------------------------------------')
t3 = torch.nn.functional.relu(t1)
print('------------------------------------------------------------------')
t4 = torch.abs(t2)
print('------------------------------------------------------------------')
t = t3 + t4
print('------------------------------------------------------------------')
xm.mark_step()
print('------------------------------------------------------------------')
print(t)