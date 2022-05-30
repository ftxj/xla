import imp
import os
import torch
import torch_xla
import torch_xla.core.xla_model as xm

os.environ['XRT_DEVICE_MAP']="CPU:0;/job:localservice/replica:0/task:0/device:XLA_CPU:0"
os.environ['XRT_WORKERS']="localservice:0;grpc://localhost:40934"
# os.environ['GPU_NUM_DEVICES']="1"

print('1------------------------------------------------------------------')
d=xm.xla_device(0)
print('2------------------------------------------------------------------')
t1 = torch.randn(2, 2, device=d)
print('3------------------------------------------------------------------')
t2 = torch.randn(2, 2, device=d)
print('4------------------------------------------------------------------')
t3 = torch.nn.functional.relu(t1)
print('5------------------------------------------------------------------')
t4 = torch.abs(t2)
print('6------------------------------------------------------------------')
t = t3 + t4
print('7------------------------------------------------------------------')
xm.mark_step()
print('8------------------------------------------------------------------')