import torch
import torchvision
import time

dummy_input = torch.randn(1, 3, 224, 224, device="cuda")
model = torchvision.models.mobilenet_v3_small(pretrained=True).cuda()


start_time = time.time()
for i in range(1000):
    outputs = model(dummy_input)
end_time = time.time()

print('Total Inference Time:', end_time - start_time)
print('Inference Time per 1 Iteration:', (end_time - start_time) / 1000)
