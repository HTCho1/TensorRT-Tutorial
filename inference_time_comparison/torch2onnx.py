import torch
import torchvision


dummy_input = torch.randn(1, 3, 224, 224, device="cuda")
model = torchvision.models.mobilenet_v3_small(pretrained=True).cuda()

input_names = [ "actual_input_1" ] + [ "learned_%d" % i for i in range(16) ]
output_names = [ "output1" ]

torch.onnx.export(model, dummy_input, "mobilenet_v3_small.onnx", verbose=True,
                  input_names=input_names, output_names=output_names)
