# Inference Time Comparison (Torch vs TensorRT)

### 1. Convert Pytorch model to ONNX model (MobileNetV3_small).  

- Make mobilenet_v3_small.onnx
  ```commandline
  python torch2onnx.py
  ```
  
### 2. Convert ONNX model to TensorRT.

- Make mobilenet_v3_small.plan
  ```commandline
  python onnx2trt.py
  ```

### 3. Load TensorRT & Inference

- Inference TensorRT
  ```
  python inference_trt.py
  ```
- Inference Pytorch
  ```commandline
  python inference_torch.py
  ```

## Inference Time Comparison
Inference time per 1 iteration.  

- MobileNetV3_small

| Framework | Inference time (ms)             | Input size                          | GPU |
|---|---------------------------------|-------------------------------------|---|
| TensorRT | <div align="center"> 1 </div>   | <div align="center"> 3 x 224 x 224 </div>    | RTX3090 |
| Pytorch | <div align="center"> 9 </div>   | <div align="center"> 3 x 224 x 224 </div>    | RTX3090 |
| TensorRT | <div align="center"> 4 </div>   | <div align="center"> 3 x 1080 x 1920 </div>  | RTX3090 |
| Pytorch | <div align="center"> 9.5 </div> | <div align="center"> 3 x 1080 x 1920 </div>  | RTX3090 |

- MobileNetV3_large

| Framework | Inference time (ms)             | Input size                          | GPU |
|---|---------------------------------|-------------------------------------|---|
| TensorRT | <div align="center"> 0.5 </div> | <div align="center"> 3 x 224 x 224 </div>    | RTX3090 |
| Pytorch | <div align="center"> 10 </div>  | <div align="center"> 3 x 224 x 224 </div>    | RTX3090 |
| TensorRT | <div align="center"> 5 </div>   | <div align="center"> 3 x 1080 x 1920 </div>  | RTX3090 |
| Pytorch | <div align="center"> 25 </div>  | <div align="center"> 3 x 1080 x 1920 </div>  | RTX3090 |

- EfficientNet_b0

| Framework | Inference time (ms)             | Input size                         | GPU |
|---|---------------------------------|------------------------------------|---|
| TensorRT | <div align="center"> 0.8 </div> | <div align="center"> 3 x 224 x 224 </div>   | RTX3090 |
| Pytorch | <div align="center"> 15 </div>  | <div align="center"> 3 x 224 x 224 </div>   | RTX3090 |
| TensorRT | <div align="center"> 9 </div>   | <div align="center"> 3 x 1080 x 1920 </div> | RTX3090 |
| Pytorch | <div align="center"> 32 </div>  | <div align="center"> 3 x 1080 x 1920 </div> | RTX3090 |

- EfficientNet_b3

| Framework | Inference time (ms)   | Input size                         | GPU |
|---|-----------------------|------------------------------------|---|
| TensorRT | <div align="center"> 1 </div>  | <div align="center"> 3 x 224 x 224 </div>   | RTX3090 |
| Pytorch | <div align="center"> 26 </div> | <div align="center"> 3 x 224 x 224 </div>   | RTX3090 |
| TensorRT | <div align="center"> 16 </div> | <div align="center"> 3 x 1080 x 1920 </div> | RTX3090 |
| Pytorch | <div align="center"> 58 </div> | <div align="center"> 3 x 1080 x 1920 </div> | RTX3090 |

- EfficientNet_b7

| Framework | Inference time (ms)                 | Input size                         | GPU |
|---|-------------------------------------|------------------------------------|---|
| TensorRT | <div align="center"> 3 </div>                | <div align="center"> 3 x 224 x 224 </div>   | RTX3090 |
| Pytorch | <div align="center"> 52 </div>               | <div align="center"> 3 x 224 x 224 </div>   | RTX3090 |
| TensorRT | <div align="center"> 43 </div>               | <div align="center"> 3 x 1080 x 1920 </div> | RTX3090 |
| Pytorch | <div align="center"> out of memory... </div> | <div align="center"> 3 x 1080 x 1920 </div> | RTX3090 |

- ResNet50

| Framework | Inference time (ms)    | Input size                         | GPU |
|---|------------------------|------------------------------------|---|
| TensorRT | <div align="center"> 0.5 </div> | <div align="center"> 3 x 224 x 224 </div>   | RTX3090 |
| Pytorch | <div align="center"> 10 </div>  | <div align="center"> 3 x 224 x 224 </div>   | RTX3090 |
| TensorRT | <div align="center"> 7 </div>   | <div align="center"> 3 x 1080 x 1920 </div> | RTX3090 |
| Pytorch | <div align="center"> 46 </div>  | <div align="center"> 3 x 1080 x 1920 </div> | RTX3090 |

- ResNet101

| Framework | Inference time (ms)            | Input size                         | GPU |
|---|--------------------------------|------------------------------------|---|
| TensorRT | <div align="center"> 1 </div>  | <div align="center"> 3 x 224 x 224 </div>   | RTX3090 |
| Pytorch | <div align="center"> 19 </div> | <div align="center"> 3 x 224 x 224 </div>   | RTX3090 |
| TensorRT | <div align="center"> 11 </div> | <div align="center"> 3 x 1080 x 1920 </div> | RTX3090 |
| Pytorch | <div align="center"> 68 </div> | <div align="center"> 3 x 1080 x 1920 </div> | RTX3090 |

- ResNext50

| Framework | Inference time (ms)            | Input size                         | GPU |
|---|--------------------------------|------------------------------------|---|
| TensorRT | <div align="center"> 1 </div>  | <div align="center"> 3 x 224 x 224 </div>   | RTX3090 |
| Pytorch | <div align="center"> 9 </div>  | <div align="center"> 3 x 224 x 224 </div>   | RTX3090 |
| TensorRT | <div align="center"> 11 </div> | <div align="center"> 3 x 1080 x 1920 </div> | RTX3090 |
| Pytorch | <div align="center"> 59 </div> | <div align="center"> 3 x 1080 x 1920 </div> | RTX3090 |

- ResNext101

| Framework | Inference time (ms)             | Input size                         | GPU |
|---|---------------------------------|------------------------------------|---|
| TensorRT | <div align="center"> 2 </div>   | <div align="center"> 3 x 224 x 224 </div>   | RTX3090 |
| Pytorch | <div align="center"> 24 </div>  | <div align="center"> 3 x 224 x 224 </div>   | RTX3090 |
| TensorRT | <div align="center"> 23 </div>  | <div align="center"> 3 x 1080 x 1920 </div> | RTX3090 |
| Pytorch | <div align="center"> 156 </div> | <div align="center"> 3 x 1080 x 1920 </div> | RTX3090 |