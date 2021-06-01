# tensorrt-unet
This is a TensorRT version Unet, inspired by [tensorrtx](https://github.com/wang-xinyu/tensorrtx) and [pytorch-unet](https://github.com/milesial/Pytorch-UNet).<br>
You can generate TensorRT engine file using this script and customize some params and network structure based on network you trained (FP32/16 precision, input size, different conv, activation function...)<br>

# requirements

TensorRT 7.0 (you need to install tensorrt first)<br>
Cuda 10.2<br>
Python3.7<br>
opencv 4.4<br>
cmake 3.18<br>
# .pth2onnx file and convert .wts

## create env

```
pip install -r requirements.txt
```

## .pth file convert to .onnx model

train your dataset by following [pytorch-unet](https://github.com/milesial/Pytorch-UNet) and generate .pth file.<br>

Download pretrain model for:[https://github.com/milesial/Pytorch-UNet/releases/tag/v1.0](https://github.com/milesial/Pytorch-UNet/releases/tag/v1.0)

## convert .onnx

clone Pytorch-UNet repo and insert this python file in it,and modify some key info && run it

```python
import torch
from unet import UNet

torch_model = "models/unet_carvana_scale1_epoch5.pth"
onnx_modle = "models/unet_carvana_scale1_epoch5.onnx"

device = "cuda"
input_size = 572
channels = 3
classes = 1

net = UNet(n_channels=channels, n_classes=classes)
net.to(device=device)
net.load_state_dict(torch.load(torch_model, map_location=device))

image = torch.zeros((1, channels, input_size, input_size)).cuda()

torch.onnx.export(net, image, onnx_modle, opset_version=11, verbose=False, input_names=["input"], output_names=["ouput_segmentation_map"])

print("[INFO] Covert to onnx model success!")
```



## convert .wts

run gen_wts from utils folder, and move it to project folder<br>

# generate engine file and infer

## create build folder in project folder
```shell
mkdir build
```

## make file, generate exec file
```shell
cd build
cmake ..
make
```

## generate TensorRT engine file and infer image
```shell
unet -s
```
then a unet exec file will generated, you can use unet -d to infer files in a folder<br>
```shell
unet -d ../samples
```

# efficiency
the speed of tensorRT engine is much faster

 pytorch | TensorRT FP32 | TensorRT FP16
 ---- | ----- | ------  
 572x572 | 572x572 | 572x572 
 131ms | 110ms (batchsize 1) | 31ms (batchsize 1) 


# Further development

1. add INT8 calibrator<br>
2. add custom plugin<br>
etc
