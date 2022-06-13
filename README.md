#                                                                 VoltaML 👋

**VoltaML** is a lightweight library to accelerate 🏎️ your machine learning and deep learning models. VoltaML can optimize, compile and deploy your models to your target CPU and GPU devices, with just **one line of code.**

VoltaML has compilation support for **Torchscript, ONNX, Apache TVM and NVIDIA TensorRT**

<img width="1102" alt="Screenshot 2022-06-13 at 3 43 03 PM" src="https://user-images.githubusercontent.com/107309002/173331905-e7f506a8-f675-45ae-aff1-b84f65972f90.png">

**VoltaML Compiler**

![Snap(3)](https://user-images.githubusercontent.com/107309002/173325658-6965c6f3-dd19-49b6-817b-ff4b416d842c.png)


# Benchmarks

## Classification Models Inference Latency
Classification has been done on Imagenet data, `batch size = 1` on NVIDIA RTX 2080Ti
|     Model     | Pytorch (ms) | VoltaGPU FP16 (ms) | VoltaGPU Int8 (ms) | Speed Gain |
|:-------------:|:------------:|:------------------:|:--------------------:|:------------:|
| `squeezenet1_1` |          2.5 |                0.2 |                0.2 |        13x |
| `resnet18`      |          2.7 |                0.4 |                0.3 |         9x |
| `resnet34`      |          4.5 |                0.7 |                0.5 |         9x |
| `resnet50`      |          6.6 |                0.7 |                0.5 |        13x |
| `resnet101`     |         13.6 |                1.3 |                1.0 |        14x |
| `densenet121`   |         15.7 |                2.4 |                2.0 |         8x |
| `densenet169`   |         22.0 |                4.4 |                3.8 |         6x |
| `densenet201`   |         26.8 |                6.3 |                5.0 |         5x |
| `vgg11`         |          2.0 |                0.9 |                0.5 |         4x |
| `vgg16`         |          3.5 |                1.2 |                0.7 |         5x |
| `vgg19`         |          4.0 |                1.3 |                0.8 |         5x |
