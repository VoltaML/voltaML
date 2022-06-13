#                                                                 VoltaML 👋

**VoltaML** is an open-source lightweight library to accelerate 🏎️ your machine learning and deep learning models. VoltaML can optimize, compile and deploy your models to your target CPU and GPU devices, with just ***one line of code.***

VoltaML has compilation support for the following:


<img width="1102" alt="Screenshot 2022-06-13 at 3 43 03 PM" src="https://user-images.githubusercontent.com/107309002/173331905-e7f506a8-f675-45ae-aff1-b84f65972f90.png">


<p align="center">
  <img width="640" height="440" src="https://user-images.githubusercontent.com/107309002/173332614-68abe0b3-e66e-4f5d-93fe-7c1362f67e31.png">
</p>

## Benchmarks
### Classification Models Inference Latency
Classification has been done on Imagenet data, `batch size = 1` on NVIDIA RTX 2080Ti. In terms of top 1% and 5% accuracy for `int8` models, we have not seen an accuracy drop of more than 1%. 
|     Model     | Pytorch (ms) | VoltaGPU `FP16` (ms) | VoltaGPU `Int8` (ms) | Speed Gain |
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
