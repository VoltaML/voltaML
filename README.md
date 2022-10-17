![lg3v3](https://user-images.githubusercontent.com/107309002/175033667-a58dab6e-ca8f-4739-a371-10a10b51e7e9.jpg)

<p align="center">
  <b> Accelerate your machine learning and deep learning models by upto 10X </b> 
</p>

<hr style="border:0.5px solid gray">

**voltaML** is an open-source lightweight library to accelerate your machine learning and deep learning models. VoltaML can optimize, compile and deploy your models to your target CPU and GPU devices, with just ***one line of code.***

#### Out of the box support for 


✅ FP16 Quantization<br/>

✅ Int8 Quantization*<br/>

✅ Hardware specific compilation


VoltaML has compilation support for the following:


<img width="1102" alt="Screenshot 2022-06-13 at 3 43 03 PM" src="https://user-images.githubusercontent.com/107309002/173331905-e7f506a8-f675-45ae-aff1-b84f65972f90.png">


## Installation

### Installation can be done via:

### Docker Container 🐳

`$ docker pull voltaml/voltaml:v0.3`

### Connect to our server 🔌

If you are unable to use docker, you can connect to our server, send your models via an API service as below:

`sign in`

<p align="center">
  <img width="640" height="440" src="https://user-images.githubusercontent.com/107309002/173332614-68abe0b3-e66e-4f5d-93fe-7c1362f67e31.png">
</p>

## Usage

Using **`VoltaCPUCompiler`**:

```python
import torch
from voltaml.compile import VoltaCPUCompiler
from voltaml.inference import cpu_performance

model = torch.load("path/to/model/dir")

# compile the model by giving paths
compiler = VoltaCPUCompiler(
        model=model,
        output_dir="destination/path/of/compiled/model",
        input_shape=(1, 3, 224, 224) # example input shape
    )

# returns the compiled model
compiled_model = compiler.compile()

# compute and compare performance
cpu_performance(compiled_model, model, compiler="voltaml", input_shape=(1, 3, 224, 224))
```

Using **`VoltaGPUCompiler`**:


```python
import torch
from voltaml.compile import VoltaGPUCompiler
from voltaml.inference import gpu_performance

model = torch.load("path/to/model/dir")

# compile the model by giving paths
compiler = VoltaGPUCompiler(
        model=model,
        output_dir="destination/path/of/compiled/model",
        input_shape=(1, 3, 224, 224), # example input shape
        precision="fp16" # specify precision, one of [fp32, fp16, int8]
    )

# returns the compiled model
compiled_model = compiler.compile()

# compute and compare performance
gpu_performance(compiled_model, model, input_shape=(1, 3, 224, 224))
```

Using **`TVMCompiler`**: 

```python
import torch
from voltaml.compile import TVMCompiler
from voltaml.inference import cpu_performance

model = torch.load("path/to/model/dir")

# compile the model by giving paths
compiler = TVMCompiler(
        model=model,
        output_dir="destination/path/of/compiled/model",
        input_shape=(1, 3, 224, 224), # example input shape
        target="llvm" # specify target device
    )

# returns the compiled model
compiled_model = compiler.compile()

# compute and compare performance
cpu_performance(compiled_model, model, compiler="tvm", input_shape=(1, 3, 224, 224))
```


## Benchmarks
### Classification Models Inference Latency (on GPU) ⏱️
Classification has been done on Imagenet data, `batch size = 1` on NVIDIA RTX 2080Ti. In terms of top 1% and 5% accuracy for `int8` models, we have not seen an accuracy drop of more than 1%. 
|     Model     | Pytorch Vanilla (ms) | VoltaGPU `FP16` (ms) | VoltaGPU `Int8` (ms) | Speed Gain |
|:-------------:|:------------:|:------------------:|:--------------------:|:------------:|
| `squeezenet1_1` |          2.5 |                0.2 |                0.2 |        13x |
| `resnet18`      |          2.7 |                0.4 |                0.3 |         9x |
| `resnet34`      |          4.5 |                0.7 |                0.5 |         9x |
| `resnet50`      |          4.8 |                0.7 |                0.5 |        13x |
| `resnet101`     |          8.8 |                1.4 |                1.0 |        14x |
| `densenet121`   |         15.7 |                2.4 |                2.0 |         8x |
| `densenet169`   |         22.0 |                4.4 |                3.8 |         6x |
| `densenet201`   |         26.8 |                6.3 |                5.0 |         5x |
| `vgg11`         |          2.0 |                0.9 |                0.5 |         4x |
| `vgg16`         |          3.5 |                1.2 |                0.7 |         5x |
| `vgg19`         |          4.0 |                1.3 |                0.8 |         5x |

### Enterpise Platform 🛣️
- [x] Hardware targeted optimised dockers for maximum performance.
- [ ] One-click deployment of the compiled models. 
- [ ] Cost-benefit analysis dashboard for optimal deployment.
- [ ] NVIDIA Triton optimzed dockers for large-scale GPU deployment.
- [ ] Quantization-Aware-Training (QAT) 
