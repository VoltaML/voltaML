import os
import time
import numpy as np
import torch
from torchvision import models
import tvm

from tqdm import tqdm
from tqdm import trange

from voltaml.preprocess import preprocess_image
from scipy.special import softmax
import tensorrt as trt

import pycuda.driver as cuda
import pycuda.autoinit
from voltaml.trt_infer import TensorRTInfer

def run_compiled_model(module, img_data):
    dtype = "float32"
    input_name = "data"
    module.set_input(input_name, img_data)
    module.run()
    output_shape = (img_data.shape[0], 1000)
    tvm_output = module.get_output(0, tvm.nd.empty(output_shape)).numpy()

    scores = softmax(tvm_output)
    scores = np.squeeze(scores)
    ranks = np.argsort(scores)[::-1]

    return tvm_output

def run_torch_model(model, img_data):
        
    with torch.no_grad():
        out = model(img_data)

    out = torch.softmax(out, 1)
    scores = torch.squeeze(out)
    ranks = torch.argsort(scores, descending=True)

    return out

def return_performance_data(model, input_shape, compiled=True):
    import timeit

    dummy_input = np.random.rand(input_shape)

    timing_number = 10
    timing_repeat = 10
    if compiled:
        time_taken = (np.array(timeit.Timer(lambda: run_compiled_model(model, dummy_input)).repeat(repeat=timing_repeat, number=timing_number)) * 1000 / timing_number)
    else:
        dummy_input_torch = torch.tensor(dummy_input).float()
        time_taken = (np.array(timeit.Timer(lambda: run_torch_model(model, dummy_input_torch)).repeat(repeat=timing_repeat, number=timing_number)) * 1000 / timing_number)

    time_taken = {
        "mean": np.mean(time_taken),
        "median": np.median(time_taken),
        "std": np.std(time_taken),
    }

    print(time_taken)

def process_image(img_data):
    image = torch.tensor(img_data).float()
    return image

def load_pytorch_model(model_name="resnet50"):
    model = getattr(models, model_name)(pretrained=True)
    return model

def cpu_performance(compiled_model, torch_model, compiler="voltaml", input_shape=(1, 3, 224, 224), throughput_batch_size=64):

    input_shape_for_throughput = list(input_shape)
    input_shape_for_throughput[0] = throughput_batch_size

    # to do: incorporate batch size 
    if compiler == "voltaml":
        voltaml_latency = measure_cpu_inference_latency(compiled_model, input_size=input_shape)
        torch_latency = measure_cpu_inference_latency(torch_model, input_size=input_shape)

        voltaml_throughput = measure_cpu_inference_throughput(compiled_model, input_size=input_shape_for_throughput)
        torch_throughput = measure_cpu_inference_throughput(torch_model, input_size=input_shape_for_throughput)

        print("Latency:")
        print("-"*50)
        print("VoltaML Inference Latency: {:.2f} ms / sample".format(voltaml_latency * 1000))
        print("PyTorch Inference Latency: {:.2f} ms / sample".format(torch_latency * 1000))
        print("\n")
        print("Throughput:")
        print("-"*50)
        print("VoltaML Inference Throughput: {:.2f} samples / s".format(voltaml_throughput))
        print("PyTorch Inference Throughput: {:.2f} samples / s".format(torch_throughput))

    elif compiler == "tvm":

        tvm_latency = measure_tvm_inference_latency(compiled_model, input_size=input_shape)
        torch_latency = measure_cpu_inference_latency(torch_model, input_size=input_shape)

        tvm_throughput = measure_tvm_inference_throughput(compiled_model, input_size=input_shape_for_throughput)
        torch_throughput = measure_cpu_inference_throughput(torch_model, input_size=input_shape_for_throughput)

        print("Latency:")
        print("-"*50)
        print("TVM Inference Latency: {:.2f} ms / sample".format(tvm_latency * 1000))
        print("PyTorch Inference Latency: {:.2f} ms / sample".format(torch_latency * 1000))
        print("")
        print("Throughput:")
        print("-"*50)
        print("TVM Inference Throughput: {:.2f} samples / s".format(tvm_throughput))
        print("PyTorch Inference Throughput: {:.2f} samples / s".format(torch_throughput))

    else:
        raise NotImplementedError(f"{compiler} compiler not implemented")

def gpu_performance(compiled_model, model, input_shape=(1, 3, 224, 224), throughput_batch_size=64):

    input_shape_for_throughput = list(input_shape)
    input_shape_for_throughput[0] = throughput_batch_size

    gpu_inference_model = TensorRTInfer(compiled_model)

    torch_latency = measure_gpu_inference_latency(model, input_size=input_shape, model_type="torch")
    voltaml_gpu_latency = measure_gpu_inference_latency(gpu_inference_model, input_size=input_shape, model_type="voltaml")

    torch_throughput = measure_gpu_inference_throughput(model, input_size=input_shape_for_throughput, model_type="torch")
    voltaml_gpu_throughput = measure_gpu_inference_throughput(gpu_inference_model, input_size=input_shape_for_throughput, model_type="voltaml")
    
    print("Latency:")
    print("-"*50)
    print("VoltaML GPU Inference Latency: {:.2f} ms / sample".format(voltaml_gpu_latency * 1000))
    print("PyTorch Inference Latency: {:.2f} ms / sample".format(torch_latency * 1000))
    print("\n")
    print("Throughput:")
    print("-"*50)
    print("VoltaML GPU Inference Throughput: {:.2f} samples / s".format(voltaml_gpu_throughput))
    print("PyTorch Inference Throughput: {:.2f} samples / s".format(torch_throughput))


class GPUInference:

    def __init__(self, model, num_classes, target_dtype=np.float32):
        self.model = model
        self.target_dtype = target_dtype
        self.num_classes = num_classes  
        self.stream = None
        
    def allocate_memory(self, batch):
        self.output = np.empty(self.num_classes, dtype=self.target_dtype) # Need to set both input and output precisions to FP16 to fully enable FP16

        # Allocate device memory
        self.d_input = cuda.mem_alloc(1 * batch.nbytes)
        self.d_output = cuda.mem_alloc(1 * self.output.nbytes)
        self.bindings = [int(self.d_input), int(self.d_output)]
        self.stream = cuda.Stream()
        
    def predict(self, batch): # result gets copied into output
        if self.stream is None:
            self.allocate_memory(batch)
            
        # Transfer input data to device
        cuda.memcpy_htod_async(self.d_input, batch, self.stream)
        # Execute model557
        self.model.execute_async_v2(self.bindings, self.stream.handle, None)
        # Transfer predictions back
        cuda.memcpy_dtoh_async(self.output, self.d_output, self.stream)
        # Syncronize threads
        self.stream.synchronize()

#################################################
#                  Latency                      #
#################################################

def measure_cpu_inference_latency(model,
                              input_size=(1,3,224,224),
                              num_samples=100,
                              num_warmups=10):

    model.eval()

    x = torch.rand(size=input_size)

    # with torch.no_grad():
    for _ in range(num_warmups):
        _ = model(x)
    # torch.cuda.synchronize()

    # with torch.no_grad():
    start_time = time.time()
    for _ in tqdm(range(num_samples), desc="calculating latency: "):
        _ = model(x)
        # torch.cuda.synchronize()
    end_time = time.time()
    elapsed_time = end_time - start_time  
    
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave

# def measure_gpu_inference_latency(model,
#                               input_size=(1,3,224,224),
#                               num_samples=100,
#                               num_warmups=10):


#     x = np.random.rand(*input_size)

#     for _ in range(num_warmups):
#         _ = model.predict(x)

#     start_time = time.time()
#     for _ in tqdm(range(num_samples), desc="calculating latency..."):
#         _ = model.predict(x)
#     end_time = time.time()
#     elapsed_time = end_time - start_time  
#     elapsed_time_ave = elapsed_time / num_samples

#     return elapsed_time_ave

def measure_gpu_inference_latency(model,
                              input_size=(1,3,224,224), num_samples=1000, num_warmups=100, model_type="torch"):

    if model_type == "torch":
        model = model.to("cuda:0")
        model.eval()

        x = torch.rand(size=input_size).to("cuda:0")

        with torch.no_grad():
            for _ in range(num_warmups):
                _ = model(x)
        torch.cuda.synchronize()

        with torch.no_grad():
            start_time = time.time()
            for _ in tqdm(range(num_samples), desc="calculating latency..."):
                _ = model(x)
                torch.cuda.synchronize()
            end_time = time.time()
        elapsed_time = end_time - start_time  

        elapsed_time_ave = elapsed_time / num_samples
        model = model.to("cpu:0")
        return elapsed_time_ave

    elif model_type == "voltaml":

        for i in range(2): 
            # spec = model.input_spec()
            batch = input_size
            times = []

            with torch.no_grad():
                start_time = time.time()
                for i in range(num_samples):
                    
                    model.infer(batch)
                    torch.cuda.synchronize()
                end_time = time.time()
#                     times.append(end_time - start)
            elapsed_time = end_time - start_time  
#             elapsed_time_ave = 1000*np.average(times)
            elapsed_time_ave = elapsed_time / num_samples
        return elapsed_time_ave
    
    else:
        raise ValueError("model_type should be one of torch and voltaml")

def measure_tvm_inference_latency(model,
                              input_size=(1,3,224,224),
                              num_samples=100,
                              num_warmups=10):

    x = np.random.rand(*input_size)

    for _ in range(num_warmups):
        _ = run_compiled_model(model, x)

    start_time = time.time()
    for _ in tqdm(range(num_samples), desc="calculating latency: "):
        _ = run_compiled_model(model, x)
    end_time = time.time()
    elapsed_time = end_time - start_time  
    
    elapsed_time_ave = elapsed_time / num_samples

    return elapsed_time_ave


#################################################
#                  Throughput                   #
#################################################


# note that input_size should contain optimal batch size for the gpu
def measure_cpu_inference_throughput(model, input_size=(64, 3, 224, 224), repetitions=100):
    model.eval()
    
    dummy_input = torch.randn(*input_size, dtype=torch.float)

    t = trange(repetitions, desc="calculating throughput: ") 

    total_time = 0
    with torch.no_grad():
        # for rep in tqdm(range(repetitions), desc="calculating throughput..."):
        for rep in t:
            # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            # starter.record()
            start_time = time.time()
            _ = model(dummy_input)
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_time = elapsed_time #/ 1000
            total_time += elapsed_time
            # t.set_description(f"calculating throughput elapsed time: {curr_time}")
    throughput = (repetitions*input_size[0])/total_time
    return throughput

# def measure_gpu_inference_throughput(model, input_size=(64, 3, 224, 224), repetitions=100):
    
#     dummy_input = np.random.rand(*input_size)

#     t = trange(repetitions, desc="calculating throughput: ") 

#     total_time = 0
#         # for rep in tqdm(range(repetitions), desc="calculating throughput..."):
#     for rep in t:
#         # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
#         # starter.record()
#         start_time = time.time()
#         _ = model.predict(dummy_input)
#         end_time = time.time()
#         elapsed_time = end_time - start_time
#         elapsed_time = elapsed_time #/ 1000
#         total_time += elapsed_time
#             # t.set_description(f"calculating throughput elapsed time: {curr_time}")
#     throughput = (repetitions*input_size[0])/total_time
#     return throughput

def measure_gpu_inference_throughput(model, input_size=(1, 3, 224, 224), repetitions=100, model_type="torch"):
    if model_type == "torch":
        model = model.to("cuda:0")
        model.eval()
        dummy_input = torch.rand(*input_size).to("cuda:0")

        t = trange(repetitions, desc="calculating throughput: ") 

        times = []
#         total_time = 0
            # for rep in tqdm(range(repetitions), desc="calculating throughput..."):
        with torch.no_grad():
            for rep in t:
                # starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                # starter.record()
                start_time = time.time()
                _ = model(dummy_input)
                torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
                
#                 elapsed_time = end_time - start_time
#                 elapsed_time = elapsed_time #/ 1000
#                 total_time += elapsed_time
                # t.set_description(f"calculating throughput elapsed time: {curr_time}")
        throughput = input_size[0] / np.average(times)
#         throughput = (repetitions*input_size[0])/total_time
        return throughput
    elif model_type == "voltaml":

        dummy_input = np.random.rand(*input_size)

#         spec = model.input_spec()
        batch = input_size
        times = []
#         total_time = 0
        for i in range(repetitions):  # GPU warmup iterations
            model.infer(batch)
        with torch.no_grad():
            for i in range(repetitions):
                start_time = time.time()
                model.infer(batch)
                torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
#                 elapsed_time = end_time - start_time
#                 total_time += elapsed_time

        throughput = model.batch_size / np.average(times)
#         throughput = (repetitions*input_size[0])/total_time
        return throughput
    else:
        raise ValueError("model_type should be one of torch and voltaml")

def measure_tvm_inference_throughput(model, input_size=(64, 3, 224, 224), repetitions=100):
    
    x = np.random.rand(*input_size)
    total_time = 0

    t = trange(repetitions, desc="calculating throughput: ") 

    # for rep in tqdm(range(repetitions), desc="Calculating throughput..."):
    for rep in t:
        start_time = time.time()
        _ = run_compiled_model(model, x)
        end_time = time.time()
        elapsed_time = end_time - start_time
        elapsed_time = elapsed_time #/ 1000
        total_time += elapsed_time
        # t.set_description(f"calculating throughput elapsed time: {elapsed_time}")
    throughput = (repetitions*input_size[0])/total_time
    return throughput

def print_size_of_model(model):
    torch.save(model.state_dict(), "temp.p")
    print('Size (MB):', os.path.getsize("temp.p")/1e6)
    os.remove('temp.p')