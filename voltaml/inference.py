import os
import time
import numpy as np
import torch
from torchvision import models
import tvm
from tqdm import tqdm
from tqdm import trange
from scipy.special import softmax
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
from voltaml.trt_infer import TensorRTInfer
from voltaml.models.common import DetectMultiBackend
from voltaml.utils.torch_utils import select_device
from voltaml.utils.detect import run
from typing import Callable, Dict, List, Tuple, Type, Union
from pathlib import Path
import argparse
import gc
import logging
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoModelForQuestionAnswering,
    AutoModelForSequenceClassification,
    AutoModelForTokenClassification,
    AutoTokenizer,
    PretrainedConfig,
    PreTrainedModel,
    PreTrainedTokenizer,
)

from voltaml.transformer.backends.ort_utils import (
    cpu_quantization,
    create_model_for_provider,
    inference_onnx_binding,
    optimize_onnx,
)
from voltaml.transformer.backends.pytorch_utils import (
    convert_to_onnx,
    get_model_size,
    infer_classification_pytorch,
    infer_feature_extraction_pytorch,
)
from voltaml.transformer.backends.st_utils import STransformerWrapper, load_sentence_transformers
from voltaml.transformer.benchmarks.utils import (
    compare_outputs,
    generate_multiple_inputs,
    print_timings,
    setup_logging,
    to_numpy,
    track_infer_time,
)
from voltaml.transformer.triton.configuration import Configuration, EngineType
from voltaml.transformer.triton.configuration_decoder import ConfigurationDec
from voltaml.transformer.triton.configuration_encoder import ConfigurationEnc
from voltaml.transformer.triton.configuration_question_answering import ConfigurationQuestionAnswering
from voltaml.transformer.triton.configuration_token_classifier import ConfigurationTokenClassifier
from voltaml.transformer.utils.args import parse_args


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
    
def time_sync():
    # PyTorch-accurate time
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    return time.time()

def process_image(img_data):
    image = torch.tensor(img_data).float()
    return image

def load_pytorch_model(model_name="resnet50"):
    model = getattr(models, model_name)(pretrained=True)
    return model

def cpu_performance(compiled_model, torch_model, compiler="voltaml", input_shape=(1, 3, 224, 224), throughput_batch_size=1):

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

def gpu_performance(compiled_model, model, input_shape=(1, 3, 224, 224), throughput_batch_size=64, is_yolo=False):

    input_shape_for_throughput = list(input_shape)
    input_shape_for_throughput[0] = throughput_batch_size
    
    if not is_yolo:
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

    else:
        t = run(weights=compiled_model, batch_size=input_shape[0], imgsz=(input_shape[2],input_shape[3]))
        torch_latency = measure_gpu_inference_latency(model, input_size=input_shape, model_type="torch")
        voltaml_gpu_latency = t[1]
        torch_fps = 1 / (torch_latency)
        voltaml_gpu_fps = 1 / (voltaml_gpu_latency / 1000)
        
        print("Latency:")
        print("-"*50)
        print("VoltaML GPU Inference Latency: {:.2f} ms / sample".format(voltaml_gpu_latency))
        print("PyTorch Inference Latency: {:.2f} ms / sample".format(torch_latency * 1000))
        
        print("\n")
        print("FPS:")
        print("-"*50)
        print("VoltaML GPU Inference Throughput: {:.2f} fps".format(voltaml_gpu_fps))
        print("PyTorch Inference Throughput: {:.2f} fps".format(torch_fps))
        
        
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


def measure_gpu_inference_latency(model,
                              input_size=(1,3,224,224), num_samples=1000, num_warmups=100, model_type="torch", is_yolo=False):

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
        if is_yolo:
            for i in range(2): 
                # spec = model.input_spec()
                batch = torch.rand(*input_size)
                # x = torch.rand(size=input_size)

                times = 0

                with torch.no_grad():
                    
                    for i in range(num_samples):
                        start_time = time_sync()
                        model(batch)
                        end_time = time_sync()
                        elapsed_time = end_time - start_time  
                        times += elapsed_time
                elapsed_time_ave = elapsed_time / num_samples * 1E3
            return elapsed_time_ave    
        else:   

            for i in range(2): 
                batch = torch.rand(*input_size)
                times = []
                with torch.no_grad():
                    start_time = time.time()
                    for i in range(num_samples):
                        model.infer(batch)
                        torch.cuda.synchronize()
                    end_time = time.time()
                elapsed_time = end_time - start_time  
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
        for rep in t:
            start_time = time.time()
            _ = model(dummy_input)
            end_time = time.time()
            elapsed_time = end_time - start_time
            elapsed_time = elapsed_time #/ 1000
            total_time += elapsed_time
    throughput = (repetitions*input_size[0])/total_time
    return throughput

def measure_gpu_inference_throughput(model, input_size=(1, 3, 224, 224), repetitions=100, model_type="torch"):
    if model_type == "torch":
        model = model.to("cuda:0")
        model.eval()
        dummy_input = torch.rand(*input_size).to("cuda:0")

        t = trange(repetitions, desc="calculating throughput: ") 

        times = []
        with torch.no_grad():
            for rep in t:
                start_time = time.time()
                _ = model(dummy_input)
                torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
                
        throughput = input_size[0] / np.average(times)
        return throughput
    elif model_type == "voltaml":

        dummy_input = np.random.rand(*input_size)
        batch = input_size
        times = []
        for i in range(repetitions):  # GPU warmup iterations
            model.infer(batch)
        with torch.no_grad():
            for i in range(repetitions):
                start_time = time.time()
                model.infer(batch)
                torch.cuda.synchronize()
                end_time = time.time()
                times.append(end_time - start_time)
        throughput = model.batch_size / np.average(times)
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



def check_accuracy(
    engine_name: str,
    pytorch_output: List[torch.Tensor],
    engine_output: List[Union[np.ndarray, torch.Tensor]],
    tolerance: float,
) -> None:
    """
    Compare engine predictions with a reference.
    Assert that the difference is under a threshold.

    :param engine_name: string used in error message, if any
    :param pytorch_output: reference output used for the comparaison
    :param engine_output: output from the engine
    :param tolerance: if difference in outputs is above threshold, an error will be raised
    """
    pytorch_output = to_numpy(pytorch_output)
    engine_output = to_numpy(engine_output)
    discrepency = compare_outputs(pytorch_output=pytorch_output, engine_output=engine_output)
    assert discrepency <= tolerance, (
        f"{engine_name} discrepency is too high ({discrepency:.2f} >= {tolerance}):\n"
        f"Pythorch:\n{pytorch_output}\n"
        f"VS\n"
        f"Engine:\n{engine_output}\n"
        f"Diff:\n"
        f"{torch.asarray(pytorch_output) - torch.asarray(engine_output)}\n"
        "Tolerance can be increased with --atol parameter."
    )


def launch_inference(
    infer: Callable[[Dict[str, torch.Tensor]], torch.Tensor],
    inputs: List[Dict[str, Union[np.ndarray, torch.Tensor]]],
    nb_measures: int,
) -> Tuple[List[Union[np.ndarray, torch.Tensor]], List[float]]:
    """
    Perform inference and measure latency.

    :param infer: a lambda which will perform the inference
    :param inputs: tensor compatible with the lambda (Torch tensor for Pytorch, or numpy otherwise)
    :param nb_measures: number of measures to perform for the latency measure
    :return: a tuple of model output and inference latencies
    """
    assert type(inputs) == list
    assert len(inputs) > 0
    outputs = list()
    for batch_input in inputs:
        output = infer(batch_input)
        outputs.append(output)
    time_buffer: List[int] = list()
    for _ in range(nb_measures):
        with track_infer_time(time_buffer):
            _ = infer(inputs[0])
    return outputs, time_buffer

def get_triton_output_shape(output: torch.Tensor, task: str) -> List[int]:
    triton_output_shape = list(output.shape)
    triton_output_shape[0] = -1  # dynamic batch size
    if task in ["text-generation", "token-classification", "question-answering"]:
        triton_output_shape[1] = -1  # dynamic sequence size
    return triton_output_shape

def nlp_performance(verbose=False,
                    device=None, #choices=["cpu", "cuda"]
                    backend=["onnx"], #choices=["onnx","tensorrt"]
                    seq_len=[16, 16, 16], # "sequence lengths to optimize for (min, optimal, max). Used by TensorRT and benchmarks."
                    seed=int(123),
                    nb_threads=1,
                    auth_token=str(''),
                    output='models',
                    tokenizer=str(''),
                    model=str(''),
                    task="classification",# ["classification", "embedding", "text-generation", "token-classification", "question-answering"]
                    batch_size=[1,1,1], #"batch sizes to optimize for (min, optimal, max). Used by TensorRT and benchmarks."
                    warmup=int(10), # "# of inferences to warm each model"
                    nb_measures=int(1000),
                    nb_instances=int(1),
                    atol=float(3e-1), #"tolerance when comparing outputs to Pytorch ones"
                    quantization=False,
                    name='transformer',
                    fast=False, # skip the Pytorch (FP16) benchmark"
                    workspace_size=int(10000)
                   ):
    
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if device == "cpu" and "tensorrt" in backend:
        raise Exception("can't perform inference on CPU and use Nvidia TensorRT as backend")

    if len(seq_len) == len(set(seq_len)) and "tensorrt" in backend:
        logging.warning("having different sequence lengths may make TensorRT slower")

    torch.manual_seed(seed)
    np.random.seed(seed)
    torch.set_num_threads(nb_threads)

    if isinstance(auth_token, str) and auth_token.lower() in ["true", "t"]:
        auth_token = True
    elif isinstance(auth_token, str):
        auth_token = auth_token
    else:
        auth_token = None
    run_on_cuda: bool = device.startswith("cuda")
    Path(output).mkdir(parents=True, exist_ok=True)
    onnx_model_path = os.path.join(output, "model-original.onnx")
    onnx_optim_model_path = os.path.join(output, "model.onnx")
    tensorrt_path = os.path.join(output, "model.plan")
    if run_on_cuda:
        assert torch.cuda.is_available(), "CUDA/GPU is not available on Pytorch. Please check your CUDA installation"
    tokenizer_path = tokenizer if tokenizer else model
    tokenizer: PreTrainedTokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_auth_token=auth_token)
    model_config: PretrainedConfig = AutoConfig.from_pretrained(
        pretrained_model_name_or_path=model, use_auth_token=auth_token
    )
    input_names: List[str] = tokenizer.model_input_names
    if task == "embedding":
        model_pytorch: Union[PreTrainedModel, STransformerWrapper] = load_sentence_transformers(
            model, use_auth_token=auth_token
        )
    elif task == "classification":
        model_pytorch = AutoModelForSequenceClassification.from_pretrained(model, use_auth_token=auth_token)
    elif task == "token-classification":
        model_pytorch = AutoModelForTokenClassification.from_pretrained(model, use_auth_token=auth_token)
    elif task == "question-answering":
        model_pytorch = AutoModelForQuestionAnswering.from_pretrained(model, use_auth_token=auth_token)
    elif task == "text-generation":
        model_pytorch = AutoModelForCausalLM.from_pretrained(model, use_auth_token=auth_token)
        input_names = ["input_ids"]
    else:
        raise Exception(f"unknown task: {task}")

    logging.info(f"axis: {input_names}")

    model_pytorch.eval()
    if run_on_cuda:
        model_pytorch.cuda()

    tensor_shapes = list(zip(batch_size, seq_len))
    # take optimial size
    inputs_pytorch = generate_multiple_inputs(
        batch_size=tensor_shapes[1][0],
        seq_len=tensor_shapes[1][1],
        input_names=input_names,
        device=device,
        nb_inputs_to_gen=warmup,
    )
    
    timings = {}

    def get_pytorch_infer(model: PreTrainedModel, cuda: bool, task: str):
        if task in ["classification", "text-generation", "token-classification", "question-answering"]:
            return infer_classification_pytorch(model=model, run_on_cuda=cuda)
        if task == "embedding":
            return infer_feature_extraction_pytorch(model=model, run_on_cuda=cuda)
        raise Exception(f"unknown task: {task}")

    with torch.inference_mode():
        logging.info("running Pytorch (FP32) benchmark")
        pytorch_output, time_buffer = launch_inference(
            infer=get_pytorch_infer(model=model_pytorch, cuda=run_on_cuda, task=task),
            inputs=inputs_pytorch,
            nb_measures=nb_measures,
        )
        if task == "text-generation":
            conf_class: Type[Configuration] = ConfigurationDec
        elif task == "token-classification":
            conf_class: Type[Configuration] = ConfigurationTokenClassifier
        elif task == "question-answering":
            conf_class: Type[Configuration] = ConfigurationQuestionAnswering
        else:
            conf_class = ConfigurationEnc

        triton_conf = conf_class(
            model_name_base=name,
            dim_output=get_triton_output_shape(
                output=pytorch_output[0] if type(pytorch_output[0]) == torch.Tensor else pytorch_output[0][0],
                task=task,
            ),
            nb_instance=nb_instances,
            tensor_input_names=input_names,
            working_directory=output,
            device=device,
        )
        timings["Pytorch (FP32)"] = time_buffer
        if run_on_cuda and not fast:
            from torch.cuda.amp import autocast

            with autocast():
                engine_name = "Pytorch (FP16)"
                logging.info("running Pytorch (FP16) benchmark")
                pytorch_fp16_output, time_buffer = launch_inference(
                    infer=get_pytorch_infer(model=model_pytorch, cuda=run_on_cuda, task=task),
                    inputs=inputs_pytorch,
                    nb_measures=nb_measures,
                )
                check_accuracy(
                    engine_name=engine_name,
                    pytorch_output=pytorch_output,
                    engine_output=pytorch_fp16_output,
                    tolerance=atol,
                )
                timings[engine_name] = time_buffer
        elif device == "cpu":
            logging.info("preparing Pytorch (INT-8) benchmark")
            model_pytorch = torch.quantization.quantize_dynamic(model_pytorch, {torch.nn.Linear}, dtype=torch.qint8)
            engine_name = "Pytorch (INT-8)"
            logging.info("running Pytorch (FP32) benchmark")
            pytorch_int8_output, time_buffer = launch_inference(
                infer=get_pytorch_infer(model=model_pytorch, cuda=run_on_cuda, task=task),
                inputs=inputs_pytorch,
                nb_measures=nb_measures,
            )
            check_accuracy(
                engine_name=engine_name,
                pytorch_output=pytorch_output,
                engine_output=pytorch_int8_output,
                tolerance=atol,
            )
            timings[engine_name] = time_buffer
    model_pytorch.cpu()

    logging.info("cleaning up")
    if run_on_cuda:
        torch.cuda.empty_cache()
    gc.collect()
    
    if "tensorrt" in backend:
        logging.info("preparing TensorRT (FP16) benchmark")
        try:
            import tensorrt as trt
            from tensorrt.tensorrt import ICudaEngine, Logger, Runtime

            from voltaml.transformer.backends.trt_utils import build_engine, load_engine, save_engine
        except ImportError:
            raise ImportError(
                "It seems that TensorRT is not yet installed. "
                "It is required when you declare TensorRT backend."
                "Please find installation instruction on "
                "https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html"
            )
        trt_logger: Logger = trt.Logger(trt.Logger.VERBOSE if verbose else trt.Logger.WARNING)
        runtime: Runtime = trt.Runtime(trt_logger)
        # important to check the engine has been correctly serialized
        tensorrt_model: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = load_engine(
            runtime=runtime, engine_file_path=tensorrt_path
        )

        if task == "question-answering":
            tensorrt_inf: Callable[[Dict[str, torch.Tensor]], List[torch.Tensor]] = lambda x: list(
                tensorrt_model(x).values()
            )
        else:
            tensorrt_inf: Callable[[Dict[str, torch.Tensor]], torch.Tensor] = lambda x: list(
                tensorrt_model(x).values()
            )[0]

        logging.info("running TensorRT (FP16) benchmark")
        engine_name = "TensorRT (FP16)"
        tensorrt_output, time_buffer = launch_inference(
            infer=tensorrt_inf, inputs=inputs_pytorch, nb_measures=nb_measures
        )
        check_accuracy(
            engine_name=engine_name,
            pytorch_output=pytorch_output,
            engine_output=tensorrt_output,
            tolerance=atol,
        )
        timings[engine_name] = time_buffer
        del tensorrt_model, runtime  # delete all tensorrt objects
        gc.collect()

    if "onnx" in backend:
        num_attention_heads, hidden_size = get_model_size(path=model)
        # create optimized onnx model and compare results
        if device == "cpu" and quantization:
            cpu_quantization(input_model_path=onnx_optim_model_path, output_model_path=onnx_optim_model_path)

        ort_provider = "CUDAExecutionProvider" if run_on_cuda else "CPUExecutionProvider"
        for provider, model_path, benchmark_name in [
            (ort_provider, onnx_model_path, "ONNX Runtime (FP32)"),
            (ort_provider, onnx_optim_model_path, "ONNX Runtime (optimized)"),
        ]:
            logging.info("preparing %s benchmark", benchmark_name)
            ort_model = create_model_for_provider(
                path=model_path,
                provider_to_use=provider,
                nb_threads=nb_threads,
            )

            def infer_ort(inputs: Dict[str, torch.Tensor]) -> torch.Tensor:
                results = inference_onnx_binding(model_onnx=ort_model, inputs=inputs, device=device)
                return results["output"] if "output" in results else (results["start_logits"], results["end_logits"])

            logging.info("running %s benchmark", benchmark_name)
            ort_output, time_buffer = launch_inference(
                infer=infer_ort, inputs=inputs_pytorch, nb_measures=nb_measures
            )
            check_accuracy(
                engine_name=benchmark_name,
                pytorch_output=pytorch_output,
                engine_output=ort_output,
                tolerance=atol,
            )
            timings[benchmark_name] = time_buffer
            del ort_model
            gc.collect()

    if run_on_cuda:
        from torch.cuda import get_device_name

        print(f"Inference done on {get_device_name(0)}")

    print("latencies:")
    for name, time_buffer in timings.items():
        print_timings(name=name, timings=time_buffer)
    print(f"Each infence engine output is within {atol} tolerance compared to Pytorch output")
