import warnings
warnings.filterwarnings("ignore")
import io
import os
from PIL import Image
import numpy as np
import torch
# import tvm.relay as relay
import tvm
from tvm.contrib.download import download_testdata
from tvm.contrib import graph_executor, utils
import tvm.auto_scheduler as auto_scheduler
from tvm.autotvm.tuner import XGBTuner
from tvm import autotvm
import pickle
import copy
import numpy as np
import random
import onnx, onnxoptimizer
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import subprocess
import tensorrt as trt

from torch.ao.quantization import get_default_qconfig 
import torch.quantization.quantize_fx as quantize_fx
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx, fuse_fx

from voltaml.build_engine import EngineBuilder


import argparse
import gc
import logging
import os
from pathlib import Path
from typing import Callable, Dict, List, Tuple, Type, Union

import numpy as np
import torch
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


class TVMCompiler:

    def  __init__(self, model, type='pytorch', target='llvm -mcpu=core-avx2', tune=True, tuning_records_dir='tuning_records.json', output_dir='tvm_model', input_shape=(1, 3, 224, 224)) -> None:
        model = model.eval()
        self.model = model
        self.type = type
        self.target = target
        self.tune = tune
        self.tuning_records_dir = tuning_records_dir
        self.output_dir = output_dir
        self.input_shape = input_shape
        self.input_shape_tvm = list(self.input_shape)
        self.input_shape_tvm[0] = relay.Any()
        self.input_shape_tvm = tuple(self.input_shape_tvm)
        input_name = "data"
        self.shape_dict = {
            input_name: self.input_shape
        }
        self.img_data = np.random.rand(*self.input_shape)

    def compile(self):
        
    
        if os.path.exists(os.path.join(self.output_dir, 'mod.pkl')) and os.path.exists(os.path.join(self.output_dir, 'params.pkl')) and os.path.exists(os.path.join(self.output_dir, 'lib.so')):
            self.load_compiled_model()
            loaded = True
        else:

            if self.type == 'pytorch':

                input_names = ["data"]
                output_names = ["output"]

                # jit_model = torch.jit.trace(model, torch.tensor(self.img_data).float()) 

                with io.BytesIO() as f:
                    torch.onnx.export(self.model, 
                        torch.tensor(self.img_data).float(),
                        f,
                        verbose=False,
                        input_names=input_names,
                        output_names=output_names,
                        export_params=True,
                    )
                    f.seek(0)

                    onnx_model = onnx.load_model(f, onnx.ModelProto)
                
                # convert pytorch model to tvm relay graph
                # self.mod, self.params = relay.frontend.from_pytorch(jit_model, self.shape_dict)
                self.mod, self.params = relay.frontend.from_onnx(onnx_model, self.shape_dict)

            elif self.type == 'onnx':
                onnx_model = onnx.load(self.model_dir)
                self.mod, self.params = relay.frontend.from_onnx(onnx_model, self.shape_dict)
            else:
                raise NotImplementedError("Models other than pytorch models are not supported yet.")

            if self.tune:
                tuning_option = self.tune_model()
                with autotvm.apply_history_best(tuning_option["tuning_records"]):
                    with tvm.transform.PassContext(opt_level=3, config={}):
                        self.lib = relay.build(self.mod, target=self.target, params=self.params)
            else:
                with tvm.transform.PassContext(opt_level=3):
                    self.lib = relay.build(self.mod, target=self.target, params=self.params)
            
            loaded = False

        self.dev = tvm.device(str(self.target), 0)
        self.module = graph_executor.GraphModule(self.lib["default"](self.dev)) # tvm python object

        if not loaded:
            self.save_compiled_model()
        
        return self.module

    def tune_model(self):
        number = 10
        repeat = 1
        min_repeat_ms = 0  # since we're tuning on a CPU, can be set to 0
        timeout = 10  # in seconds

        # create a TVM runner
        runner = autotvm.LocalRunner(
            number=number,
            repeat=repeat,
            timeout=timeout,
            min_repeat_ms=min_repeat_ms,
            enable_cpu_cache_flush=True,
        )

        tuning_option = {
            "tuner": "xgb",
            "trials": 10,
            "early_stopping": 100,
            "measure_option": autotvm.measure_option(
                builder=autotvm.LocalBuilder(build_func="default"), runner=runner
            ),
            "tuning_records": self.tuning_records_dir,
        }

        # begin by extracting the tasks from the onnx model
        tasks = autotvm.task.extract_from_program(self.mod["main"], target=self.target, params=self.params)

        # Tune the extracted tasks sequentially.
        for i, task in enumerate(tasks):
            prefix = "[Task %2d/%2d] " % (i + 1, len(tasks))
            tuner_obj = XGBTuner(task, loss_type="rank")
            tuner_obj.tune(
                n_trial=min(tuning_option["trials"], len(task.config_space)),
                early_stopping=tuning_option["early_stopping"],
                measure_option=tuning_option["measure_option"],
                callbacks=[
                    autotvm.callback.progress_bar(tuning_option["trials"], prefix=prefix),
                    autotvm.callback.log_to_file(tuning_option["tuning_records"]),
                ],
            )

        return tuning_option

    def save_compiled_model(self):
        # TODO: Save all model specific objects properly.
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        # saving the relay representation
        with open(os.path.join(self.output_dir, 'mod.pkl'), 'wb') as f:
            pickle.dump(self.mod, f)

        # saving params
        params = relay.save_param_dict(self.params)
        with open(os.path.join(self.output_dir, 'params.pkl'), 'wb') as f:
            pickle.dump(params, f)

        # # saving lib
        self.lib.export_library(os.path.join(self.output_dir, 'lib.so'))

    def load_compiled_model(self):
        with open(os.path.join(self.output_dir, 'mod.pkl'), 'rb') as f:
            self.mod = pickle.load(f)
        with open(os.path.join(self.output_dir, 'params.pkl'), 'rb') as f:
            param_bytes = pickle.load(f)
        self.params = relay.load_param_dict(param_bytes)
        self.lib = tvm.runtime.load_module(os.path.join(self.output_dir, 'lib.so'))

    def infer(self, input_name, img_data, output_shape):
        # dtype = "float32"
        self.module.set_input(input_name, img_data)
        self.module.run()
        tvm_output = self.module.get_output(0, tvm.nd.empty(output_shape)).numpy()

        return tvm_output


class VoltaCPUCompiler:

    def __init__(self, model, output_dir, input_shape, save=True) -> None:
        model = model.eval()
        model = model.to(torch.device("cpu:0"))
        self.model = model
        self.output_dir = output_dir
        self.input_shape = input_shape
        self.save = save

    def compile(self):
        # `qconfig` means quantization configuration, it specifies how should we
        #  observe the activation and weight of an operator
        # `qconfig_dict`, specifies the `qconfig` for each operator in the model
        #  we can specify `qconfig` for certain types of modules
        #  we can specify `qconfig` for a specific submodule in the model
        #  we can specify `qconfig` for some functioanl calls in the model
        #  we can also set `qconfig` to None to skip quantization for some operators
        
        model_fused = quantize_fx.fuse_fx(self.model)
        qconfig = get_default_qconfig("fbgemm")
        qconfig_dict = {"": qconfig}
        
        
        #  calibration runs the model with some sample data, which allows observers to record the statistics of
        #  the activation and weigths of the operators
        calibration_data = [torch.randn(self.input_shape) for _ in range(100)]
        for i in range(len(calibration_data)):
            self.model(calibration_data[i])
        
        # `convert_fx` converts a calibrated model to a quantized model, this includes inserting
        #  quantize, dequantize operators to the model and swap floating point operators with quantized operators
        prepared_model = prepare_fx(self.model, qconfig_dict)  # fuse modules and insert observers
        model_quantized = convert_fx(prepared_model) 
        # model_quantized = convert_fx(copy.deepcopy(model_prepared))# benchmark
        model_jit = torch.jit.script(model_quantized)

        if self.save:
            torch.jit.save(model_jit, self.output_dir)
        
        return model_jit

class VoltaGPUCompiler:

    def __init__(self, model, output_dir, 
                 input_shape, precision,
                 input_name='',
                 output_name='',
                 dynamic=False,
                 simplify=False,
                 opset_version=13,
                 calib_input='/workspace/TensorRT/ILSVRC2012_img_val/', 
                 calib_cache='./calibration.cache', calib_num_images=25000, 
                 calib_batch_size=8, 
                 calib_preprocessor='V2', save=True) -> None:
        
        model = model.eval()
        model = model.to(torch.device("cpu:0"))
        self.model = model
        self.output_dir = output_dir
        self.input_shape = input_shape
        self.precision = precision
        self.save = save
        self.calib_input = calib_input
        self.calib_cache = calib_cache
        self.calib_num_images = calib_num_images
        self.calib_batch_size = calib_batch_size
        self.calib_preprocessor = calib_preprocessor
        self.input_name = input_name
        self.output_name = output_name
        self.dynamic = dynamic
        self.simplify = simplify
        self.opset_version = opset_version

    def compile(self):
        dummy_input = torch.rand(*self.input_shape)

        torch.onnx.export(self.model, dummy_input, "tmp.onnx", verbose=False,
                          input_names=[self.input_name],
                          output_names=[self.output_name],
                          opset_version=self.opset_version,
                          dynamic_axes={
                                            'images': {
                                                0: 'batch',
                                                2: 'height',
                                                3: 'width'},  # shape(1,3,640,640)
                                            'output': {
                                                0: 'batch',
                                                1: 'anchors'}  # shape(1,25200,85)
                                        } if self.dynamic else None
                         )
        
        # Simplify
        if self.simplify:
            try:
                import onnxsim
                f = 'tmp.onnx'
                print('-------- Loading ONNX ---------------')
                # Checks
                model_onnx = onnx.load(f)  # load onnx model
                onnx.checker.check_model(model_onnx)  # check onnx model
                model_onnx, check = onnxsim.simplify(f,
                                                     dynamic_input_shape=self.dynamic,
                                                     input_shapes={'images': list((640,640),(1280,1280))} if self.dynamic else None)
                assert check, 'assert check failed'
                onnx.save(model_onnx, f)
            except Exception as e:
                print(e)
                pass
                # LOGGER.info(f'{prefix} simplifier failure: {e}')
        
        builder = EngineBuilder(verbose=False)
        builder.create_network("tmp.onnx")
        if self.precision == 'int8':
            builder.create_engine(self.output_dir, self.precision, self.calib_input, self.calib_cache, self.calib_num_images,
                          self.calib_batch_size, self.calib_preprocessor)
        else:
            builder.create_engine(self.output_dir, self.precision)        
        
        os.remove("tmp.onnx")
        if os.path.exists('./calibration.cache'):
            os.remove('./calibration.cache')
        context = self.load_engine()
        return context

    def load_engine(self):
        f = open(self.output_dir, "rb")
        runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING)) 

        engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()
        return context
        
        
def VoltaNLPCompile(verbose=False,
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
    
    setup_logging(level=logging.INFO if verbose else logging.WARNING)
    # logging.info("running with commands: %s", commands)

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

    # create onnx model and compare results
    convert_to_onnx(
        model_pytorch=model_pytorch,
        output_path=onnx_model_path,
        inputs_pytorch=inputs_pytorch[0],
        quantization=quantization,
        var_output_seq=task in ["text-generation", "token-classification", "question-answering"],
        output_names=["output"] if task != "question-answering" else ["start_logits", "end_logits"],
    )

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
        engine: ICudaEngine = build_engine(
            runtime=runtime,
            onnx_file_path=onnx_model_path,
            logger=trt_logger,
            min_shape=tensor_shapes[0],
            optimal_shape=tensor_shapes[1],
            max_shape=tensor_shapes[2],
            workspace_size=workspace_size * 1024 * 1024,
            fp16=not quantization,
            int8=quantization,
        )
        save_engine(engine=engine, engine_file_path=tensorrt_path)
        # important to check the engine has been correctly serialized
        tensorrt_model: Callable[[Dict[str, torch.Tensor]], Dict[str, torch.Tensor]] = load_engine(
            runtime=runtime, engine_file_path=tensorrt_path
        )

        del engine, tensorrt_model, runtime  # delete all tensorrt objects
        gc.collect()
        
    if "onnx" in backend:
        num_attention_heads, hidden_size = get_model_size(path=model)
        # create optimized onnx model and compare results
        optimize_onnx(
            onnx_path=onnx_model_path,
            onnx_optim_model_path=onnx_optim_model_path,
            fp16=run_on_cuda,
            use_cuda=run_on_cuda,
            num_attention_heads=num_attention_heads,
            hidden_size=hidden_size,
            architecture=model_config.model_type,
        )
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

            del ort_model
            gc.collect()

    print("Engine Building Completed Successfully.")
    
