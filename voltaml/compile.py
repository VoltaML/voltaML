import warnings
warnings.filterwarnings("ignore")
import os
from PIL import Image
import numpy as np
import torch
import tvm.relay as relay
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

from voltaml.preprocess import preprocess_image
from voltaml.build_engine import EngineBuilder

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
        # self.shape_dict = shape_dict
        # self.img_data = preprocess_image(img_dir)
        input_name = "data"
        # self.shape_dict = [(input_name, self.input_shape_tvm)]
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

                torch.onnx.export(self.model, 
                    torch.tensor(self.img_data).float(),
                    "tmp.onnx",
                    verbose=False,
                    input_names=input_names,
                    output_names=output_names,
                    export_params=True,
                )

                onnx_model = onnx.load("tmp.onnx")
                
                # convert pytorch model to tvm relay graph
                # self.mod, self.params = relay.frontend.from_pytorch(jit_model, self.shape_dict)
                self.mod, self.params = relay.frontend.from_onnx(onnx_model, self.shape_dict)
                os.remove("tmp.onnx")

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
        qconfig = get_default_qconfig("qnnpack")
        qconfig_dict = {"": qconfig}
        
        # `prepare_fx` inserts observers in the model based on the configuration in `qconfig_dict`
        model_prepared = prepare_fx(model_fused, qconfig_dict)
        
        #  calibration runs the model with some sample data, which allows observers to record the statistics of
        #  the activation and weigths of the operators
        calibration_data = [torch.randn(self.input_shape) for _ in range(100)]
        for i in range(len(calibration_data)):
            model_prepared(calibration_data[i])
        
        # `convert_fx` converts a calibrated model to a quantized model, this includes inserting
        #  quantize, dequantize operators to the model and swap floating point operators with quantized operators
        model_quantized = convert_fx(copy.deepcopy(model_prepared))# benchmark
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
                # LOGGER.info(f'{prefix} simplifying with onnx-simplifier {onnxsim.__version__}...')
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
        