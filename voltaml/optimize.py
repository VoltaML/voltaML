import copy
import torch
from torchvision import datasets, transforms

from torch.ao.quantization import get_default_qconfig 
import torch.quantization.quantize_fx as quantize_fx
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx, fuse_fx


class CPUOptimizer:

    def __init__(self, model_dir, input_shape, output_dir, save=True) -> None:
        self.model_dir = model_dir
        self.input_shape = input_shape
        self.output_dir = output_dir
        self.save = save
        self.cpu_device = torch.device("cpu:0")
        self.gpu_device = torch.device("cuda:0")
        self.model = self.load_model(self.model_dir)

    def load_model(self, model_dir):
        model = torch.load(model_dir)
        model = model.eval()
        model = model.to(self.cpu_device)
        return model

    def optimize(self):
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
        
        return model_jit, self.model

    def fuse_model(self, model):
        for module_name, module in model.named_children():
            if "layer" in module_name:
                for basic_block_name, basic_block in module.named_children():
                    torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu"], ["conv2", "bn2"]], inplace=True)
                    # torch.quantization.fuse_modules(basic_block, [["conv1", "bn1", "relu" ,"conv2", "bn2"]], inplace=True)
                    for sub_block_name, sub_block in basic_block.named_children():
                        if sub_block_name == "downsample":
                            torch.quantization.fuse_modules(sub_block, [["0", "1"]], inplace=True)
        return model
        