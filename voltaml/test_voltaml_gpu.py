import os
import sys
sys.path.append(os.path.abspath(os.path.pardir))
from argparse import ArgumentParser
from voltaml.models.common import DetectMultiBackend

import torchvision
import torch
from voltaml.compile import VoltaGPUCompiler
from voltaml.inference import gpu_performance
from voltaml.utils.torch_utils import select_device, time_sync

from voltaml.yolov6.utils.checkpoint import load_checkpoint
from voltaml.yolov6.layers.common import DetectBackend

def main():
    args = get_args()
    
    # model = torch.load(args.torch_model_dir)
    # model = torch.hub.load('ultralytics/yolov5', 'yolov5s', device='cuda:0')
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)


    # if args.is_yolo:
    #     device = select_device('0')
    #     model = DetectMultiBackend(args.torch_model_dir, device=device)
    # else:
    #     model = torch.load(args.torch_model_dir)
    # Download a pretrained model
    torch.hub.download_url_to_file('https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6l.pt', 'yolov6l.pt')
    torch_model_dir = 'yolov6l.pt'
    device = torch.device(0)
    model = load_checkpoint(torch_model_dir, map_location=device, inplace=True, fuse=False)  # load FP32 model
    # download or load the model from disk
    model.eval()
    
    # model = torch.hub.load('pytorch/vision:v0.10.0', 'deeplabv3_resnet50', pretrained=True)

    compiler = VoltaGPUCompiler(
        model=model,
        output_dir=args.compiled_model_dir,
        input_shape=args.input_shape,
        precision=args.precision,
        input_name=args.input_name,
        output_name=args.output_name,
        dynamic=args.dynamic,
        simplify=args.simplify,
        opset_version=args.opset,
        calib_input=args.calib_input,
        calib_cache=args.calib_cache,
        calib_num_images=args.calib_num_images,
        calib_batch_size=args.calib_batch_size,
        calib_preprocessor=args.calib_preprocessor
    )

    compiled_model = compiler.compile()

    gpu_performance(args.compiled_model_dir, model, input_shape=args.input_shape, throughput_batch_size=args.throughput_batch_size, is_yolo=args.is_yolo)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--torch_model_dir", default="/workspace/voltaML/voltaml/yolov6l.pt", type=str)
    parser.add_argument("--compiled_model_dir", default="/workspace/voltaML/voltaml/yolov6l.engine", type=str)
    parser.add_argument("--input_shape", default=(1, 3, 640, 640), type=tuple)
    parser.add_argument("--input_name",default="", type=str)
    parser.add_argument("--output_name",default="", type=str)
    parser.add_argument("--dynamic", action='store_true', help='Dynamic ONNX')
    parser.add_argument("--simplify", action='store_true', help='Simplify ONNX')
    parser.add_argument("--is_yolo", action='store_true', help='If model is YoloV5/V6')
    parser.add_argument("--throughput_batch_size", default=8, type=int)
    parser.add_argument("--opset", default=13, type=int)
    parser.add_argument("--precision", default="fp16", type=str)
    parser.add_argument("--calib_input", help="The directory holding images to use for calibration")
    parser.add_argument("--calib_cache", default="models/calibration.cache",
                        help="The file path for INT8 calibration cache to use, default: ./calibration.cache")
    parser.add_argument("--calib_num_images", default=25000, type=int,
                        help="The maximum number of images to use for calibration, default: 25000")
    parser.add_argument("--calib_batch_size", default=8, type=int,
                        help="The batch size for the calibration process, default: 1")
    parser.add_argument("--calib_preprocessor", default="V2", choices=["V1", "V1MS", "V2"],
                        help="Set the calibration image preprocessor to use, either 'V2', 'V1' or 'V1MS', default: V2")
    
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()
    
   



