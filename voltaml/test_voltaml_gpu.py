import os
import sys
sys.path.append(os.path.abspath(os.path.pardir))
from argparse import ArgumentParser

import torch
from voltaml.compile import VoltaGPUCompiler
from voltaml.inference import gpu_performance


def main():
    args = get_args()

    model = torch.load(args.torch_model_dir)

    compiler = VoltaGPUCompiler(
        model=model,
        output_dir=args.compiled_model_dir,
        input_shape=args.input_shape,
        precision=args.precision,
        calib_input=args.calib_input,
        calib_cache=args.calib_cache,
        calib_num_images=args.calib_num_images,
        calib_batch_size=args.calib_batch_size,
        calib_preprocessor=args.calib_preprocessor
    )

    compiled_model = compiler.compile()

    gpu_performance(args.compiled_model_dir, model, input_shape=args.input_shape, throughput_batch_size=args.throughput_batch_size)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--torch_model_dir", default="/media/abhirooptejomay/7867a1d9-0cf5-4857-a984-03e6265ebdcf/studies/voltaml/resnet18.pth", type=str)
    parser.add_argument("--compiled_model_dir", default="/media/abhirooptejomay/7867a1d9-0cf5-4857-a984-03e6265ebdcf/studies/voltaml/resnet18_compiled.pth", type=str)
    parser.add_argument("--input_shape", default=(1, 3, 224, 224), type=tuple)
    parser.add_argument("--throughput_batch_size", default=64, type=int)
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