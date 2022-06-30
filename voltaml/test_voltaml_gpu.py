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
        precision=args.precision
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
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()