import os
import sys
sys.path.append(os.path.abspath(os.path.pardir))
from argparse import ArgumentParser

import torch
from voltaml.compile import VoltaCPUCompiler
from voltaml.inference import cpu_performance


def main():
    args = get_args()

    model = torch.load(args.torch_model_dir)

    compiler = VoltaCPUCompiler(
        model=model,
        output_dir=args.compiled_model_dir,
        input_shape=args.input_shape,
    )

    compiled_model = compiler.compile()

    cpu_performance(compiled_model, model, compiler="voltaml", input_shape=args.input_shape, throughput_batch_size=args.throughput_batch_size)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--torch_model_dir", default="/resnet18.pth", type=str)
    parser.add_argument("--compiled_model_dir", default="/resnet18_compiled.pth", type=str)
    parser.add_argument("--input_shape", default=(1, 3, 224, 224), type=tuple)
    parser.add_argument("--throughput_batch_size", default=64, type=int)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()