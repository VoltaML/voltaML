import os
import sys
sys.path.append(os.path.abspath(os.path.pardir))
from argparse import ArgumentParser

from voltaml.optimize import CPUOptimizer
from voltaml.inference import compute_optimized_performance


def main():
    args = get_args()

    optimizer = CPUOptimizer(
        model_dir=args.model_dir,
        input_shape=args.input_shape,
        output_dir=args.output_dir
    )

    optimized_model, unoptimized_model = optimizer.optimize()
    compute_optimized_performance(optimized_model, unoptimized_model, device="cpu", input_shape=args.input_shape)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--model_dir", default="/media/abhirooptejomay/7867a1d9-0cf5-4857-a984-03e6265ebdcf/studies/voltaml/resnet18.pth", type=str)
    parser.add_argument("--input_shape", default=(1, 3, 224, 224), type=tuple)
    parser.add_argument("--output_dir", default="/media/abhirooptejomay/7867a1d9-0cf5-4857-a984-03e6265ebdcf/studies/voltaml/resnet18_quantized_run.pth", type=str)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()