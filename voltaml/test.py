import os
import sys
# sys.path.append("/media/ritesh/HHD21/apache-tvm/")
sys.path.append(os.path.abspath(os.path.pardir))
from argparse import ArgumentParser

from voltaml.compile import CPUCompiler
from voltaml.inference import performance


def main():
    args = get_args()

    compiler = CPUCompiler(
        model_dir=args.torch_model_dir,
        output_dir=args.compiled_model_dir,
        img_dir=args.image_dir,
        target="llvm"
    )

    compiled_model, torch_model = compiler.compile(return_torch_model=True)

    performance(args.image_dir, compiled_model, torch_model)
    # compute_performance(args.image_dir, args.compiled_model_dir, args.torch_model_dir)

def get_args():
    parser = ArgumentParser()
    parser.add_argument("--torch_model_dir", default="/media/abhirooptejomay/7867a1d9-0cf5-4857-a984-03e6265ebdcf/studies/voltaml/resnet18.pth", type=str)
    parser.add_argument("--compiled_model_dir", default="/media/abhirooptejomay/7867a1d9-0cf5-4857-a984-03e6265ebdcf/studies/voltaml/resnet18_quantized_compiled_run5", type=str)
    parser.add_argument("--input_shape", default=(1, 3, 224, 224), type=tuple)
    args = parser.parse_args()
    return args

if __name__ == "__main__":
    main()