import sys
import argparse
import torch
import torch.onnx
from model import load_model

def torch2onnx(model, output_path, input_shape):
    model.eval()

    dummy_input = torch.randn(input_shape)

    # export to onnx model
    torch.onnx.export(
            model,
            dummy_input,
            output_path,
            opset_version=11,
            do_constant_folding=False,
            export_params=True,     # 学習したパラメータを格納する
            verbose=True
    )

    return output_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=None)
    parser.add_argument('--weights', default="yolo-tiny_doll.pt")
    args = parser.parse_args()

    sep = True if args.model == 'sep' else False
    device = torch.device("cpu")

    INPUT_WIDTH  = 416
    INPUT_HEIGHT = 416
    INPUT_SHAPE  = (1, 3, INPUT_WIDTH, INPUT_HEIGHT)
    NUM_CLASSES  = 1

    model     = load_model(args.weights, device, num_classes=NUM_CLASSES)
    input_path, _ = os.path.splitext(weights_path)
    onnx_path     = input_path + '.onnx'

    print("Converting torch to onnx...")
    torch2onnx(model, onnx_path, INPUT_SHAPE)

if __name__ == "__main__":
    main()
