import sys
import argparse
import subprocess
import torch
import torch.onnx
from model import YOLO, load_model

def torch2onnx(model, output_path, input_shape):
    model.eval()

    dummy_input = torch.randn(input_shape)

    # export to onnx model
    torch.onnx.export(
            model,
            dummy_input,
            output_path,
            export_params=True,     # 学習したパラメータを格納する
            verbose=True
    )

    return output_path

def onnx2tf(input_path, output_path):
    subprocess.run(
            f"onnx-tf convert -i {input_path} -o {output_path}", shell=True)

def convert(model, onnx_filepath, pb_filepath, input_shape):
    print("Converting torch to onnx...")
    torch2onnx(model, onnx_filepath, input_shape)

    print("Converting onnx to tensorflow...")
    onnx2tf(onnx_filepath, pb_filepath)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', default="results/20211109_191529/yolo-tiny.pt")
    parser.add_argument('--onnx_path', default="yolo-tiny.onnx")
    parser.add_argument('--pb_path', default="yolo-tiny.pb")
    args = parser.parse_args()

    device = torch.device("cpu")

    INPUT_WIDTH  = 416
    INPUT_HEIGHT = 416
    INPUT_SHAPE  = (1, 3, INPUT_WIDTH, INPUT_HEIGHT)
    NUM_CLASSES  = 1

    model     = load_model(None, device, num_classes=NUM_CLASSES)
    onnx_path = args.onnx_path
    pb_path   = args.pb_path

    #convert(model, onnx_path, pb_path, INPUT_SHAPE)
    torch2onnx(model, onnx_path, INPUT_SHAPE)

if __name__ == "__main__":
    main()
