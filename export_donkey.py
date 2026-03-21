"""Export the best DonkeyCar model to ONNX for deployment on Raspberry Pi / Jetson."""
import argparse

import numpy as np
import torch

from train_donkey import DonkeyNet
from prepare_donkey import IMG_H, IMG_W, BEST_MODEL_PATH


def export(model_path=BEST_MODEL_PATH, output="autopilot.onnx"):
    model = DonkeyNet()
    model.load_state_dict(torch.load(model_path, map_location="cpu", weights_only=True))
    model.eval()
    dummy = torch.zeros(1, 3, IMG_H, IMG_W)
    torch.onnx.export(model, dummy, output,
                      input_names=["image"], output_names=["controls"],
                      opset_version=17, dynamic_axes=None)
    # Validation: compare ONNX output against PyTorch
    import onnxruntime as ort
    session = ort.InferenceSession(output)
    pt_out = model(dummy).detach().numpy()
    onnx_out = session.run(None, {"image": dummy.numpy()})[0]
    max_diff = np.max(np.abs(pt_out - onnx_out))
    print(f"Exported {output}  (max PyTorch/ONNX diff: {max_diff:.6f})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=BEST_MODEL_PATH, help="Path to .pth checkpoint")
    parser.add_argument("--output", default="autopilot.onnx", help="Output ONNX path")
    args = parser.parse_args()
    export(args.model, args.output)
