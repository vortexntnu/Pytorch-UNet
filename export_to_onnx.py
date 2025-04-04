import torch
import torch.nn as nn
from unet.unet_model import UNet  # Import from your local repo

# ─── Wrapper to Include Argmax and Reorder Dimensions ────────────────────────
class UNetWithArgmax(nn.Module):
    def __init__(self, model):
        super(UNetWithArgmax, self).__init__()
        self.model = model

    def forward(self, x):
        logits = self.model(x)  # Output shape: (B, C, H, W)
        pred = torch.argmax(logits, dim=1, keepdim=True)  # Shape: (B, 1, H, W)
        pred = pred.permute(0, 2, 3, 1)  # Reorder to (B, H, W, 1)
        return pred

# ─── Load Base Model ─────────────────────────────────────────────────────────
pth_file = "/home/vortex/Downloads/pipeline_segmentation_v0.2.pth"
base_model = UNet(n_channels=3, n_classes=2, bilinear=False)
print(f"Loading model from {pth_file}")
base_model.load_state_dict(torch.load(pth_file, map_location="cpu"), strict=False)
base_model.eval()

# ─── Wrap with Argmax ────────────────────────────────────────────────────────
model = UNetWithArgmax(base_model)
model.eval()

# ─── Dummy Input ─────────────────────────────────────────────────────────────
batch_size = 1
input_shape = (batch_size, 3, 540, 720)  # Keep original resolution
dummy_input = torch.randn(input_shape)
print(f"Dummy input shape: {dummy_input.shape}")

# ─── Sanity Check ────────────────────────────────────────────────────────────
with torch.no_grad():
    output = model(dummy_input)
    print(f"Model output shape (after argmax and reorder): {output.shape}")
    print(f"Output min value: {output.min().item()}, max value: {output.max().item()}")

# ─── Export to ONNX ──────────────────────────────────────────────────────────
onnx_file = "unet_argmax_nhwc.onnx"
print(f"Exporting model to {onnx_file}")
torch.onnx.export(
    model,
    dummy_input,
    onnx_file,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size", 1: "height", 2: "width"}  # Adjusted for NHWC
    },
    opset_version=11  # Compatible with TensorRT
)

print(f"✅ Model exported to {onnx_file}")