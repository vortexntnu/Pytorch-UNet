import torch
from unet.unet_model import UNet  # Import from your local repo

# Load the trained model
pth_file = "/home/simon/Downloads/checkpoint_epoch100.pth"
model = UNet(n_channels=3, n_classes=2, bilinear=False)  # 2 classes, RGB input, bilinear=False
print(f"Loading model from {pth_file}")
model.load_state_dict(torch.load(pth_file, map_location="cpu"), strict=False)
print(f"{model.n_classes}")
model.eval()

# Define dummy input (match your inference pipeline size)
batch_size = 1
input_shape = (batch_size, 3, 540, 720)  # Matches your ROS 2 pipeline
dummy_input = torch.randn(input_shape)
print(f"Dummy input shape: {dummy_input.shape}")

# Run a forward pass to check output shape
with torch.no_grad():
    output = model(dummy_input)
    print(f"Model output shape: {output.shape}")
    print(f"Model output min value: {output.min().item()}, max value: {output.max().item()}")

# Export to ONNX with dynamic axes
onnx_file = "unet.onnx"
print(f"Exporting model to {onnx_file}")
torch.onnx.export(
    model,
    dummy_input,
    onnx_file,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size", 2: "height", 3: "width"},
        "output": {0: "batch_size", 2: "height", 3: "width"}
    },
    opset_version=11,  # Compatible with TensorRT
    # verbose=True  # For debugging
)

print(f"Model exported to {onnx_file}")