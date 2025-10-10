import argparse
import logging
import torch
import torch.nn as nn
from pathlib import Path

from unet import UNet


class OutputWrapper(nn.Module):
    """Wraps the UNet model to add a post-processing operation before ONNX export."""
    def __init__(self, model, output_type: str):
        super().__init__()
        self.model = model
        self.output_type = output_type.lower()

        valid_types = {'sigmoid', 'softmax', 'argmax', 'none'}
        if self.output_type not in valid_types:
            raise ValueError(f"Invalid output type '{output_type}'. Must be one of {valid_types}")

    def forward(self, x):
        x = self.model(x)

        if self.output_type == 'sigmoid':
            return torch.sigmoid(x).to(torch.float32)

        elif self.output_type == 'softmax':
            # Softmax along the class dimension (dim=1)
            return torch.softmax(x, dim=1).to(torch.float32)

        elif self.output_type == 'argmax':
            # Argmax along the class dimension, output int32 tensor
            return torch.argmax(x, dim=1, keepdim=True).to(torch.int32)

        # No change to raw logits
        return x


def get_args():
    """Parses command-line arguments for the export script."""
    parser = argparse.ArgumentParser(description='Export a PyTorch U-Net model to ONNX format')
    
    # --- Required Arguments ---
    parser.add_argument('--model-path', '-m', required=True, type=str,
                        help='Absolute path to the trained PyTorch model (.pth file)')
    parser.add_argument('--output-name', '-o', required=True, type=str,
                        help='Name of the exported ONNX model (.onnx file)')
    
    # --- Model Architecture Arguments ---
    parser.add_argument('--classes', '-c', type=int, default=1,
                        help='Number of output classes for the model')
    parser.add_argument('--bilinear', action='store_true', default=False,
                        help='Flag to indicate if the model uses bilinear upsampling')
    parser.add_argument('--simple', action='store_true', default=False,
                        help='Flag to indicate if the model is the "simple" smaller version')
    
    # --- ONNX Export Arguments ---
    parser.add_argument('--height', type=int, default=660,
                        help='The input image height for the ONNX model graph')
    parser.add_argument('--width', type=int, default=660,
                        help='The input image width for the ONNX model graph')

    # --- Output type selection ---
    parser.add_argument('--network-output-type', '-t', type=str, default='none',
                        choices=['sigmoid', 'softmax', 'argmax', 'none'],
                        help='Post-processing operation to apply before export. '
                             'Choices: sigmoid, softmax, argmax, none')

    return parser.parse_args()


def export_model_to_onnx(args):
    """Loads a trained U-Net model, applies post-processing, and exports it to ONNX format."""
    logging.info("Starting model export to ONNX...")

    # 1. Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # 2. Instantiate the U-Net
    n_channels = 3  # RGB
    net = UNet(n_channels=n_channels, n_classes=args.classes,
               bilinear=args.bilinear, simple=args.simple)
    net = net.to(memory_format=torch.channels_last) 
    logging.info(f"Instantiated U-Net with {n_channels} input channels and {args.classes} output classes.")

    # 3. Load the trained weights
    try:
        logging.info(f"Loading model state from {args.model_path}")
        state_dict = torch.load(args.model_path, map_location=device)
        if 'mask_values' in state_dict:
            del state_dict['mask_values']
            logging.info("Removed 'mask_values' key from state dictionary.")
        net.load_state_dict(state_dict)
        logging.info("Model weights loaded successfully.")
    except FileNotFoundError:
        logging.error(f"Model file not found at {args.model_path}. Please check the path.")
        return
    except Exception as e:
        logging.error(f"An error occurred while loading the model: {e}")
        return

    # 4. Wrap model with post-processing
    wrapped_model = OutputWrapper(net, args.network_output_type)
    wrapped_model.to(device)
    wrapped_model.eval()

    # 5. Create dummy input
    dummy_input = torch.randn(1, n_channels, args.height, args.width, requires_grad=False).to(device)
    logging.info(f"Created dummy input tensor of shape: {dummy_input.shape}")
    dummy_input = dummy_input.to(memory_format=torch.channels_last, device=device)
    logging.info("Dummy input tensor moved to device and memory format set.")
    logging.info(f"dummy_input shape after memory format change: {dummy_input.shape}")
    # 6. Export to ONNX
    output_path = Path(args.output_name)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Exporting model with '{args.network_output_type}' output type to {output_path}...")

    try:
        torch.onnx.export(
            wrapped_model,
            dummy_input,
            str(output_path),
            export_params=True,
            opset_version=11,
            do_constant_folding=True,
            input_names=['input'],
            output_names=['output'],
            dynamic_axes={
                'input': {0: 'batch_size', 2: 'height', 3: 'width'},
                'output': {0: 'batch_size', 2: 'height', 3: 'width'}
            }
        )

        logging.info(f"Model successfully exported to {output_path}")
        logging.info(f"Output type: {args.network_output_type}")
    except Exception as e:
        logging.error(f"An error occurred during ONNX export: {e}")


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    args = get_args()
    export_model_to_onnx(args)
