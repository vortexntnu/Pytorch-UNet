import argparse
import logging
import torch

from pathlib import Path

from unet import UNet

def get_args():
    """Parses command-line arguments for the export script."""
    parser = argparse.ArgumentParser(description='Export a PyTorch U-Net model to ONNX format')
    
    # --- Required Arguments ---
    parser.add_argument('--model-path', '-m', required=True, type=str,
                        help='Absolute path to the trained PyTorch model (.pth file)')
    parser.add_argument('--output-name', '-o', required=True, type=str,
                        help='name of the exported ONNX model (.onnx file)')
    
    # --- Model Architecture Arguments (Must match the trained model) ---
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

    return parser.parse_args()

def export_model_to_onnx(args):
    """
    Loads a trained U-Net model, and exports it to the ONNX format.
    """
    logging.info("Starting model export to ONNX...")

    # 1. Set up the device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device: {device}')

    # 2. Instantiate the U-Net model with the correct architecture
    # The parameters (n_channels, n_classes, bilinear, simple) must match the
    # model that was trained to load the weights correctly.
    n_channels = 3  # Assuming RGB input images
    net = UNet(n_channels=n_channels, n_classes=args.classes, bilinear=args.bilinear, simple=args.simple)
    logging.info(f"Instantiated U-Net with {n_channels} input channels and {args.classes} output classes.")

    # 3. Load the trained weights from the .pth file
    try:
        logging.info(f"Loading model state from {args.model_path}")
        state_dict = torch.load(args.model_path, map_location=device)
        
        # The training script might save extra keys (like 'mask_values').
        # We remove it here to ensure compatibility with model.load_state_dict().
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
        
    # 4. Set the model to evaluation mode
    # This is a crucial step to disable dropout and batch normalization layers' training behavior.
    net.eval()
    net.to(device)
    logging.info("Model set to evaluation mode.")

    # 5. Create a dummy input tensor
    # ONNX export requires a sample input to trace the model's execution path.
    # The dimensions should match what the model expects for a single inference.
    batch_size = 1 # ONNX models are typically exported with a batch size of 1
    dummy_input = torch.randn(batch_size, n_channels, args.height, args.width, requires_grad=True).to(device)
    logging.info(f"Created a dummy input tensor of shape: {dummy_input.shape}")

    # 6. Export the model to ONNX
    output_path = Path(args.output_name)
    # # Ensure the output directory exists
    # output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Exporting model to {output_path}...")
    try:
        torch.onnx.export(
            net,                          # The model to be exported
            dummy_input,                  # A sample input to trace the graph
            str(output_path),             # Where to save the model
            export_params=True,           # Store the trained weights in the model file
            opset_version=11,             # The ONNX version to export the model to
            do_constant_folding=True,     # Whether to execute constant folding for optimization
            input_names=['input'],        # The model's input names
            output_names=['output'],      # The model's output names
            dynamic_axes={'input': {0: 'batch_size', 2: 'height', 3: 'width'},    # Dynamic axes for flexible input sizes
                          'output': {0: 'batch_size', 2: 'height', 3: 'width'}}
        )
        logging.info("Model successfully exported to ONNX.")
        logging.info(f"You can now use '{output_path}' for inference with TensorRT or other ONNX runtimes.")

    except Exception as e:
        logging.error(f"An error occurred during ONNX export: {e}")

if __name__ == '__main__':
    # Set up basic logging
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    
    # Parse arguments and run the export function
    args = get_args()
    export_model_to_onnx(args)
