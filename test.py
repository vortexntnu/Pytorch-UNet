import argparse
import logging
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from tqdm import tqdm
from torchvision import transforms

from unet import UNet
from utils.roboflow_dataloader import SemanticSegmentationDataset
import datetime


def predict_mask(net, image_tensor, device, out_threshold=0.5):
    """
    Performs inference on a single image tensor.
    """
    net.eval()
    
    # Add a batch dimension and move to the correct device
    img = image_tensor.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        
        if net.n_classes > 1:
            # Multi-class: get the class with the highest score
            mask = output.argmax(dim=1)
        else:
            # Binary: apply sigmoid and threshold
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def blend_image_and_mask(original_image, mask_array, color, alpha=0.4):
    """
    Blends a mask over an original image.
    
    Args:
        original_image (PIL.Image): The source image.
        mask_array (np.ndarray): A 2D numpy array of the mask (with 0s and 1s).
        color (tuple): The RGB color for the mask overlay.
        alpha (float): The transparency of the mask overlay.
    
    Returns:
        PIL.Image: The blended image.
    """
    # Convert original image to RGBA to handle alpha channel
    original_image = original_image.convert("RGBA")
    
    # Create a solid color overlay from the mask
    overlay = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
    overlay_draw = np.array(overlay)
    
    # Find where the mask is active (value of 1) and apply the color
    overlay_draw[mask_array == 1] = (*color, int(255 * alpha))
    
    overlay = Image.fromarray(overlay_draw)
    
    # Blend the overlay with the original image
    blended_image = Image.alpha_composite(original_image, overlay)
    
    return blended_image.convert("RGB")


def concatenate_images(img1, img2):
    """
    Concatenates two PIL images horizontally.
    """
    new_width = img1.width + img2.width
    new_height = max(img1.height, img2.height)
    
    new_img = Image.new('RGB', (new_width, new_height))
    new_img.paste(img1, (0, 0))
    new_img.paste(img2, (img1.width, 0))
    
    return new_img


def get_args():
    parser = argparse.ArgumentParser(description='Test a trained UNet model on a test set')
    parser.add_argument('--model', '-m', default=None,
                        help='Specify the absolute file path of the trained model')
    parser.add_argument('--test_dir', type=str, default='data/test',
                        help='Directory of the test set')
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save the output images')
    parser.add_argument('--mask-threshold', '-t', type=float, default=0.5,
                        help='Minimum probability value to consider a mask pixel white')
    parser.add_argument('--bilinear', action='store_true', default=False,
                        help='Use bilinear upsampling')
    parser.add_argument('--simple', action='store_true', default=False,
                        help='Use a smaller UNet architecture')
    parser.add_argument('--classes', '-c', type=int, default=1,
                        help='Number of classes in the model')

    return parser.parse_args()


if __name__ == '__main__':
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    # Setup directories
    script_dir = Path(__file__).resolve().parent
    test_data_path = script_dir / args.test_dir

    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S') # e.g., '20250622_175526'
    run_name = f'test_{run_timestamp}'

    output_path = script_dir / args.output_dir / run_name

    model_path = args.model
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model from {model_path}')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear, simple=args.simple)
    net.to(device=device)
    
    try:
        state_dict = torch.load(model_path, map_location=device)
        # Remove metadata key before loading weights
        _ = state_dict.pop('mask_values', None) 
        net.load_state_dict(state_dict)
        logging.info('Model loaded!')
    except FileNotFoundError:
        logging.error(f"Model file not found at {model_path}. Please check the path.")
        exit()

    # Create dataset (without transformations for the original image)
    # Define transformations for the model input
    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = SemanticSegmentationDataset(
        dir=test_data_path,
        image_transform=image_transforms
    )

    # Colors for blending
    GT_COLOR = (0, 255, 0)      # Green for Ground Truth
    PRED_COLOR = (255, 0, 0)    # Red for Prediction

    # Loop through the test set
    for i in tqdm(range(len(test_dataset)), desc="Processing test images"):
        # Get data from dataset
        sample = test_dataset[i]
        image_tensor = sample['image']
        gt_mask_tensor = sample['mask']

        # The dataset returns tensors. We need the original PIL image for blending.
        original_pil_img = Image.open(test_dataset.images[i]).convert("RGB")

        # Predict the mask
        predicted_mask_np = predict_mask(
            net=net,
            image_tensor=image_tensor,
            device=device,
            out_threshold=args.mask_threshold
        )
        
        # Convert ground truth tensor to numpy array for blending
        gt_mask_np = gt_mask_tensor.squeeze().cpu().numpy()

        # Create blended images
        blended_gt = blend_image_and_mask(original_pil_img, gt_mask_np, color=GT_COLOR)
        blended_pred = blend_image_and_mask(original_pil_img, predicted_mask_np, color=PRED_COLOR)
        
        # Concatenate images side-by-side
        final_image = concatenate_images(blended_gt, blended_pred)
        
        # Save the result
        image_filename = Path(test_dataset.images[i]).name
        output_filename = output_path / f'{Path(image_filename).stem}_result.png'
        final_image.save(output_filename)

    logging.info(f"Testing complete. Results saved to '{output_path}'.")