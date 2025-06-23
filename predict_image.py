import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from unet import UNet
import datetime


def predict_mask(net, image_tensor, device, out_threshold=0.5):
    """
    Performs inference on a single image tensor.
    """
    net.eval()
    
    img = image_tensor.unsqueeze(0)
    img = img.to(device=device, dtype=torch.float32)

    with torch.no_grad():
        output = net(img).cpu()
        
        if net.n_classes > 1:
            mask = output.argmax(dim=1)
        else:
            mask = torch.sigmoid(output) > out_threshold

    return mask[0].long().squeeze().numpy()


def blend_image_and_mask(original_image, mask_array, color, alpha=0.4):
    """
    Blends a mask over an original image.
    """
    original_image = original_image.convert("RGBA")
    overlay = Image.new("RGBA", original_image.size, (0, 0, 0, 0))
    overlay_draw = np.array(overlay)
    
    overlay_draw[mask_array == 1] = (*color, int(255 * alpha))
    
    overlay = Image.fromarray(overlay_draw)
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
    parser = argparse.ArgumentParser(description='Predict masks from a single image')
    parser.add_argument('--model', '-m', required=True,
                        help='Specify the absolute file path of the trained model')
    parser.add_argument('--image_path', '-i', required=True, type=str,
                        help='Absolute path to the input image')
    parser.add_argument('--output_dir', '-o', type=str, default='results/image_predictions',
                        help='Directory to save the output image')
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

    script_dir = Path(__file__).resolve().parent
    run_timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    run_name = f'predicted_image_{run_timestamp}'

    output_path = script_dir / args.output_dir / run_name
 
    model_path = args.model
    output_path.mkdir(parents=True, exist_ok=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Loading model from {args.model}')
    logging.info(f'Using device {device}')

    net = UNet(n_channels=3, n_classes=args.classes, bilinear=args.bilinear, simple=args.simple)
    net.to(device=device)
    
    try:
        state_dict = torch.load(args.model, map_location=device)
        _ = state_dict.pop('mask_values', None) 
        net.load_state_dict(state_dict)
        logging.info('Model loaded!')
    except FileNotFoundError:
        logging.error(f"Model file not found at {args.model}. Please check the path.")
        exit()

    image_transforms = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    try:
        original_pil_img = Image.open(args.image_path).convert("RGB")
    except FileNotFoundError:
        logging.error(f"Input image not found at {args.image_path}.")
        exit()

    image_tensor = image_transforms(original_pil_img)

    logging.info("Predicting mask...")
    predicted_mask_np = predict_mask(
        net=net,
        image_tensor=image_tensor,
        device=device,
        out_threshold=args.mask_threshold
    )
    
    PRED_COLOR = (255, 0, 0)  # Red

    blended_pred = blend_image_and_mask(original_pil_img, predicted_mask_np, color=PRED_COLOR)
    
    final_image = concatenate_images(original_pil_img, blended_pred)
    
    image_filename = Path(args.image_path).name
    output_filename = output_path / f'{Path(image_filename).stem}_prediction.png'
    final_image.save(output_filename)

    logging.info(f"Prediction complete. Result saved to '{output_filename}'.")