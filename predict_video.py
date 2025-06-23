import argparse
import logging
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from tqdm import tqdm
import cv2

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


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from a video file')
    parser.add_argument('--model', '-m', required=True,
                        help='Specify the absolute file path of the trained model')
    parser.add_argument('--video_path', '-i', required=True, type=str,
                        help='Absolute path to the input video file')
    parser.add_argument('--output_dir', '-o', type=str, default='results/video_predictions',
                        help='Directory to save the output video')
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
    run_name = f'predicted_video_{run_timestamp}'

    output_path = script_dir / args.output_dir / run_name
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
        cap = cv2.VideoCapture(args.video_path)
        if not cap.isOpened():
            raise IOError
    except IOError:
        logging.error(f"Cannot open video file at {args.video_path}")
        exit()
        
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    video_filename = Path(args.video_path).name
    output_filename = output_path / f'{Path(video_filename).stem}_prediction.mp4'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_filename), fourcc, fps, (frame_width, frame_height))

    logging.info(f"Processing video: {args.video_path}")
    
    PRED_COLOR = (255, 0, 0)

    with tqdm(total=total_frames, desc="Processing video frames") as pbar:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            
            image_tensor = image_transforms(pil_img)
            
            predicted_mask_np = predict_mask(
                net=net,
                image_tensor=image_tensor,
                device=device,
                out_threshold=args.mask_threshold
            )
            
            blended_img = blend_image_and_mask(pil_img, predicted_mask_np, color=PRED_COLOR)
            
            blended_frame = cv2.cvtColor(np.array(blended_img), cv2.COLOR_RGB2BGR)
            out.write(blended_frame)
            
            pbar.update(1)

    cap.release()
    out.release()
    cv2.destroyAllWindows()

    logging.info(f"Video processing complete. Result saved to '{output_filename}'.")