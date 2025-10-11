import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import torch
from PIL import Image as PILImage
from torchvision import transforms

from unet import UNet
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy

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
    overlay = PILImage.new("RGBA", original_image.size, (0, 0, 0, 0))
    overlay_draw = np.array(overlay)
    
    overlay_draw[mask_array == 1] = (*color, int(255 * alpha))
    
    overlay = PILImage.fromarray(overlay_draw)
    blended_image = PILImage.alpha_composite(original_image, overlay)
    
    return blended_image.convert("RGB")

class ResizeIfLargerKeepAspect:
    """
    Downscale a PIL image only if it's larger than the target size,
    preserving aspect ratio. Never upscales.
    """
    def __init__(self, max_width, max_height, interpolation=transforms.InterpolationMode.BILINEAR):
        self.max_width = max_width
        self.max_height = max_height
        self.interpolation = interpolation

    def __call__(self, img):
        w, h = img.size

        # Only downscale if image exceeds limits
        if w > self.max_width or h > self.max_height:
            scale = min(self.max_width / w, self.max_height / h)
            new_w, new_h = int(w * scale), int(h * scale)
            img = transforms.functional.resize(img, (new_h, new_w), interpolation=self.interpolation)
        return img



class UnetSegmentationNode(Node):

    def __init__(self):
        super().__init__('unet_segmentation_node')

        self.declare_parameter('model_path', '/home/jorgen/deep/Pytorch-UNet/checkpoints/run_20251011_212141/best_model_epoch_15_val_0.9756.pth')
        self.declare_parameter('input_topic', '/image_color')
        self.declare_parameter('output_topic', '/image_masked')
        self.declare_parameter('resize_width', 320)
        self.declare_parameter('resize_height', 240)
        self.declare_parameter('mask_threshold', 0.5)
        self.declare_parameter('bilinear', False)
        self.declare_parameter('simple', True)
        self.declare_parameter('classes', 1)
        self.declare_parameter('device', 'cuda' if torch.cuda.is_available() else 'cpu')

        self.model_path = self.get_parameter('model_path').get_parameter_value().string_value
        self.input_topic = self.get_parameter('input_topic').get_parameter_value().string_value
        self.output_topic = self.get_parameter('output_topic').get_parameter_value().string_value
        self.mask_threshold = self.get_parameter('mask_threshold').get_parameter_value().double_value
        self.bilinear = self.get_parameter('bilinear').get_parameter_value().bool_value
        self.simple = self.get_parameter('simple').get_parameter_value().bool_value
        self.n_classes = self.get_parameter('classes').get_parameter_value().integer_value
        self.device_name = self.get_parameter('device').get_parameter_value().string_value
        self.resize_width = self.get_parameter('resize_width').get_parameter_value().integer_value
        self.resize_height = self.get_parameter('resize_height').get_parameter_value().integer_value

        
        if not self.model_path:
            self.get_logger().fatal("Parameter 'model_path' is not set. Please provide the absolute path to the model.")
            exit()

        self.device = torch.device(self.device_name)
        self.bridge = CvBridge()
        self.PRED_COLOR = (255, 0, 0) # Red for prediction

        self.net = self.load_model()

        self.image_transforms = transforms.Compose([
            ResizeIfLargerKeepAspect(
                max_width=self.resize_width,
                max_height=self.resize_height
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

        qos_profile = QoSProfile(
            reliability=QoSReliabilityPolicy.BEST_EFFORT,
            history=QoSHistoryPolicy.KEEP_LAST,
            depth=3
        )

        self.subscription = self.create_subscription(
            Image,
            self.input_topic,
            self.image_callback,
            qos_profile
        )
        self.publisher = self.create_publisher(Image, self.output_topic, qos_profile)
        
        self.get_logger().info(f"Node initialized. Subscribing to '{self.input_topic}' and publishing to '{self.output_topic}'.")

    def load_model(self):
        self.get_logger().info(f'Loading model from {self.model_path}')
        self.get_logger().info(f'Using device {self.device}')
        
        net = UNet(n_channels=3, n_classes=self.n_classes, bilinear=self.bilinear, simple=self.simple)
        net.to(device=self.device)
        
        try:
            state_dict = torch.load(self.model_path, map_location=self.device)
            _ = state_dict.pop('mask_values', None)
            net.load_state_dict(state_dict)
            self.get_logger().info('Model loaded!')
            return net
        except FileNotFoundError:
            self.get_logger().fatal(f"Model file not found at {self.model_path}. Please check the path.")
            exit()
            
    def image_callback(self, msg):
        try:
            # --- Convert ROS image to OpenCV ---
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(frame_rgb)

            # --- Resize input (keep aspect ratio, downscale only) ---
            resize_transform = ResizeIfLargerKeepAspect(
                max_width=self.resize_width,
                max_height=self.resize_height
            )
            resized_pil = resize_transform(pil_img)
            resized_w, resized_h = resized_pil.size

            # --- Preprocess for model ---
            image_tensor = self.image_transforms(resized_pil)

            # --- Run model prediction ---
            predicted_mask_np = predict_mask(
                net=self.net,
                image_tensor=image_tensor,
                device=self.device,
                out_threshold=self.mask_threshold
            )

            # --- Convert mask to color overlay (same size as model input) ---
            mask_rgb = np.zeros((resized_h, resized_w, 3), dtype=np.uint8)
            mask_rgb[predicted_mask_np == 1] = self.PRED_COLOR

            # Blend resized image + mask overlay
            base_np = np.array(resized_pil)
            blended_np = cv2.addWeighted(base_np, 1.0, mask_rgb, 0.4, 0)

            # --- Create side-by-side composite ---
            combined_np = np.concatenate((base_np, blended_np), axis=1)

            # --- Publish combined image ---
            output_msg = self.bridge.cv2_to_imgmsg(combined_np, "rgb8")
            output_msg.header = msg.header
            self.publisher.publish(output_msg)

            # Optional debug
            self.get_logger().debug(
                f"Published visualization {resized_w}x{resized_h}, "
                f"original {pil_img.width}x{pil_img.height}"
            )

        except Exception as e:
            self.get_logger().error(f'Failed to process image: {e}')



def main(args=None):
    rclpy.init(args=args)
    unet_node = UnetSegmentationNode()
    try:
        rclpy.spin(unet_node)
    except KeyboardInterrupt:
        pass
    finally:
        unet_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()