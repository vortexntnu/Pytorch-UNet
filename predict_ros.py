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


class UnetSegmentationNode(Node):

    def __init__(self):
        super().__init__('unet_segmentation_node')

        self.declare_parameter('model_path', '/home/jorgen/deep/Pytorch-UNet/checkpoints/run_20250622_194023/model.pth')
        self.declare_parameter('input_topic', '/gripper_camera/image_raw')
        self.declare_parameter('output_topic', '/image_masked')
        self.declare_parameter('mask_threshold', 0.5)
        self.declare_parameter('bilinear', False)
        self.declare_parameter('simple', False)
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
        
        if not self.model_path:
            self.get_logger().fatal("Parameter 'model_path' is not set. Please provide the absolute path to the model.")
            exit()

        self.device = torch.device(self.device_name)
        self.bridge = CvBridge()
        self.PRED_COLOR = (255, 0, 0) # Red for prediction

        self.net = self.load_model()
        
        self.image_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
        """
        Callback function for the image subscriber.
        """
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            frame_rgb = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            pil_img = PILImage.fromarray(frame_rgb)
            
            image_tensor = self.image_transforms(pil_img)
            
            predicted_mask_np = predict_mask(
                net=self.net,
                image_tensor=image_tensor,
                device=self.device,
                out_threshold=self.mask_threshold
            )
            
            blended_img = blend_image_and_mask(pil_img, predicted_mask_np, color=self.PRED_COLOR)
            
            blended_frame_rgb = np.array(blended_img)
            
            output_msg = self.bridge.cv2_to_imgmsg(blended_frame_rgb, "rgb8")
            output_msg.header = msg.header
            self.publisher.publish(output_msg)

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