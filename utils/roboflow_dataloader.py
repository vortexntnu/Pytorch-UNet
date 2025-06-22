import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import functional as F
import os
from PIL import Image


class SemanticSegmentationDataset(Dataset):
    """Image (semantic) segmentation dataset."""

    def __init__(self, dir, image_transform=None, mask_transform=None):
        """
        Args:
            dir (string): Directory of the dataset containing the images + annotations.
        """
        self.dir = dir
        self.image_transform = image_transform
        self.mask_transform = mask_transform

        self.classes_csv_file = os.path.join(self.dir, "_classes.csv")
        try:
            with open(self.classes_csv_file, 'r') as fid:
                data = [l.strip().split(',') for i,l in enumerate(fid) if i !=0]
            self.mask_values = {x[0]:x[1] for x in data}
        except FileNotFoundError:
            print(f"Warning: _classes.csv not found in {self.dir}. Class labels will not be loaded.")
            self.mask_values = {}

        all_files = os.listdir(self.dir)

        image_file_names = sorted([f for f in all_files if f.lower().endswith('.jpg')])
        mask_file_names = sorted([f for f in all_files if f.lower().endswith('.png')])

        self.images = [os.path.join(self.dir, f) for f in image_file_names]
        self.masks = [os.path.join(self.dir, f) for f in mask_file_names]

        if len(self.images) != len(self.masks):
            raise ValueError(f"Number of images ({len(self.images)}) does not match number of masks ({len(self.masks)}) in directory {self.dir}.")
        if not self.images or not self.masks:
            raise ValueError(f"No images or masks found in directory {self.dir}. Please check the dataset structure.")
        if len(self.images) == 0:
            raise ValueError(f"No images found in directory {self.dir}. Please check the dataset structure.")
        if len(self.masks) == 0:
            raise ValueError(f"No masks found in directory {self.dir}. Please check the dataset structure.")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):

        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx]).convert("L")
        mask = torch.as_tensor(np.array(mask), dtype=torch.int64)

        if self.image_transform:
            image = self.image_transform(image)
        
        if self.mask_transform:
            mask = self.mask_transform(mask)
  
        return {'image': image, 'mask': mask}
    

