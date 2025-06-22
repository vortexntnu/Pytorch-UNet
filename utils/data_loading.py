import logging
import numpy as np
import torch
from PIL import Image
from functools import lru_cache
from functools import partial
from itertools import repeat
from multiprocessing import Pool
from PIL import Image, UnidentifiedImageError
from pathlib import Path
from torch.utils.data import Dataset
from tqdm import tqdm

from roboflow import Roboflow


def load_image(filename):
    ext = splitext(filename)[1]
    if ext == '.npy':
        return Image.fromarray(np.load(filename))
    elif ext in ['.pt', '.pth']:
        return Image.fromarray(torch.load(filename).numpy())
    else:
        return Image.open(filename)


def unique_mask_values(idx, mask_dir, mask_suffix):
    # Use glob with iter(). If it's empty, it won't yield anything.
    # Check if any file matches the pattern before trying to access [0]
    mask_files = list(mask_dir.glob(idx + mask_suffix + '.*'))
    if not mask_files:
        # This case should ideally not happen if self.ids is correctly populated
        # but it's a good defensive check.
        raise FileNotFoundError(f"Mask file not found for ID: {idx} with suffix: {mask_suffix} in {mask_dir}")
        
    mask_file_path = mask_files[0] # Get the first (and hopefully only) matching mask file
    
    try:
        mask = np.asarray(Image.open(mask_file_path))
    except UnidentifiedImageError as e:
        # Catch if a file that's thought to be an image can't be identified
        raise UnidentifiedImageError(f"Failed to identify mask image: {mask_file_path}. Original error: {e}")


    if mask.ndim == 2:
        # If it's a 2D (grayscale) mask, get unique values and reshape them into a 2D column vector.
        # Example: [0, 1, 255] becomes [[0], [1], [255]]
        return np.unique(mask).reshape(-1, 1)
    elif mask.ndim == 3:
        # If it's a 3D (RGB) mask, reshape to (num_pixels, channels) and get unique rows.
        # This already returns a 2D array (e.g., [[0,0,0], [255,0,0]])
        mask_reshaped = mask.reshape(-1, mask.shape[-1])
        return np.unique(mask_reshaped, axis=0)
    else:
        raise ValueError(f'Loaded masks should have 2 or 3 dimensions, found {mask.ndim} for {mask_file_path}')

class BasicDataset(Dataset):
    def __init__(self, images_dir: str, mask_dir: str, scale: float = 1.0, mask_suffix: str = ''):
        self.images_dir = Path(images_dir)
        self.mask_dir = Path(mask_dir)
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'
        self.scale = scale
        self.mask_suffix = mask_suffix

        # Define accepted image and mask extensions
        self.img_exts = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        self.mask_exts = ['.png'] # Masks are typically PNG for semantic segmentation

        # --- MODIFIED self.ids POPULATION ---
        # We will populate self.ids based on mask files, then verify image files.
        self.ids = []
        
        # Get all potential mask filenames (base_id + suffix + ext)
        # Iterate over files in the mask_dir and extract IDs
        for mask_file_path in self.mask_dir.iterdir():
            if mask_file_path.is_file() and not mask_file_path.name.startswith('.'):
                mask_name_no_ext, mask_ext = splitext(mask_file_path.name)
                
                # Check if it's a valid mask file based on suffix and extension
                if mask_ext.lower() in self.mask_exts and mask_name_no_ext.endswith(mask_suffix):
                    base_id = mask_name_no_ext.replace(mask_suffix, '')
                    
                    # Now check if a corresponding image file exists for this base_id
                    found_img_for_id = False
                    for img_ext in self.img_exts:
                        img_file_path = self.images_dir / (base_id + img_ext)
                        if img_file_path.is_file():
                            self.ids.append(base_id)
                            found_img_for_id = True
                            break # Found image for this mask, move to next mask
                    
                    if not found_img_for_id:
                        logging.warning(f"No corresponding image found for mask: {mask_file_path.name}. Skipping this pair.")
                elif mask_file_path.is_file() and mask_file_path.name not in ['_classes.csv', 'data.yaml']:
                    # Log if there are unexpected files in mask_dir that aren't masks or known metadata
                    # logging.warning(f"Skipping non-mask/non-image file in mask_dir: {mask_file_path.name}")
                    pass


        logging.info(f'Creating dataset with {len(self.ids)} examples')
        logging.info('Scanning mask files to determine unique values')
        with Pool() as p:
            unique = list(tqdm(
                p.imap(partial(unique_mask_values, mask_dir=self.mask_dir, mask_suffix=self.mask_suffix), self.ids),
                total=len(self.ids)
            ))
        concatenated_unique = np.concatenate(unique, axis=0)
        self.mask_values = list(sorted(map(tuple, np.unique(concatenated_unique, axis=0)))) # Convert to tuple for sorting consistent

        logging.info(f'Unique mask values: {self.mask_values}')

    def __len__(self):
        return len(self.ids)

    @staticmethod
    def preprocess(mask_values, pil_img, scale, is_mask):
        w, h = pil_img.size
        newW, newH = int(scale * w), int(scale * h)
        assert newW > 0 and newH > 0, 'Scale is too small, resized images would have no pixel'
        pil_img = pil_img.resize((newW, newH), resample=Image.NEAREST if is_mask else Image.BICUBIC)
        img = np.asarray(pil_img)

        if is_mask:
            mask = np.zeros((newH, newW), dtype=np.int64)
            for i, v in enumerate(mask_values):
                if img.ndim == 2:
                    mask[img == v] = i
                else:
                    mask[(img == v).all(-1)] = i

            return mask

        else:
            if img.ndim == 2:
                img = img[np.newaxis, ...]
            else:
                img = img.transpose((2, 0, 1))

            if (img > 1).any():
                img = img / 255.0

            return img

    def __getitem__(self, idx):
        name = self.ids[idx]
        mask_file = list(self.mask_dir.glob(name + self.mask_suffix + '.*'))
        img_file = list(self.images_dir.glob(name + '.*'))

        assert len(img_file) == 1, f'Either no image or multiple images found for the ID {name}: {img_file}'
        assert len(mask_file) == 1, f'Either no mask or multiple masks found for the ID {name}: {mask_file}'
        mask = Image.open(mask_file[0])
        img = Image.open(img_file[0])

        assert img.size == mask.size, \
            f'Image and mask {name} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess(self.mask_values, img, self.scale, is_mask=False)
        mask = self.preprocess(self.mask_values, mask, self.scale, is_mask=True)

        return {
            'image': torch.as_tensor(img.copy()).float().contiguous(),
            'mask': torch.as_tensor(mask.copy()).long().contiguous()
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, images_dir, mask_dir, scale=1):
        super().__init__(images_dir, mask_dir, scale, mask_suffix='_mask')
