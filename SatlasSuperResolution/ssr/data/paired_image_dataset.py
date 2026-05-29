import os
import glob
import torch
import random
import torchvision
import numpy as np
from torch.utils import data as data

from basicsr.utils.registry import DATASET_REGISTRY

@DATASET_REGISTRY.register()
class SimpleImagePairDataset(data.Dataset):
    """
    Simple dataset for paired input/ground_truth images.
    
    Args:
        opt (dict): Config for datasets. It contains the following keys:
            input_path (str): Data path for input images (low-res).
            gt_path (str): Data path for ground truth images (high-res).
            phase (str): 'train', 'val', or 'test'.
            io_backend (dict): io backend for loading images.
    """

    def __init__(self, opt):
        super(SimpleImagePairDataset, self).__init__()
        self.opt = opt
        self.phase = opt['phase'] if 'phase' in opt else 'train'
        
        # Get image paths
        self.input_path = opt['input_path']
        self.gt_path = opt['gt_path']
        
        if not (os.path.exists(self.input_path) and os.path.exists(self.gt_path)):
            raise Exception(f"Please make sure the paths are correct. input: {self.input_path}, gt: {self.gt_path}")
        
        # Scan for image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.image_files.extend(glob.glob(os.path.join(self.gt_path, ext)))
            self.image_files.extend(glob.glob(os.path.join(self.gt_path, ext.upper())))
        
        self.image_files = list(set(self.image_files))  # Remove duplicates
        self.image_files.sort()
        
        if len(self.image_files) == 0:
            raise Exception(f"No images found in {self.gt_path}")
        
        print(f"Found {len(self.image_files)} image pairs for phase '{self.phase}'")

    def __getitem__(self, index):
        gt_path = self.image_files[index]
        filename = os.path.basename(gt_path)
        input_path = os.path.join(self.input_path, filename)
        
        # Load images
        import cv2
        hr = cv2.imread(gt_path, cv2.IMREAD_COLOR)  # High-res (ground truth)
        lr = cv2.imread(input_path, cv2.IMREAD_COLOR)  # Low-res (input)
        
        if hr is None or lr is None:
            raise Exception(f"Failed to load images: hr={gt_path}, lr={input_path}")
        
        # Convert BGR to RGB
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        
        # Convert to tensors
        hr = torch.from_numpy(np.transpose(hr, (2, 0, 1)))
        lr = torch.from_numpy(np.transpose(lr, (2, 0, 1)))
        
        return {
            'lr': lr,
            'hr': hr,
            'lr_path': input_path,
            'hr_path': gt_path
        }

    def __len__(self):
        return len(self.image_files)
