import os.path
import numpy as np
import torch
import json
import re

from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
from PIL import Image
import random
import util.util as util


class AlignedDataset(BaseDataset):
    """
    This dataset class can load aligned/paired datasets.

    It requires two directories to host training images from domain A '/path/to/data/trainA'
    and from domain B '/path/to/data/trainB' respectively.
    You can train the model with the dataset flag '--dataroot /path/to/data'.
    Similarly, you need to prepare two directories:
    '/path/to/data/testA' and '/path/to/data/testB' during test time.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # self.dir_A = os.path.join(opt.dataroot, "trainA")  # create a path '/path/to/data/trainA'
        # self.dir_B = os.path.join(opt.dataroot, "trainB")  # create a path '/path/to/data/trainB'
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        if opt.phase == "test" and not os.path.exists(self.dir_A) \
           and os.path.exists(os.path.join(opt.dataroot, "valA")):
            self.dir_A = os.path.join(opt.dataroot, "valA")
            self.dir_B = os.path.join(opt.dataroot, "valB")

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'

        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B
        assert self.A_size == self.B_size
        
        # Extract labels from filenames
        self.labels = []
        for A_path in self.A_paths:
            label = self._extract_label_from_path(A_path)
            self.labels.append(label)
    
    def _extract_label_from_path(self, path):
        """Extract label from filename and convert to continuous value.
        File format: e.g., '00000_test_1+.png'
        Labels: 0 -> 0.0, 1+ -> 0.33, 2+ -> 0.66, 3+ -> 1.0
        """
        filename = os.path.basename(path)
        # Extract label part (e.g., '0', '1+', '2+', '3+') from filename
        # Pattern: ends with number or number+ before .png
        match = re.search(r'_(\d\+?)(?:\.|$)', filename)
        if match:
            label_str = match.group(1)
            if label_str == '0':
                return 0.0
            elif label_str == '1+':
                return 0.33
            elif label_str == '2+':
                return 0.66
            elif label_str == '3+':
                return 1.0
            else:
                # Try to parse as integer (fallback)
                try:
                    label_int = int(label_str)
                    return float(label_int) / 3.0  # Normalize to [0, 1]
                except ValueError:
                    return 0.0  # Default value
        return 0.0  # Default value if no match

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index (int)      -- a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor)       -- an image in the input domain
            B (tensor)       -- its corresponding image in the target domain
            A_paths (str)    -- image paths
            B_paths (str)    -- image paths
        """
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index = random.randint(0, self.A_size - 1)
            index_B = index % self.B_size
            
        A_path = self.A_paths[index]  # make sure index is within then range
        B_path = self.B_paths[index_B]

        assert A_path == B_path.replace('trainB', 'trainA').replace('valB', 'valA').replace('testB', 'testA')

        A_img = Image.open(A_path).convert('RGB')
        B_img = Image.open(B_path).convert('RGB')

        # Apply image transformation
        # For CUT/FastCUT mode, if in finetuning phase (learning rate is decaying),
        # do not perform resize-crop data augmentation of CycleGAN.
        is_finetuning = self.opt.isTrain and self.current_epoch > self.opt.n_epochs
        modified_opt = util.copyconf(self.opt, load_size=self.opt.crop_size if is_finetuning else self.opt.load_size)
        transform = get_transform(modified_opt)

        # FDL: synchronize transforms
        seed = np.random.randint(2147483647) # make a seed with numpy generator 
        random.seed(seed) # apply this seed to img tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        A = transform(A_img)
        random.seed(seed) # apply this seed to target tranfsorms
        torch.manual_seed(seed) # needed for torchvision 0.7
        B = transform(B_img)

        # Get label for this sample
        label = self.labels[index]
        
        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path, 'label': label}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)