import torch
from torch.utils.data import Dataset
from PIL import Image


class SingleImageDataset(Dataset):
    """Dataset for single image input"""
    def __init__(self, image_paths, labels, transform=None, return_paths=False):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform
        self.return_paths = return_paths

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        if self.return_paths:
            return image, label, img_path
        else:
            return image, label


class MultiImageDataset(Dataset):
    """Dataset for multi image input (pair of images from trainA and trainB)"""
    def __init__(self, image_pairs, labels, transform=None, return_paths=False):
        self.image_pairs = image_pairs  # List of (pathA, pathB) tuples
        self.labels = labels
        self.transform = transform
        self.return_paths = return_paths

    def __len__(self):
        return len(self.image_pairs)

    def __getitem__(self, idx):
        pathA, pathB = self.image_pairs[idx]
        imageA = Image.open(pathA).convert('RGB')
        imageB = Image.open(pathB).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            imageA = self.transform(imageA)
            imageB = self.transform(imageB)

        # Concatenate images along channel dimension
        image = torch.cat([imageA, imageB], dim=0)

        if self.return_paths:
            return image, label, (pathA, pathB)
        else:
            return image, label
