"""
Code adopted from pix2pixHD:
https://github.com/NVIDIA/pix2pixHD/blob/master/data/image_folder.py
"""
import os
import torch
from torch.utils.data import Dataset

class MultiResolutiontrainData(Dataset):

    def __init__(self, X_data, y_data): 
        """
        x_data : list(features)
        y_data : sketch
        """
        
        X_data_ = []
        for x_data in X_data:
            tmp = [torch.from_numpy(x)[0] for x in x_data]
            X_data_.append(tmp)
        self.X_data = X_data_ # List( List (Torch.tensor))
        self.y_data = y_data # Torch.tensor: shape (h,w)

        #print("[dataloader] y_data shape: ", self.y_data[0].shape)
        
        assert len(self.X_data) == len(self.y_data), "dataset len does not match"

    def __getitem__(self, index):
        return self.X_data[index], self.y_data[index]

    def __len__(self):
        return len(self.X_data)

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir
    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images