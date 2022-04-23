import os
import cv2
import numpy as np
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image
import gdal


class AlignedDataset(BaseDataset):
    """A dataset class for paired image dataset.

    It assumes that the directory '/path/to/data/train' contains image pairs in the form of {A,B}.
    During test time, you need to prepare a directory '/path/to/data/test'.
    """

    def __init__(self, opt):
        """Initialize this dataset class.

        Parameters:
            opt (Option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions
        """
        BaseDataset.__init__(self, opt)
        # get the image directory
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)
        self.AB_paths = sorted(make_dataset(
            self.dir_AB, opt.max_dataset_size))  # get image paths
        # crop_size should be smaller than the size of loaded image
        assert(self.opt.load_size >= self.opt.crop_size)
        # input_nc (int)  -- the number of channels in input images
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

    def __getitem__(self, index):
        """Return a data point and its metadata information.

        Parameters:
            index - - a random integer for data indexing

        Returns a dictionary that contains A, B, A_paths and B_paths
            A (tensor) - - an image in the input domain
            B (tensor) - - its corresponding image in the target domain
            A_paths (str) - - image paths
            B_paths (str) - - image paths (same as A_paths)
        """
        # gdal读AB中的fiff,分割成AB
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        # gdal读取tiff,分割(源码使用Image读取并分割),最后传入 transfomer.Compose()
        AB = gdal.Open(AB_path)
        AB = AB.ReadAsArray().astype(np.float32)  # shape AB (3, 900, 1800)chw
        w = np.shape(AB)[2]
        w2 = int(w / 2)
        A = AB[:, :, :w2]  # (3, 900, 900)chw
        B = AB[:, :, w2:]  # (3, 900, 900)chw
        A = np.swapaxes(A,0,2)
        A = np.swapaxes(A,0,1)
        B = np.swapaxes(B,0,2)
        B = np.swapaxes(B,0,1)  # hwc

        # apply the same transform to both A and B
        transform_params = get_params(
            self.opt, (np.shape(A)[1], np.shape(A)[0]))
        A_transform = get_transform(
            self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(
            self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(Image.fromarray(np.uint8(A)))
        B = B_transform(Image.fromarray(np.uint8(B)))

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
