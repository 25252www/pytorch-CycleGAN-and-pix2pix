import os
import cv2
import numpy as np
import torch
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
from PIL import Image


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
        self.dir_AB = os.path.join(opt.dataroot, opt.phase)  # get the image directory
        self.AB_paths = sorted(make_dataset(self.dir_AB, opt.max_dataset_size))  # get image paths
        assert(self.opt.load_size >= self.opt.crop_size)   # crop_size should be smaller than the size of loaded image
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
        # read a image given a random integer index
        AB_path = self.AB_paths[index]
        AB = cv2.imread(AB_path,-1)  # IMREAD_UNCHANGED = -1#不进行转化，比如保存为了32位的图片，读取出来仍然为32位。
        # split AB image into A and B
        w = np.shape(AB)[1]
        w2 = int(w / 2)
        A = AB[:,:w2,:]
        B = AB[:,w2:,:]
        # (900,900,3)(height,weight,channel)->(3,900,900)(channel,height,weight)
        A = np.swapaxes(A,0,2)
        A = np.swapaxes(A,1,2)
        B = np.swapaxes(B,0,2)
        B = np.swapaxes(B,1,2)

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, (np.shape(A)[1],np.shape(A)[0]))
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1))
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1))

        A = A_transform(torch.from_numpy(A))
        B = B_transform(torch.from_numpy(B))

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
