import os
from data.base_dataset import BaseDataset, get_params, get_transform
from data.image_folder import make_dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import torch


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
        self.input_nc = self.opt.output_nc if self.opt.direction == 'BtoA' else self.opt.input_nc
        self.output_nc = self.opt.input_nc if self.opt.direction == 'BtoA' else self.opt.output_nc

        color_maps = {(0, 0, 0): 0, (255, 0, 0): 1, (0, 255, 0): 2, (0, 0, 255): 3}
        self.color_keys = np.array(list(color_maps.keys()))
        self.color_values = np.array(list(color_maps.values()))

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
        AB = Image.open(AB_path).convert('RGB')
        # split AB image into A and B
        w, h = AB.size
        w2 = int(w / 2)
        A = AB.crop((0, 0, w2, h))
        B = AB.crop((w2, 0, w, h))

        # remap B to 1 channel
        mask = np.array(B)
        reshaped_mask = mask.reshape(-1, 3)
        indices = np.where((reshaped_mask[:, None] == self.color_keys).all(-1))[1]
        single_channel_mask = self.color_values[indices].reshape(mask.shape[:2])
        single_channel_mask = single_channel_mask.astype(np.uint8)
        B = Image.fromarray(single_channel_mask)
        # remap A to 1 channel
        A = A.convert('L')

        # apply the same transform to both A and B
        transform_params = get_params(self.opt, A.size)
        A_transform = get_transform(self.opt, transform_params, grayscale=(self.input_nc == 1), method=transforms.InterpolationMode.BILINEAR, is_mask=False)
        B_transform = get_transform(self.opt, transform_params, grayscale=(self.output_nc == 1), method=transforms.InterpolationMode.NEAREST, is_mask=True)

        A = A_transform(A)
        B = B_transform(B)
        B = torch.tensor(np.array(B)).unsqueeze(0).float()

        return {'A': A, 'B': B, 'A_paths': AB_path, 'B_paths': AB_path}

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.AB_paths)
