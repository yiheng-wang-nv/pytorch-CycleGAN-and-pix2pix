import os
from data.base_dataset import BaseDataset, get_transform
from data.image_folder import make_dataset
import cv2
import random
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
import torch
import numpy as np

def rotate(image_with_masks, angle):
    """
    Perform a random rotation transform on an image with masks.
    
    Parameters:
    image_with_masks (numpy.ndarray): The input array with shape (h, w, 3).
    angle (float): The rotation angle in degrees.
    
    Returns:
    numpy.ndarray: The rotated array with shape (h, w, 3).
    """
    # Convert the numpy array to a PyTorch tensor
    image_with_masks = torch.from_numpy(image_with_masks)
    
    # Separate the image and masks
    image = image_with_masks[..., 0]
    mask1 = image_with_masks[..., 1]
    mask2 = image_with_masks[..., 2]
    
    # Apply the rotation with bilinear interpolation to the image
    rotated_image = F.rotate(image.unsqueeze(0), angle, interpolation=transforms.InterpolationMode.BILINEAR).squeeze(0)
    
    # Apply the rotation with nearest interpolation to the masks
    rotated_mask1 = F.rotate(mask1.unsqueeze(0), angle, interpolation=transforms.InterpolationMode.NEAREST).squeeze(0)
    rotated_mask2 = F.rotate(mask2.unsqueeze(0), angle, interpolation=transforms.InterpolationMode.NEAREST).squeeze(0)
    
    # Combine the rotated image and masks back into a single array
    rotated_image_with_masks = torch.stack((rotated_image, rotated_mask1, rotated_mask2), dim=-1)
    
    return rotated_image_with_masks

def get_mask_index_range(image):
    """
    Get the index range in dimensions 0 and 1 that have non-zero values in the mask channel.
    
    Parameters:
    image (torch.Tensor): The image with shape (h, w, 3) and type uint16.
    
    Returns:
    tuple: The start and end indices for dimensions 0 and 1.
    """
    # Extract the mask channel
    mask = image[..., 1]
    mask = mask / 65535.0
    
    # Find the indices of the non-zero values in the mask
    non_zero_indices = torch.nonzero(mask > 0)
    
    # Determine the minimum and maximum indices along dimensions 0 and 1
    min_indices = non_zero_indices.min(dim=0)[0]
    max_indices = non_zero_indices.max(dim=0)[0] + 1  # Add 1 to include the max index
    
    # Get the start and end indices for dimensions 0 and 1
    start_index_dim0, start_index_dim1 = min_indices.tolist()
    end_index_dim0, end_index_dim1 = max_indices.tolist()
    
    return (start_index_dim0, end_index_dim0), (start_index_dim1, end_index_dim1)

def random_crop(image, range_dim0, range_dim1, debug=False):
    """
    Perform a random crop based on the specified range and scales, ensuring coverage.
    
    Parameters:
    image (torch.Tensor): The image with shape (h, w, 3) and type uint16.
    range_dim0 (tuple): The start and end indices for dimension 0.
    range_dim1 (tuple): The start and end indices for dimension 1.
    
    Returns:
    torch.Tensor: The cropped image.
    """
    start_index_dim0, end_index_dim0 = range_dim0
    start_index_dim1, end_index_dim1 = range_dim1
    
    start_dim0 = np.random.randint(start_index_dim0 // 2, start_index_dim0)
    end_dim0 = np.random.randint(end_index_dim0, end_index_dim0 + (image.shape[0] - end_index_dim0) // 2)

    start_dim1 = np.random.randint(start_index_dim1 // 2, start_index_dim1)
    end_dim1 = np.random.randint(end_index_dim1, end_index_dim1 + (image.shape[1] - end_index_dim1) // 2)
    
    if debug:
        start_dim0 = start_index_dim0 // 2
        end_dim0 = end_index_dim0 + (image.shape[0] - end_index_dim0) // 2
        start_dim1 = start_index_dim1 // 2
        end_dim1 = end_index_dim1 + (image.shape[1] - end_index_dim1) // 2
        # start_dim0 = start_index_dim0
        # end_dim0 = end_index_dim0
        # start_dim1 = start_index_dim1
        # end_dim1 = end_index_dim1
    # Perform the crop
    cropped_image = image[start_dim0:end_dim0, start_dim1:end_dim1]
    
    return cropped_image

def create_fan_mask(mask, center_offset):
    """
    Create a fan-shaped mask for the given image shape and mask.
    
    Parameters:
    mask (numpy.ndarray): The mask array with shape (h, w).
    center_offset (int): The vertical offset of the fan center from the top of the image.
    
    Returns:
    numpy.ndarray: The fan-shaped mask with the same shape as the input image.
    """
    h, w = mask.shape[:2]
    center_y = -center_offset
    center_x = w // 2
    
    # Create a grid of coordinates
    Y, X = np.ogrid[:h, :w]
    
    # Calculate the angle of each pixel relative to the center
    angles = np.arctan2(Y - center_y, X - center_x) * 180 / np.pi
    # Calculate the distance of each pixel from the center
    distances = np.sqrt((Y - center_y)**2 + (X - center_x)**2)
    
    top_boundary = 50
    bottom_boundary = h - 25
    
    left_angle = 60
    right_angle = 180 - left_angle
    # Create the fan-shaped mask
    fan_mask = np.zeros((h, w), dtype=np.uint8)
    fan_mask[(angles >= left_angle) & (angles <= right_angle) & 
             (distances >= top_boundary) & (distances <= bottom_boundary)] = 1
    
    return torch.from_numpy(fan_mask).unsqueeze(-1).expand(-1, -1, mask.shape[-1])

import torch
import torchvision.transforms.functional as F

def resize_image_and_masks(tensor, size):
    """
    Resize an image and masks in a tensor.
    
    Parameters:
    tensor (torch.Tensor): The input tensor with shape (h, w, 3) and dtype float32.
    size (tuple): The desired output size (height, width).
    
    Returns:
    torch.Tensor: The resized tensor with shape (size[0], size[1], 3).
    """
    # Separate the image and masks
    image = tensor[..., 0]
    mask1 = tensor[..., 1]
    mask2 = tensor[..., 2]
    
    # Resize the image with bilinear interpolation
    resized_image = F.resize(image.unsqueeze(0), size, interpolation=F.InterpolationMode.BILINEAR).squeeze(0)
    
    # Resize the masks with nearest interpolation
    resized_mask1 = F.resize(mask1.unsqueeze(0), size, interpolation=F.InterpolationMode.NEAREST).squeeze(0)
    resized_mask2 = F.resize(mask2.unsqueeze(0), size, interpolation=F.InterpolationMode.NEAREST).squeeze(0)
    
    # Combine the resized image and masks back into a single tensor
    resized_tensor = torch.stack((resized_image, resized_mask1, resized_mask2), dim=-1)
    
    return resized_tensor


class UnalignedDataset(BaseDataset):
    """
    This dataset class can load unaligned/unpaired datasets.

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
        self.dir_A = os.path.join(opt.dataroot, opt.phase + 'A')  # create a path '/path/to/data/trainA'
        self.dir_B = os.path.join(opt.dataroot, opt.phase + 'B')  # create a path '/path/to/data/trainB'

        self.A_paths = sorted(make_dataset(self.dir_A, opt.max_dataset_size))   # load images from '/path/to/data/trainA'
        self.B_paths = sorted(make_dataset(self.dir_B, opt.max_dataset_size))    # load images from '/path/to/data/trainB'
        self.A_size = len(self.A_paths)  # get the size of dataset A
        self.B_size = len(self.B_paths)  # get the size of dataset B

        self.color_jitter = transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2)
        self.size = opt.load_size

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
        A_path = self.A_paths[index % self.A_size]  # make sure index is within then range
        if self.opt.serial_batches:   # make sure index is within then range
            index_B = index % self.B_size
        else:   # randomize the index for domain B to avoid fixed pairs.
            index_B = random.randint(0, self.B_size - 1)
        B_path = self.B_paths[index_B]
        A_img = cv2.imread(A_path, cv2.IMREAD_UNCHANGED)
        B_img = cv2.imread(B_path)
        if "rotate" in self.opt.preprocess:
            # random rotate A_img
            angle = random.uniform(-15, 15)
            A_img = rotate(A_img, angle)

            # random rotate B_img
            angle = random.uniform(-15, 15)
            B_img = rotate(B_img, angle)

        if "crop" in self.opt.preprocess:
            # random crop A_img
            range_dim0, range_dim1 = get_mask_index_range(A_img)
            A_img = random_crop(A_img, range_dim0, range_dim1)

        A_img = A_img * 1.0
        if "resize" in self.opt.preprocess:
            # resize A_img
            A_img = resize_image_and_masks(A_img, (self.size, self.size))
            # resize B_img
            B_img = resize_image_and_masks(B_img, (self.size, self.size))

        fan_mask = create_fan_mask(A_img, 0)
        A_img = A_img * fan_mask
        A_img = A_img[..., :1]
        B_img = B_img[..., :1]

        A_img = A_img / 65535.0
        B_img = B_img / 255.0

        # apply image transformation
        A = torch.permute(A_img, (2, 0, 1))
        B = torch.permute(B_img, (2, 0, 1))

        return {'A': A, 'B': B, 'A_paths': A_path, 'B_paths': B_path}

    def __len__(self):
        """Return the total number of images in the dataset.

        As we have two datasets with potentially different number of images,
        we take a maximum of
        """
        return max(self.A_size, self.B_size)
