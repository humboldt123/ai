"""
I use two datasets here:

Textures:
G. J. Burghouts and J. M. Geusebroek, Material-specific adaptation of color invariant features,
Pattern Recognition Letters, vol. 30, 306-313, 2009.


MNIST:
The mnist dataset idk
"""

import glob
import os

import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset


class MnistDataset(Dataset):
    r"""
    Nothing special here. Just a simple dataset class for mnist images.
    Created a dataset class rather using torchvision to allow
    replacement with any other image dataset
    """
    def __init__(self, split, im_path="data/mnist_png4/", im_ext='png'):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.im_ext = im_ext
        self.images, self.labels = self.load_images(im_path)

    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        :param im_path:
        :return:
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        labels = []

        # Check if images are in subdirectories (labeled) or flat directory
        subdirs = [d for d in os.listdir(im_path) if os.path.isdir(os.path.join(im_path, d))]

        if subdirs:
            # Images organized in subdirectories by label
            for d_name in tqdm(subdirs):
                for fname in glob.glob(os.path.join(im_path, d_name, '*.{}'.format(self.im_ext))):
                    ims.append(fname)
                    labels.append(int(d_name))
        else:
            # Flat directory - no labels needed for diffusion
            for fname in tqdm(glob.glob(os.path.join(im_path, '*.{}'.format(self.im_ext)))):
                ims.append(fname)
                labels.append(0)  # Dummy label

        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im = Image.open(self.images[index])
        im_tensor = torchvision.transforms.ToTensor()(im)

        # Convert input to -1 to 1 range.
        im_tensor = (2 * im_tensor) - 1
        return im_tensor


class AlotDataset(Dataset):
    r"""
    Dataset class for ALOT texture images.
    ALOT has 250 texture categories in subdirectories.
    """
    def __init__(self, split, im_path="data/alot_png4/", im_ext='png'):
        r"""
        Init method for initializing the dataset properties
        :param split: train/test to locate the image files
        :param im_path: root folder of images
        :param im_ext: image extension. assumes all
        images would be this type.
        """
        self.split = split
        self.im_ext = im_ext
        self.images, self.labels = self.load_images(im_path)

    def load_images(self, im_path):
        r"""
        Gets all images from the path specified
        and stacks them all up
        :param im_path:
        :return:
        """
        assert os.path.exists(im_path), "images path {} does not exist".format(im_path)
        ims = []
        labels = []
        for d_name in tqdm(os.listdir(im_path)):
            d_path = os.path.join(im_path, d_name)
            if not os.path.isdir(d_path):
                continue
            for fname in glob.glob(os.path.join(d_path, '*.{}'.format(self.im_ext))):
                ims.append(fname)
                labels.append(int(d_name))
        print('Found {} images for split {}'.format(len(ims), self.split))
        return ims, labels

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        im = Image.open(self.images[index])

        # Resize and center crop to ensure consistent dimensions
        # ALOT images have varying sizes, so we need to standardize them
        transform = torchvision.transforms.Compose([
            torchvision.transforms.Resize(64),  # Resize shortest edge to 64
            torchvision.transforms.CenterCrop(64),  # Center crop to 64x64
            torchvision.transforms.ToTensor(),
        ])
        im_tensor = transform(im)

        # Convert input to -1 to 1 range.
        im_tensor = (2 * im_tensor) - 1
        return im_tensor