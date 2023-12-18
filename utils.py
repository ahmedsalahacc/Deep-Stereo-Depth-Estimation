"""Util functions depth estimation notebook"""
import os

from typing import List, Tuple

import matplotlib.pyplot as plt

import glob

import numpy as np

from torch.utils.data import Dataset, DataLoader

import torch

import cv2


class StereoDataset(Dataset):
    """Stereo dataset class"""

    def __init__(
        self, imgs_path: str, depth_path: str, img_size: int = 128, transform=None
    ) -> None:
        """Constructor of the Stereo dataset class

        Parameters:
        -----------
            imgs_path: str
                path to images
            depth_path: str
                path to depth images
        """
        self.dataset = self._get_imgs_from_dataset(imgs_path, depth_path)
        self.transform = transform
        self.max_depth = 65280.0
        self.img_size = img_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor]:
        """Gets item from the dataset

        Parameters:
        -----------
            idx: int
                index of the item to get

        Returns:
        --------
            Tuple[torch.Tensor]:
                tuple of left image, right image, left depth image, right depth image
        """
        (
            left_img_path,
            right_imgs_path,
            left_depth_path,
            right_depth_path,
        ) = self.dataset[idx]

        left_img = cv2.imread(left_img_path)
        right_img = cv2.imread(right_imgs_path)
        left_depth = cv2.imread(
            left_depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
        )
        right_depth = cv2.imread(
            right_depth_path, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH
        )

        # downsample images to 256x256
        left_img = cv2.resize(left_img, (self.img_size, self.img_size))
        right_img = cv2.resize(right_img, (self.img_size, self.img_size))
        left_depth = cv2.resize(left_depth, (self.img_size, self.img_size))
        right_depth = cv2.resize(right_depth, (self.img_size, self.img_size))

        # apply transform if any
        if self.transform:
            self.transform([left_img, right_img, left_depth, right_depth])

        # permute image to have channel dimension first as pytorch convention
        left_img = torch.from_numpy(left_img).permute(2, 0, 1).float() / 255
        right_img = torch.from_numpy(right_img).permute(2, 0, 1).float() / 255

        # unsqueeze depth image to include channel dimension
        left_depth = torch.from_numpy(left_depth).unsqueeze(0).float() / self.max_depth
        right_depth = (
            torch.from_numpy(right_depth).unsqueeze(0).float() / self.max_depth
        )

        return left_img, right_img, left_depth, right_depth

    def __len__(self) -> int:
        """Returns the length of the dataset"""
        return len(self.dataset)

    def _get_imgs_from_dataset(self, imgs_path: str, depth_path: str) -> List[str]:
        """Gets images path from datasets and return them as list

        Parameters:
        -----------
            imgs_path: str
                path to images
            depth_path: str
                path to depth images

        Returns:
        --------
            List[str]:
                list of images paths
        """

        def strip_name_from_path(path: str) -> str:
            """Strips name from path

            Parameters:
            -----------
                path: str
                    path to image

            Returns:
            --------
                str:
                    name of image
            """
            path = path.split("/")[-1]
            path = path.split(".")[0]
            return path[: -len("image")]

        # get image from depth based on the name in the imgs_path
        left_imgs = sorted(glob.glob(os.path.join(imgs_path, "*l-image.png")))
        right_imgs = sorted(glob.glob(os.path.join(imgs_path, "*r-image.png")))
        left_depth_imgs = sorted(glob.glob(os.path.join(depth_path, "*l-depth.exr")))
        right_depth_imgs = sorted(glob.glob(os.path.join(depth_path, "*r-depth.exr")))

        results = []

        return [
            (i, j, k, l)
            for i, j, k, l in zip(
                left_imgs, right_imgs, left_depth_imgs, right_depth_imgs
            )
        ]


def get_train_valid_test_loaders(
    dataset: Dataset,
    batchsize=32,
    split=(0.8, 0.15, 0.05),
) -> Tuple[DataLoader]:
    """Gets the dataloaders from training, testing and validation according to the train_valid_test_split"""
    train_size, valid_size, test_size = split

    train_size = int(train_size * len(dataset))
    valid_size = int(valid_size * len(dataset))
    test_size = int(test_size * len(dataset))

    # samplers
    train_sampler = torch.utils.data.SubsetRandomSampler(range(train_size))
    valid_sampler = torch.utils.data.SubsetRandomSampler(
        range(train_size, train_size + valid_size)
    )
    test_sampler = torch.utils.data.SubsetRandomSampler(
        range(train_size + valid_size, train_size + valid_size + test_size)
    )

    trainloader = DataLoader(dataset, batch_size=batchsize, sampler=train_sampler)
    validloader = DataLoader(dataset, batch_size=batchsize, sampler=valid_sampler)
    testloader = DataLoader(dataset, batch_size=batchsize, sampler=test_sampler)

    return trainloader, validloader, testloader


def highest_divisor(num: int) -> int:
    """Gets the highest divisor of a number

    Parameters:
    -----------
        num: int
            number to get highest divisor from

    Returns:
    --------
        int:
            highest divisor
    """
    for i in range(int(num**0.5), 0, -1):
        if num % i == 0:
            return i

    return 1


def display_batch(batch: torch.Tensor) -> None:
    """Displays image in grid from batch

    Parameters:
    -----------
        batch: torch.Tensor
            batch of images
    """
    # ensure batchsize is square
    bs = batch.shape[0]
    # convert images to numpy and display using matplotlib
    batch = batch.permute(0, 2, 3, 1).detach().numpy()

    highest_divisor_num = highest_divisor(bs)

    rows = int(bs / highest_divisor_num)
    cols = highest_divisor_num

    fig, ax = plt.subplots(rows, cols, figsize=(20, 20))

    # display images
    for i in range(rows):
        for j in range(cols):
            ax[i, j].imshow(batch[i * cols + j])
            ax[i, j].axis("off")

    # mininize space between subplots
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    x = torch.randn(32, 3, 128, 128)
    display_batch(x)
