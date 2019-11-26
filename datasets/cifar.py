from __future__ import print_function
from PIL import Image
import torchvision.datasets as datasets
import torch.utils.data as data
import numpy as np



class CIFAR10Instance(datasets.CIFAR10):
    """CIFAR10Instance Dataset.
    """
    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], self.targets[index]
        else:
            img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index


class SVHNInstance(datasets.SVHN):
    """CIFAR10Instance Dataset.
    """
    def __init__(self, root, train=True, transform=None, target_trainsform=None, download=False):
        self.train = train
        super(SVHNInstance, self).__init__(root, split=('train' if train else 'test'),transform=transform, target_transform=target_trainsform, download=download)

    def __getitem__(self, index):
        if self.train:
            img, target = self.data[index], int(self.labels[index])
        else:
            img, target = self.data[index], int(self.labels[index])
        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(np.transpose(img, (1, 2, 0)))

        if self.target_transform is not None:
            target = self.target_transform(target)

        if self.transform is not None:
            img = self.transform(img)

        return img, target, index
