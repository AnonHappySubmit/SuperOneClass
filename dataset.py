from torch.utils.data import Subset
from PIL import Image
from torchvision.datasets import CIFAR10
from torchvision import datasets
import torch
import numpy as np

import torch.utils.data as data
import torchvision.transforms as transforms
import pdb

def get_target_label_idx(labels, targets):
    """
    Get the indices of labels that are included in targets.
    :param labels: array of labels
    :param targets: list/tuple of target labels
    :return: list with indices of target labels
    """
    return np.argwhere(np.isin(labels, targets)).flatten().tolist()



class FMNIST_Dataset(data.Dataset):

    def __init__(self, root: str, normal_class=0):

        self.n_classes = 2  # 0: normal, 1: outlier
        if isinstance(normal_class, int):
            normal_class = [normal_class]
        self.normal_classes = tuple(normal_class)
        self.outlier_classes = list(set(range(0, 10)) - set(normal_class))
        print ('normal_class: ', self.normal_classes)
        print ('outlier_classes: ', self.outlier_classes)
        self.root = root

        transform = transforms.Compose([transforms.Resize(32), transforms.ToTensor(), transforms.Normalize([0.5], [0.5])]) 
        mnist_transform=transforms.Compose([transforms.Resize((32, 32)),
                                                 transforms.Grayscale(num_output_channels = 1),
                                        transforms.ToTensor(),
                                        transforms.Normalize([0.5], [0.5])]) 
        
        target_transform = transforms.Lambda(lambda x: int(x not in self.outlier_classes))

        train_set = FashionMNIST(root=self.root, train=True, download=True,
                            transform=transform, target_transform=target_transform)
        assert len(train_set) == 60000
        train_idx_normal = get_target_label_idx(train_set.train_labels.clone().data.cpu().numpy(), self.normal_classes)
        self.train_set = Subset(train_set, train_idx_normal)
        
        self.test_set = FashionMNIST(root=self.root, train=False, download=True,
                                transform=transform, target_transform=target_transform)
        assert len(self.test_set) == 10000
        

class FashionMNIST(datasets.FashionMNIST):
    """Torchvision MNIST class with patch of __getitem__ method to also return the index of a data sample."""

    def __init__(self, *args, **kwargs):
        super(FashionMNIST, self).__init__(*args, **kwargs)

    def __getitem__(self, index):
        """Override the original method of the MNIST class.
        Args:
            index (int): Index
        Returns:
            triple: (image, target, index) where target is index of the target class.
        """

        img, target = self.data[index], int(self.targets[index])

        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target  
