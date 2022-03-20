import os
import numpy as np
import torch
from torchvision.datasets import ImageFolder
import torchvision.transforms as transforms


data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../data/imagenet')


class ImageNet(ImageFolder):
    def __getitem__(self, index):
        img, target = super().__getitem__(index)
        return img, target, index


class ImageNetInstanceSample(ImageNet):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """
    def __init__(self, folder, transform=None, target_transform=None,
                 is_sample=False, k=4096):
        super().__init__(folder, transform=transform)

        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            print('preparing contrastive data...')
            num_classes = 1000
            num_samples = len(self.samples)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                _, target = self.samples[i]
                label[i] = target

            self.cls_positive = [[] for i in range(num_classes)]
            for i in range(num_samples):
                self.cls_positive[label[i]].append(i)

            self.cls_negative = [[] for i in range(num_classes)]
            for i in range(num_classes):
                for j in range(num_classes):
                    if j == i:
                        continue
                    self.cls_negative[i].extend(self.cls_positive[j])

            self.cls_positive = [np.asarray(self.cls_positive[i], dtype=np.int32) for i in range(num_classes)]
            self.cls_negative = [np.asarray(self.cls_negative[i], dtype=np.int32) for i in range(num_classes)]
            print('done.')

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is class_index of the target class.
        """
        img, target, index = super().__getitem__(index)

        if self.is_sample:
            # sample contrastive examples
            pos_idx = index
            neg_idx = np.random.choice(self.cls_negative[target], self.k, replace=True)
            sample_idx = np.hstack((np.asarray([pos_idx]), neg_idx))
            return img, target, index, sample_idx
        else:
            return img, target, index

def get_imagenet_train_transform(mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return train_transform

def get_imagenet_test_transform(mean, std):
    normalize = transforms.Normalize(mean=mean, std=std)
    test_transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]
    )
    return test_transform

def get_imagenet_dataloaders(batch_size, val_batch_size, num_workers,
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    train_transform = get_imagenet_train_transform(mean, std)
    train_folder = os.path.join(data_folder, 'train')
    train_set = ImageNet(train_folder, transform=train_transform)
    num_data = len(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = get_imagenet_val_loader(val_batch_size, mean, std)
    return train_loader, test_loader, num_data

def get_imagenet_dataloaders_sample(batch_size, val_batch_size, num_workers, k=4096, 
    mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    train_transform = get_imagenet_train_transform(mean, std)
    train_folder = os.path.join(data_folder, 'train')
    train_set = ImageNetInstanceSample(train_folder, transform=train_transform, is_sample=True, k=k)
    num_data = len(train_set)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, 
        shuffle=True, num_workers=num_workers, pin_memory=True)
    test_loader = get_imagenet_val_loader(val_batch_size, mean, std)
    return train_loader, test_loader, num_data

def get_imagenet_val_loader(val_batch_size, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    test_transform = get_imagenet_test_transform(mean, std)
    test_folder = os.path.join(data_folder, 'val')
    test_set = ImageFolder(test_folder, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set,
        batch_size=val_batch_size, shuffle=False, num_workers=16, pin_memory=True)
    return test_loader
