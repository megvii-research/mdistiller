import os
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms
import numpy as np


data_folder = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "../../data/tiny-imagenet-200"
)


class ImageFolderInstance(datasets.ImageFolder):
    def __getitem__(self, index):
        path, target = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return img, target, index


class ImageFolderInstanceSample(ImageFolderInstance):
    """: Folder datasets which returns (img, label, index, contrast_index):
    """
    def __init__(self, folder, transform=None, target_transform=None,
                 is_sample=False, k=4096):
        super().__init__(folder, transform=transform)

        self.k = k
        self.is_sample = is_sample
        if self.is_sample:
            num_classes = 200
            num_samples = len(self.samples)
            label = np.zeros(num_samples, dtype=np.int32)
            for i in range(num_samples):
                img, target = self.samples[i]
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
        print('dataset initialized!')

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


def get_tinyimagenet_dataloader(batch_size, val_batch_size, num_workers):
    """Data Loader for tiny-imagenet"""
    train_transform = transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
    test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
    train_folder = os.path.join(data_folder, "train")
    test_folder = os.path.join(data_folder, "val")
    train_set = ImageFolderInstance(train_folder, transform=train_transform)
    num_data = len(train_set)
    test_set = datasets.ImageFolder(test_folder, transform=test_transform)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, num_workers=1
    )
    return train_loader, test_loader, num_data


def get_tinyimagenet_dataloader_sample(batch_size, val_batch_size, num_workers, k):
    """Data Loader for tiny-imagenet"""
    train_transform = transforms.Compose([
                transforms.RandomRotation(20),
                transforms.RandomHorizontalFlip(0.5),
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
    test_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize([0.4802, 0.4481, 0.3975], [0.2302, 0.2265, 0.2262]),
            ])
    train_folder = os.path.join(data_folder, "train")
    test_folder = os.path.join(data_folder, "val")
    train_set = ImageFolderInstanceSample(train_folder, transform=train_transform, is_sample=True, k=k)
    num_data = len(train_set)
    test_set = datasets.ImageFolder(test_folder, transform=test_transform)
    train_loader = DataLoader(
        train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_loader = DataLoader(
        test_set, batch_size=val_batch_size, shuffle=False, num_workers=1
    )
    return train_loader, test_loader, num_data
