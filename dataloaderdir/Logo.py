import numpy as np
from PIL import Image
import os
import io
import json
import glob
import h5py

from torchmeta.utils.data import Dataset, ClassDataset, CombinationMetaDataset
from torchvision.datasets.utils import download_url
from torchmeta.datasets.utils import get_asset
from torchmeta.transforms import ClassSplitter, Categorical, Rotation
from torchvision.transforms import ToTensor, Resize, Compose
from torchmeta.utils.data import BatchMetaDataLoader
from tqdm import tqdm
import pickle
import torch

class Logo(CombinationMetaDataset):

    def __init__(self, root, num_classes_per_task=None, meta_train=False,
                 meta_val=False, meta_test=False, meta_split=None,
                 transform=None, target_transform=None, dataset_transform=None,
                 class_augmentations=None, download=False, folder = None):
                
        dataset = LogoClassDataset(root, meta_train=meta_train, meta_val=meta_val,
            meta_test=meta_test, meta_split=meta_split, transform=transform,
            class_augmentations=class_augmentations, download=download, folder = folder)
        super(Logo, self).__init__(dataset, num_classes_per_task,
            target_transform=target_transform, dataset_transform=dataset_transform)


class LogoClassDataset(ClassDataset):
    filename_labels = '{0}_labels.json'
    def __init__(self, root, meta_train=False, meta_val=False, meta_test=False,
                 meta_split=None, transform=None, class_augmentations=None,
                 download=False, folder = None):
        super(LogoClassDataset, self).__init__(meta_train=meta_train,
            meta_val=meta_val, meta_test=meta_test, meta_split=meta_split,
            class_augmentations=class_augmentations)


        self.folder = folder
        self.root = os.path.join(os.path.expanduser(root), self.folder)
        self.transform = transform
  
        self.split_filename_labels = os.path.join(self.root,
            self.filename_labels.format(self.meta_split))

        self._data_file = None
        self._data = None
        self._labels = None   
        self._num_classes = len(self.labels)

        #print('self.meta_split', self.meta_split)
    def __getitem__(self, index):
        label = self.labels[index % self.num_classes]
        transform = self.get_transform(index, self.transform)
        target_transform = self.get_target_transform(index)
        class_dict = torch.load(self.root + '/'+self.meta_split  + '/' + '{}.pt'.format(label))
        data = class_dict[label]
        
        return LogoDataset(index, data, label, transform=transform,
                          target_transform=target_transform)

    @property
    def num_classes(self):
        return self._num_classes

    @property
    def labels(self):
        if self._labels is None:
            with open(self.split_filename_labels, 'r') as f:
                self._labels = json.load(f)
        return self._labels


class LogoDataset(Dataset):
    def __init__(self, index, data, label,
                 transform=None, target_transform=None):
        super(LogoDataset, self).__init__(index, transform=transform,
                                         target_transform=target_transform)
        self.data = data
        self.label = label

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):

        image = self.data[index]
        target = self.label
        if self.target_transform is not None:
            target = self.target_transform(target)
        return (image, target)





