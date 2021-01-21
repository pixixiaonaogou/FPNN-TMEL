import torch
from torchvision import models
import os
import albumentations
from torch.nn import functional as F
import numpy as np
import os
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tnrange, _tqdm_notebook
from scipy import ndimage
import cv2
from sklearn.utils import class_weight, shuffle
from torch.utils import data
from torch.utils.data import DataLoader
import torch
from torchvision import transforms
import matplotlib.pyplot as plt

original_height = 128
original_width  = 128

#easy_path = './train_easy/'
#easy_img_list = os.listdir(easy_path)
#print(len(easy_img_list))

from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomContrast,
    RandomGamma,
    RandomBrightness,
    ShiftScaleRotate
)

aug = Compose(
    [
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        ShiftScaleRotate(shift_limit=0.0625,scale_limit=0.5,rotate_limit=45,p=0.5),
        GridDistortion(p=0.5),
       # RandomRotate90(p=0.5),
           # ], p=0.8),
       # RandomContrast(p=0.5),
       # RandomBrightness(p=0.5),
      #  RandomGamma(p=0.5),
    ],
    p=0.5)

def augment_flips_color(p=.5):
    return  Compose([
        VerticalFlip(p=0.5),
        HorizontalFlip(p=0.5),
        #GridDistortion(p=0.5),
        ShiftScaleRotate(shift_limit=0.0625, scale_limit=0.50, rotate_limit=10, p=0.5),
#        RandomRotate90(p=0.5),
       # OneOf([,
            #ElasticTransform(p=0.5, alpha=120, sigma=120 * 0.05, alpha_affine=120 * 0.03),
            #GridDistortion(p=0.5),
           # OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5)
           # ], p=0.8),
#        RandomContrast(p=0.8),
#        RandomBrightness(p=0.8),
#        RandomGamma(p=0.8),
        ],p=p)

def load_image(path,shape,is_mask=False):
    #print(path)
    img = cv2.imread(path)

    if not is_mask:
        img = img
    else:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img , (shape[0],shape[1]))
    return img

class SkinDataset(data.Dataset):
    def __init__(self,image_dir,mask_dir,file_list,shape,is_test=False,num_class=1,is_mask=False):
        self.is_test = is_test
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.file_list = file_list
        self.shape = shape
        self.num_class = num_class
        self.is_mask = is_mask
        
    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
      #  if index not in range(0, len(self.file_list)):
      #      return self.__getitem__(np.random.randint(0,self.__len__()))

        file_id = self.file_list[index]
        image_path = os.path.join(self.image_dir,file_id)
        mask_name = file_id.replace(').png',')_mask.png')
        #mask_name = file_id.replace('.jpg','_segmentation.png')
        mask_path = os.path.join(self.mask_dir,mask_name)
        image = load_image(image_path,self.shape,is_mask=False)
        mask = load_image(mask_path,self.shape,is_mask=True)

        # augmented = augment_flips_color()
        if not self.is_test:
            augmented =aug(image=image,mask=mask)
            image = augmented['image']
            mask = augmented['mask']

        if self.is_mask:
            image_pred = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image_pred = np.expand_dims(image_pred, -1)
            image_pred = torch.from_numpy(np.transpose(image_pred, (2,0,1)).astype('float32')/255)
        
        image = torch.from_numpy(np.transpose(image, (2,0,1)).astype('float32')/255)
        if self.num_class == 1:
            mask = np.expand_dims(mask, -1)
            mask = torch.from_numpy(np.transpose(mask, (2, 0, 1)).astype('float32') / 255)

        if self.is_mask:
            return image, image_pred, mask
        else:
            return image, mask

def imshow(tensor, title=None):

    unloder = transforms.ToPILImage()
    plt.ion()

    image = tensor.cpu().clone()
    image = unloder(image)
    plt.imshow(image)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)


def run_check_dataset():
    train_image_dir = './prediction_train_224_224_NFN/'
    train_mask_dir = './train_384_gt/'
    train_file_list = os.listdir(train_image_dir)
    shape = (224,224)
    skindataset = SkinDataset(image_dir=train_image_dir,
                              mask_dir=train_mask_dir,
                              file_list=train_file_list,
                              shape=shape,is_test=False)

    batch_szie = 8
    number_workers = 2
    dataloader = DataLoader(
        dataset=skindataset,
        batch_size=batch_szie,
        num_workers = number_workers,
        pin_memory=True,
        shuffle=True
    )
    for index, (image,mask) in enumerate(dataloader):
        print(index)
        print(image.shape)
        print(mask.shape)

import os
import torch
import numpy as np
from glob import glob
from torch.utils.data import Dataset
import h5py
import itertools
from torch.utils.data.sampler import Sampler

class TwoStreamBatchSampler(Sampler):
    """Iterate two sets of indices
    An 'epoch' is one iteration through the primary indices.
    During the epoch, the secondary indices are iterated through
    as many times as needed.
    """
    def __init__(self, primary_indices, secondary_indices, batch_size, secondary_batch_size):
        self.primary_indices = primary_indices
        self.secondary_indices = secondary_indices
        self.secondary_batch_size = secondary_batch_size
        self.primary_batch_size = batch_size - secondary_batch_size

        assert len(self.primary_indices) >= self.primary_batch_size > 0
        assert len(self.secondary_indices) >= self.secondary_batch_size > 0

    def __iter__(self):
        primary_iter = iterate_once(self.primary_indices)
        secondary_iter = iterate_eternally(self.secondary_indices)
        return (
            primary_batch + secondary_batch
            for (primary_batch, secondary_batch)
            in zip(grouper(primary_iter, self.primary_batch_size),
                    grouper(secondary_iter, self.secondary_batch_size))
        )

    def __len__(self):
        return len(self.primary_indices) // self.primary_batch_size

def iterate_once(iterable):
    return np.random.permutation(iterable)


def iterate_eternally(indices):
    def infinite_shuffles():
        while True:
            yield np.random.permutation(indices)
    return itertools.chain.from_iterable(infinite_shuffles())


def grouper(iterable, n):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3) --> ABC DEF"
    args = [iter(iterable)] * n
    return zip(*args)


