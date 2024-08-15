import torch
import numpy as np
import imutils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import os
from PIL import Image


class RNR_data(Dataset):
    def __init__(self, dataset_path, data_list, type='train'):
        self.dataset_path = dataset_path
        self.data_list = data_list
        
        self.type = type


    def __transforms(self, aug, img, mask):
        if aug:
            img, mask = imutils.random_fliplr(img, mask)
            img, mask = imutils.random_flipud(img, mask)
            img, mask = imutils.random_rot(img, mask)

        #img = imutils.normalize_img(img)  # imagenet normalization
        img = np.transpose(img, (2, 0, 1))

        return img, mask

    def __getitem__(self, index):
        img_path = os.path.join(self.dataset_path, 'images', self.data_list[index])
        mask_path = os.path.join(self.dataset_path, 'masks', self.data_list[index])

        #print('img', img_path)
        #print('mask', mask_path)
        img = np.array(Image.open(img_path))
        img = np.array(img, dtype=float)
        mask = np.array(Image.open(mask_path))
        mask = np.array(mask, dtype=float)

        if '_0' in self.data_list[index]:
            label=0
        else:
            label=1

        #label = label / 255

        if 'train' in self.type:
            img, mask = self.__transforms(True, img, mask)
        else:
            img, mask = self.__transforms(False, img, mask)

        data_idx = self.data_list[index]
        return np.array(img, dtype=float), np.array(mask, dtype=float), label, data_idx

    def __len__(self):
        return len(self.data_list)

