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


    def __transforms(self, aug, img1, img2):
        #if aug:
        #    img1, img2, mask = imutils.random_fliplr(img1, img2, mask)
        #    img1, img2, mask = imutils.random_flipud(img1, img2, mask)
        #    img1, img2, mask = imutils.random_rot(img1, img2, mask)

        img1 = imutils.normalize_img(img1)  # imagenet normalization
        img2 = imutils.normalize_img(img2)  # imagenet normalization

        img1 = np.transpose(img1, (2, 0, 1))
        img2 = np.transpose(img2, (2, 0, 1))

        return img1, img2

    def __getitem__(self, index):
            
        img1 = Image.open(os.path.join(self.dataset_path, self.type, 'im1', self.data_list[index]))
        img2 = Image.open(os.path.join(self.dataset_path, self.type, 'im2', self.data_list[index]))

        img1, img2 = np.array(img1), np.array(img2)

        #class_int = int(self.data_list[index][-5])
        #if class_int==0:
        #    label=0
        #else:
        #    label=1
        label = int(self.data_list[index][-5])


#        if 'train' in self.type:
#            img1, img2, mask = self.__transforms(True, img1, img2, mask)
#        else:
        img1, img2 = self.__transforms(False, img1, img2)

        img = np.concatenate((img1, img2), 0)

        data_idx = self.data_list[index]
        return img, label, data_idx
#        return np.array(img2, dtype=float), np.array(mask, dtype=float), label, data_idx

    def __len__(self):
        return len(self.data_list)

