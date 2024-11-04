import numpy as np
from PIL import Image
import os
import cv2
import subprocess

split='train'
train_ids = os.listdir('/home/mariapap/DATASETS/dataset_FP_v0/dataset_FP/{}/im1/'.format(split))
#test_ids = os.listdir('/home/mariapap/DATASETS/dataset_FP_v0/dataset_FP/test/im1/')
#val_ids = os.listdir('/home/mariapap/DATASETS/dataset_FP_v0/dataset_FP/val/im1/')

print('train', len(train_ids))
#print('val', len(val_ids))
#print('test', len(test_ids))

class_names = ["background", "none", "agricultural field", "agricultural other", "coregistration/stitching", "incident angle", 
               "shadow", "cloud", "not relevant", "vegetation clearing", "groundworks", "construction works", "vehicle", "road works", "snow"]
class_list = [0] * 15
summ=0
for id in train_ids:
    label = Image.open('/home/mariapap/DATASETS/dataset_FP_v0/dataset_FP/{}/label/{}'.format(split,id))
    label = np.array(label)
    l_uni = np.unique(label)
    if len(l_uni)==2:
        lu = l_uni[1]
    else:
        lu = l_uni[0]

    class_list[lu] = class_list[lu] + 1

total_sum = sum(class_list)
print(class_list)

for i in range(0,15):
    print(class_names[i], ': ', "{:.2f}".format((class_list[i]/total_sum)*100))

    

