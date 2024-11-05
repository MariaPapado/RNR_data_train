import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

def determine_label(lu):

    class_agri = [2,3] #0
    class_veg = [9]
    class_rel = [10, 11, 13] #1
    class_nonrel = [5, 6, 7, 8, 14] #2

    if lu in class_agri:
        final_class=0
    elif lu in class_nonrel:
        final_class=1
    elif lu in class_veg:
        final_class=2
    elif lu in class_rel:
        final_class=3

    return final_class

def tile_label(xpoint, ypoint, width, height, psize, step):
    img_x, img_y = [], []
    #print('xpoint', xpoint, height)
    for x in range(xpoint, xpoint+height, step):
        if x+psize<xpoint+height:
            #print('xxxx', x, 'height', xpoint+height)
            img_x.append(x)
        else:
            #img_x.append(xpoint+height-psize)
            #print('aaaaaaaa', xpoint+height-psize)
            break
    for y in range(ypoint, ypoint+width, step):
        if y+psize<ypoint+width:
            img_y.append(y)
        else:
            #img_y.append(ypoint+width-psize)
            break

    return img_x, img_y


def cut_patches(im1, im2, label, xs, ys, psize, perc):
    patches1, patches2 = [], []
    area_patch = psize*psize
    for x in xs:
        for y in ys:
            patch = label[x:x+psize, y:y+psize]
            idx = np.where(patch!=0)
            prop = len(idx[0])/area_patch
            if prop>=perc:
                patches1.append(im1[x:x+psize, y:y+psize, :])
                patches2.append(im2[x:x+psize, y:y+psize, :])

    return patches1, patches2

def centered_patch(contour, psize):
    M = cv2.moments(contour)
    if M["m00"] != 0:
        center_x = int(M["m10"] / M["m00"])
        center_y = int(M["m01"] / M["m00"])
        
        # Calculate the top-left corner for the rectangle
        x = center_x - psize // 2
        y = center_y - psize // 2


    return int(x), int(y), psize, psize           

def is_black_image(image):
    return np.all(image == 0)

def save_patches(patches1, patches2, lu, save_folder, id):
    class_agri = [2,3] #0
    class_veg = [9]
    class_rel = [10, 11, 13] #1
    class_nonrel = [5, 6, 7, 8, 14] #2

    #--4 --12

    if lu in class_agri:
        final_class=0
    elif lu in class_nonrel:
        final_class=1
    elif lu in class_veg:
        final_class=2
    elif lu in class_rel:
        final_class=3

    if lu in class_agri:
        length = len(patches1)
        if length>2:
            indices = random.sample(range(length), 2)
            mod_patches1 = [patches1[i] for i in indices]
            mod_patches2 = [patches2[i] for i in indices]

            patches1 = mod_patches1
            patches2 = mod_patches2        

    for i in range(0, len(patches1)):
        patch = patches1[i].astype(np.uint8)
        patch = Image.fromarray(patch)
        patch.save('{}/im1/{}_{}_class{}.png'.format(save_folder, id[:-6], i, final_class))
    for i in range(0, len(patches2)):
        patch = patches2[i].astype(np.uint8)
        patch = Image.fromarray(patch)
        patch.save('{}/im2/{}_{}_class{}.png'.format(save_folder, id[:-6], i, final_class))        


def plotit(label, x, y, w, h):
    label255 = cv2.rectangle(label*255, (x, y), (x + w, y + h), (128, 128, 128), 2)
    plt.figure(figsize=(8, 6))
    plt.imshow(label255)
    plt.title("Image with Bounding Box")
    plt.axis('off')
    plt.show()
