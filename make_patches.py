import numpy as np
import cv2
import os
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm
import random

def tile_label(xpoint, ypoint, width, height, psize, step, imwh):
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
    patches1, patches2, masks = [], [], []
    area_patch = psize*psize
    for x in xs:
        for y in ys:
            patch = label[x:x+psize, y:y+psize]
            idx = np.where(patch!=0)
            prop = len(idx[0])/area_patch
            if prop>=perc:
                patches1.append(im1[x:x+psize, y:y+psize, :])
                patches2.append(im2[x:x+psize, y:y+psize, :])
                masks.append(label[x:x+psize, y:y+psize])
    #print('len12', len(patches1), len(patches2))
    return patches1, patches2, masks

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

def save_patches(patches1, patches2, masks, lu, save_folder, id):
    class_agri = [2,3] #0
    class_nonrel = [6, 7, 8, 14] #2 ##unsure
    class_veg = [9]
    class_rel = [10, 11, 12, 13, 15] #1
     

    #--4 --12
    #print('lu', lu)

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
            mod_masks = [masks[i] for i in indices]

            patches1 = mod_patches1
            patches2 = mod_patches2
            masks = mod_masks        
    #print('bbbbbb')

    for i in range(0, len(patches1)):
        patch = patches1[i].astype(np.uint8)
        patch = Image.fromarray(patch)
        patch.save('{}/im1/{}_{}_v2_class{}.png'.format(save_folder, id[:-4], i, final_class))
        #print('iiiiiiiiiiiiiiiiiiiiiiiiii', id[:-4])

    for i in range(0, len(patches2)):
        patch = patches2[i].astype(np.uint8)
        patch = Image.fromarray(patch)
        patch.save('{}/im2/{}_{}_v2_class{}.png'.format(save_folder, id[:-4], i, final_class))        

    for i in range(0, len(masks)):
        patch = (masks[i]*255).astype(np.uint8)
        patch = Image.fromarray(patch)
        patch.save('{}/mask/{}_{}_v2_class{}.png'.format(save_folder, id[:-4], i, final_class)) 


def plotit(label, x, y, w, h):
    label255 = cv2.rectangle(label*255, (x, y), (x + w, y + h), (128, 128, 128), 2)
    plt.figure(figsize=(8, 6))
    plt.imshow(label255)
    plt.title("Image with Bounding Box")
    plt.axis('off')
    plt.show()

split='train'
train_ids = os.listdir('/home/mariapap/DATASETS/dataset_FP_v2/{}/im1/'.format(split))

class_names = ["background", "none", "agricultural field", "agricultural other", "coregistration/stitching", "incident angle", 
               "shadow", "cloud", "not relevant", "vegetation clearing", "groundworks", "construction works", "vehicle", "road works", "snow"]
psize=128
step=48
area_patch = psize*psize
imwh=512
class_ignore = [0, 1, 4, 5]

train_ids = ['233491_0.png', '233494_6.png', '233490_0.png']
train_ids = ['233493_2.png', '233493_1.png', '233495_1.png', '233595_3.png', '233595_6.png', '233595_16.png', '233494_7.png']
#train_ids = ['233495_2.png']
train_ids = ['233594_5.png']
sum1=0

for _, id in enumerate(tqdm(train_ids)):
    #print('aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa')
  #if id=='Gas-Connect-2024_2267ML_1.png':
    print(id)
    im1 = Image.open('/home/mariapap/DATASETS/dataset_FP_v2/{}/im1/{}'.format(split,id))
    im2 = Image.open('/home/mariapap/DATASETS/dataset_FP_v2/{}/im2/{}'.format(split,id))
    label = Image.open('/home/mariapap/DATASETS/dataset_FP_v2/{}/label/{}'.format(split,id))
    im1, im2, label = np.array(im1), np.array(im2), np.array(label)
    l_uni = np.unique(label)
    if len(l_uni)==2:
        lu = l_uni[1]
    else:
        lu = l_uni[0]
    print('lu', lu)
    if lu==1:
        sum1 = sum1+1
    if lu not in class_ignore:
        contours, _ = cv2.findContours(label, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Example: get the first contour and calculate its area
        contour = contours[0]  # Access the first contour (or loop through contours if multiple)
        area = cv2.contourArea(contour)
        idxn0 = np.where(label!=0)
        label[idxn0]=1
        #print('area', area, area_patch)
        x, y, w, h = cv2.boundingRect(contour)
        #print('xxxx', x,y,w,h)

        if w<=psize:
            diff = int((psize-w)/2.)
            x = x-diff
            w = psize+1
        if h<=psize:
            diff = int((psize-h)/2.)
            y = y-diff
            h = psize  +1


        if area>0:
            if area<=2000: #area_patch:

                #x, y, w, h = cv2.boundingRect(contour)
                #if w<=128 or h<=128:
                x, y, w, h = centered_patch(contour, psize)
                #print('x', 'y', x, y)
                patches1, patches2, masks = cut_patches(im1, im2, label, [x], [y], psize, 0.001)
                #plotit(label, x, y, w, h)          

            else:
                #print('eeeeeeeeeee')

                #x, y, w, h = cv2.boundingRect(contour)
                #print('x', 'y', x, y, w, h)
                xs, ys = tile_label(y, x, w, h, psize, step, imwh)
                #print('xy ys', xs, ys)
                patches1, patches2, masks = cut_patches(im1, im2, label, xs, ys, psize, 0.05)
                #plotit(label, x, y, w, h)

        #    print(x, y, h ,w) #h| w-

            if patches1 and patches2:
                #print('PPPPPPPPPPPPPPPPPPPPPPPPPPP')
                save_patches(patches1, patches2, masks, lu, './PATCHES/{}'.format(split), id)

    
#print('sum', sum1)

