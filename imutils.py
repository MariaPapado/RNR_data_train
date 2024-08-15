import random
import numpy as np
from PIL import Image
# from scipy import misc
import torch
import torchvision
import torchvision.transforms.functional as F
import time
from PIL import ImageEnhance


def normalize_img(img, mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375]):   #imgnet
#def normalize_img(img, mean=[96.812, 101.919, 92.033], std=[60.064, 57.084, 56.213]):   #NatGridFuel train/test

    """Normalize image by subtracting mean and dividing by std."""
    img_array = np.asarray(img, dtype=np.uint8)
    normalized_img = np.empty_like(img_array, np.float32)

    for i in range(3):  # Loop over color channels
        normalized_img[..., i] = (img_array[..., i] - mean[i]) / std[i]
    
    return normalized_img

def random_fliplr(img, mask):
    if random.random() > 0.5:
        img = np.fliplr(img)
        mask = np.fliplr(mask)

    return img, mask


def random_flipud(img, mask):
    if random.random() > 0.5:
        img = np.flipud(img)
        mask = np.flipud(mask)

    return img, mask


def random_rot(img, mask):
    k = random.randrange(3) + 1

    img = np.rot90(img, k).copy()
    mask = np.rot90(mask, k).copy()

    return img, mask



TOTAL_BAR_LENGTH = 65.
last_time = time.time()
begin_time = last_time
def progress_bar(current, total, msg=None):
    global last_time, begin_time
    if current == 0:
        begin_time = time.time()  # Reset for new bar.

    cur_len = int(TOTAL_BAR_LENGTH*current/total)
    rest_len = int(TOTAL_BAR_LENGTH - cur_len) - 1

    sys.stdout.write(' [')
    for i in range(cur_len):
        sys.stdout.write('=')
    sys.stdout.write('>')
    for i in range(rest_len):
        sys.stdout.write('.')
    sys.stdout.write(']')

    cur_time = time.time()
    step_time = cur_time - last_time
    last_time = cur_time
    tot_time = cur_time - begin_time

    L = []
    L.append('  Step: %s' % format_time(step_time))
    L.append(' | Tot: %s' % format_time(tot_time))
    if msg:
        L.append(' | ' + msg)

    msg = ''.join(L)
    sys.stdout.write(msg)
    for i in range(term_width-int(TOTAL_BAR_LENGTH)-len(msg)-3):
        sys.stdout.write(' ')

    # Go back to the center of the bar.
    for i in range(term_width-int(TOTAL_BAR_LENGTH/2)+2):
        sys.stdout.write('\b')
    sys.stdout.write(' %d/%d ' % (current+1, total))

    if current < total-1:
        sys.stdout.write('\r')
    else:
        sys.stdout.write('\n')
    sys.stdout.flush()