import os
import random


#2663
train_ids = os.listdir('./PATCHES/train/im1/')
with open("./PATCHES/lists/train_list.txt", "w") as file:
    for item in train_ids:
        file.write(f"{item}\n")

val_ids = os.listdir('./PATCHES/val/im1/')
with open("./PATCHES/lists/val_list.txt", "w") as file:
    for item in val_ids:
        file.write(f"{item}\n")

