import torch
import numpy as np
import att_net as Net
import os
import dataloader
from tqdm import tqdm
from torch.utils.data import DataLoader
from PIL import Image
import torch.nn as nn
import imutils
from sklearn.metrics import confusion_matrix
import shutil



preds_folder = 'PREDS'
if os.path.exists(preds_folder):
    shutil.rmtree(preds_folder)

# Create the directory
os.mkdir(preds_folder)
os.mkdir('./PREDS/wrong')
os.mkdir('./PREDS/right')


dataset_path = '/home/mariapap/DATASETS/RNR/RNR_data/'

with open("/home/mariapap/DATASETS/RNR/RNR_data/lists/test_list.txt", "r") as file:
    test_ids = [line.strip() for line in file.readlines()]


testset = dataloader.RNR_data(dataset_path, test_ids, 'test')

testloader = DataLoader(testset, batch_size=1, shuffle=False, drop_last=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

UseCuda=True
NumClasses = 2
Trained_model_path="" 
Net=Net.Net(NumClasses=NumClasses,UseGPU=UseCuda) # Create main resnet image classification brach
Net.AddAttententionLayer()

Net.load_state_dict(torch.load('./saved_models/net_8.pt'))
Net.eval()
Net.to(device)

c_matrix = np.zeros((NumClasses, NumClasses), dtype=int)


for i, batch in enumerate(tqdm(testloader)):            
    img, mask, label, data_idx = batch
    #print(data_idx)
    img, mask, label = img.to(device), mask.to(device), label.to(device)

    prob, pred = Net(img, mask)

    c_matrix = c_matrix + confusion_matrix(label.data.cpu().numpy(), pred.data.cpu().numpy(), labels=np.arange(NumClasses))

    img_save = Image.open('/home/mariapap/DATASETS/RNR/RNR_data/images/{}'.format(data_idx[0]))

    if label!=pred:
        img_save.save('./PREDS/wrong/{}'.format(data_idx[0][:-4] + '_' + str(int(pred[0].data.cpu().numpy())) + '.png'))
    else:
        img_save.save('./PREDS/right/{}'.format(data_idx[0][:-4] + '_' + str(int(pred[0].data.cpu().numpy())) + '.png'))

        
        

    

print('TEST_c_matrix')
print(c_matrix)