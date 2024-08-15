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


dataset_path = '/home/mariapap/DATASETS/RNR/RNR_data/'

with open("/home/mariapap/DATASETS/RNR/RNR_data/lists/train_list.txt", "r") as file:
    train_ids = [line.strip() for line in file.readlines()]
#(train_ids)

with open("/home/mariapap/DATASETS/RNR/RNR_data/lists/val_list.txt", "r") as file:
    val_ids = [line.strip() for line in file.readlines()] 

trainset = dataloader.RNR_data(dataset_path, train_ids, 'train')
valset = dataloader.RNR_data(dataset_path, val_ids, 'val')

trainloader = DataLoader(trainset, batch_size=8, shuffle=True, drop_last=False)
valloader = DataLoader(valset, batch_size=8, shuffle=True, drop_last=False)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

UseCuda=True
NumClasses = 2
Trained_model_path="" 
Net=Net.Net(NumClasses=NumClasses,UseGPU=UseCuda) # Create main resnet image classification brach
Net.AddAttententionLayer()
Net.to(device)

#rgb_img = np.random.randint(0,255, size=(1,128,128,3))
#mask = np.random.randint(0, 2, size=(1,128,128))
#out = Net.forward(rgb_img,ROI=mask)
#print(out[0].shape)
#print(out[1].shape)
#print(out[0])
#print(out[1])
#print(Net)

classes = ('NR', 'R')

weight_tensor=torch.FloatTensor(2)
weight_tensor[0]= 0.30
weight_tensor[1]= 0.70
weight_tensor.to(device)


criterion = nn.CrossEntropyLoss(weight_tensor)
criterion.to(device)
optimizer = torch.optim.Adam(Net.parameters(), lr=0.00001)
#scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)
epochs = 15

c_matrix = np.zeros((NumClasses, NumClasses), dtype=int)
iter_ = 0

save_folder = 'saved_models'
if os.path.exists(save_folder):
    shutil.rmtree(save_folder)

# Create the directory
os.mkdir(save_folder)

for epoch in range(0, epochs):
    Net.train()
    train_loss = []
    correct = 0
    total = 0
    for i, batch in enumerate(tqdm(trainloader)):
        img, mask, label, data_idx = batch

        img, mask, label = img.to(device), mask.to(device), label.to(device)
        optimizer.zero_grad()
        prob, pred = Net(img, mask)
        loss = criterion(prob, label)
        loss.backward()
        optimizer.step()

        train_loss.append(loss.item())

        c_matrix = c_matrix + confusion_matrix(label.data.cpu().numpy(), pred.data.cpu().numpy(), labels=np.arange(NumClasses))

        iter_ += 1

        if iter_ % 20 == 0:
            print('Train (epoch {}/{}) [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                      epoch, epochs, i, len(trainloader),100.*i/len(trainloader), loss.item()))

    print('TRAIN_c_matric')
    print(c_matrix)
    print('TRAIN_mean_loss: ', np.mean(train_loss)) 



    Net.eval()    
    c_matrix = np.zeros((NumClasses, NumClasses), dtype=int)
    with torch.no_grad():
        val_loss = []
        for i, batch in enumerate(tqdm(valloader)):
            img, mask, label, data_idx = batch
            img, mask, label = img.to(device), mask.to(device), label.to(device)

            prob, pred = Net(img, mask)
            loss = criterion(prob, label)

            val_loss.append(loss.item())

            c_matrix = c_matrix + confusion_matrix(label.data.cpu().numpy(), pred.data.cpu().numpy(), labels=np.arange(NumClasses))


    print('VAL_c_matric')
    print(c_matrix)
    print('VAL_mean_loss: ', np.mean(val_loss)) 

    torch.save(Net.state_dict(), './' + save_folder + '/net_{}.pt'.format(epoch))

        #imutils.progress_bar(data_idx, len(trainloader), 'Loss: %.3f | Acc: %.3f%% (%d/%d)'
        #             % (train_loss/(data_idx+1), 100.*correct/total, correct, total))

        

        #img = img[0].permute(1,2,0).numpy()
        #img = np.array(img, dtype=np.uint8)
        #img = Image.fromarray(img)
        #img.save('./check/img/{}'.format(data_idx[0]))

        #mask = mask[0].numpy()*255
        #mask = np.array(mask, dtype=np.uint8)
        #mask = Image.fromarray(mask)
        #mask.save('./check/mask/{}'.format(data_idx[0]))    


