import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
import math
import sys
import pdb
import sys
import torch.nn.functional as F
import random
import  copy
import pickle
from torch.utils.data import Dataset

from  model import *
from dataset import *

####################################################################
#                          Hyperparameters
###################################################################
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
num_epochs = 300
batch_size = 50
learning_rate = 0.001


# ####################################################################
# #                         Prepare Data
# ###################################################################
with open('data.pickle','rb') as f:
    file_list = pickle.load(f)

# train dataset
train_dataset = one_byte_dataset(file_list)

# train dataloader
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,batch_size=batch_size,shuffle=True)
print(len(train_loader))

# test dataset
random.shuffle(file_list)
test_list = file_list[0:20]
test_dataset = one_byte_dataset(test_list)
test_loader = torch.utils.data.DataLoader(dataset=test_dataset,batch_size=batch_size,shuffle=True)


####################################################################
#                         Define Model
####################################################################
# Autoencoder Model
Autoencoder = AutoencoderNet().to(device)
Encoder_save_path = './model_ckpt/Autoencoder'
if not os.path.exists(Encoder_save_path):
    os.makedirs(Encoder_save_path)

# optimizer and criterion
optimizer_Aut = torch.optim.Adam(Autoencoder.parameters(), lr = learning_rate)
criterion = nn.CrossEntropyLoss()

####################################################################
#                   TRAIN Model (Encoder & Decoder)
####################################################################
for epoch in range(0, num_epochs):
    Autoencoder.train()
    
    epoch_loss = 0

    for i, data in enumerate(train_loader):
        input_data = data[0].to(device)
        target = data[1].to(device)
        # pdb.set_trace()
        ___, output = Autoencoder(input_data)   
        loss = criterion(output,target)
        optimizer_Aut.zero_grad()
        loss.backward()
        optimizer_Aut.step()

        temp_loss = loss.item()
        epoch_loss += temp_loss

    average_loss = float(epoch_loss/batch_size)
    print('Epoch [{}/{}], Step [{}/{}], G_Loss: {:.4f}'.format(epoch+1, num_epochs, i+1, len(train_loader), average_loss))
    torch.save(Autoencoder.state_dict(),Encoder_save_path+'/last_model.pth')

print("train done!")
