import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import sys
import torch.nn.functional as F
import random
import  copy
import pickle
from torch.utils.data import Dataset

from  model import *
from dataset import *

###################################################################
#                              TEST
###################################################################

Test_model = AutoencoderNet()
param_path = './model_ckpt/Autoencoder/last_model.pth'
Test_model.load_state_dict(torch.load(param_path))

Test_model.eval()

with torch.no_grad():
    output_encoded = []
    output_decoded = []

    for i in range(8):
        input_data = [0] * 8
        input_data[i] = 1
        input_data = torch.Tensor(input_data)
        encoded, decoded = Test_model(input_data)
        output_encoded.append(encoded)
        output_decoded.append(decoded)

    n = nn.Softmax(dim=0)

    for i in range(8):
        print("======================== {}th raw data ========================".format(i))
        print("{}th encoded : {}".format(i,output_encoded[i]))
        print("{}th decoded : {}".format(i,n(output_decoded[i])))

    print("\n")
    for i in range(8):
        print("======================== {}th processed data ========================".format(i))
        print("{}th encoded : {}".format(i,torch.round(output_encoded[i])))
        print("{}th decoded : {}".format(i,torch.argmax(n(output_decoded[i]))))