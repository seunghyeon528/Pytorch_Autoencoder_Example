# -*- coding: utf-8 -*-
import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import math
import random
import random
import  copy
import pickle

file_list = list()

for i in range(1000):
    temp_list = [1,0,0,0,0,0,0,0]    
    file_list.append(temp_list)

for i in range(1000):
    temp_list = [0,1,0,0,0,0,0,0]    
    file_list.append(temp_list)

for i in range(1000):
    temp_list = [0,0,1,0,0,0,0,0]    
    file_list.append(temp_list)

for i in range(1000):
    temp_list = [0,0,0,1,0,0,0,0]    
    file_list.append(temp_list)

for i in range(1000):
    temp_list = [0,0,0,0,1,0,0,0]    
    file_list.append(temp_list)

for i in range(1000):
    temp_list = [0,0,0,0,0,1,0,0]    
    file_list.append(temp_list)

for i in range(1000):
    temp_list = [0,0,0,0,0,0,1,0]    
    file_list.append(temp_list)

for i in range(1000):
    temp_list = [0,0,0,0,0,0,0,1]    
    file_list.append(temp_list)

print(len(file_list))

# pickle file 로 list 저장
with open('data.pickle','wb') as f:
    pickle.dump(file_list,f)

