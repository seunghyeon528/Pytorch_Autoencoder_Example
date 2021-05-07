import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import librosa
import torchaudio
import os
import math
import random
import sys
from tqdm import tqdm
import pdb
import sys
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as plt
from torchsummary import summary
import torch.nn.functional as F
import random
import  copy
import pickle
from torch.utils.data import Dataset


class one_byte_dataset(Dataset):

    def __init__(self,filelist):
        self.filelist = filelist

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self,index):
        data = []
        input_data = self.filelist[index]
        data.append(torch.Tensor(input_data))
        data.append(input_data.index(1))
        return data

