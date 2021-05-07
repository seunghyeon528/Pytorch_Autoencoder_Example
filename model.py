import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
import os
import sys

import torch.nn.functional as F

""" 
metric = correlation
"""
class AutoencoderNet(nn.Module):
    def __init__(self):
        super(AutoencoderNet,  self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(8,3),
            nn.Sigmoid(),           
        )

        self.decoder = nn.Sequential(
            nn.Linear(3,8),
        )

    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded,decoded
