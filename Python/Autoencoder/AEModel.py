#Importing packages
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import pandas as pd
import numpy as np



class AE(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=256
        )
        self.bn1=nn.BatchNorm1d(256)
        
        self.encoder_output_layer = nn.Linear(
            in_features=256, out_features=128
        )
        self.bn2=nn.BatchNorm1d(128)
        
        self.decoder_hidden_layer = nn.Linear(
            in_features=128, out_features=256
        )
        self.bn3 =nn.BatchNorm1d(256)

        self.decoder_output_layer = nn.Linear(
            in_features=256, out_features=kwargs["input_shape"]
        )
        self.bn4 =nn.BatchNorm1d(kwargs["input_shape"])

    def forward(self, features):
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        activation = self.bn1(activation)
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        code = self.bn2(code)
        activation = self.decoder_hidden_layer(code)
        activation = torch.relu(activation)
        activation = self.bn3(activation)
        activation = self.decoder_output_layer(activation)
        reconstructed = torch.relu(activation)
        reconstructed = self.bn4(reconstructed)
        return code, reconstructed


class AE_T(nn.Module):
    def __init__(self,**kwargs):
        super().__init__()

        #Hidden Layer 1
        self.encoder_hidden_layer = nn.Linear(
            in_features=kwargs["input_shape"], out_features=256, bias = False
        )
        self.encoder_hidden_layer.weight = nn.Parameter(torch.randn(256, kwargs["input_shape"]))

        #BN1
        self.bn1=nn.BatchNorm1d(256)
        
        #Encoder Output Layer
        self.encoder_output_layer = nn.Linear(
            in_features=256, out_features=128, bias = False
        )
        self.encoder_output_layer.weight = nn.Parameter(torch.randn(128, 256))

        #BN2 layer
        self.bn2=nn.BatchNorm1d(128)
        
        #BN3 layer after decoder layer1
        self.bn3 =nn.BatchNorm1d(256)

        #BN4 Layer after decoder layer2
        self.bn4 =nn.BatchNorm1d(kwargs["input_shape"])

    def forward(self, features):
        #Layer 1 input -> 256
        activation = self.encoder_hidden_layer(features)
        activation = torch.relu(activation)
        activation = self.bn1(activation)

        #Layer2 256 ->128
        code = self.encoder_output_layer(activation)
        code = torch.relu(code)
        code = self.bn2(code)

        #Layer3 128 -> 256
        activation = F.linear(code, self.encoder_output_layer.weight.t())
        activation = torch.relu(activation)
        activation = self.bn3(activation)

        #Layer4 256->input
        activation = F.linear(activation, self.encoder_hidden_layer.weight.t())
        reconstructed = torch.relu(activation)
        reconstructed = self.bn4(reconstructed)
        return code, reconstructed


class MyDataset(Dataset):
  def __init__(self,data_df):
    data_df = data_df
    self.x_train = torch.tensor(data_df,dtype=torch.float32)
    #self.y_train = torch.tensor(labels)
 
  def __len__(self):
    return len(self.x_train)
   
  def __getitem__(self,idx):
    return self.x_train[idx]