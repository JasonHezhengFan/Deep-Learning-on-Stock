#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.autograd import Variable


# In[3]:


from torch.utils.data import Dataset
#实现custom pytorch dataset
class GetLoader(Dataset):
# 初始化函数，得到数据
    def __init__(self, data_root, data_label):
        self.data = data_root
        self.label = data_label
    # index是根据batchsize划分数据后得到的索引，最后将data和对应的labels进行一起返回
    def __getitem__(self, index):
        data = self.data[index]
        labels = self.label[index]
        return data, labels
    # 该函数返回数据大小长度，目的是DataLoader方便划分，如果不知道大小，DataLoader会一执行错误
    def __len__(self):
        return len(self.data)


# In[4]:


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,device):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)  # batch_first=True仅仅针对输入而言
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # 设置初始状态h_0与c_0的状态是初始的状态，一般设置为0，尺寸是,x.size(0)
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)

        # Forward propagate RNN
        out, (h_n, c_n) = self.lstm(x, (h0, c0))  # 送入一个初始的x值，作为输入以及(h0, c0)

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])  # output也是batch_first, 实际上h_n与c_n并不是batch_first
        return out


# In[5]:


# 搭建CONV1D-LSTM Model (Many-to-One)
class CONV1D_LSTM(nn.Module):
    def __init__(self ,in_channel,out_channel, hidden_size, num_layers, num_classes,device):
        super(CONV1D_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel,kernel_size=1, stride=1, bias=False)
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel,kernel_size=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.lstm = nn.LSTM(out_channel, hidden_size, num_layers, batch_first=True)  # batch_first=True仅仅针对输入而言
        self.fc = nn.Linear(hidden_size, num_classes)
        
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight.to(device), mode='fan_out', nonlinearity='relu')


        
        
        
    def forward(self, x):
        
        #forward prop
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out=torch.transpose(out,2,1)
        #因为pytorch里lstm和conv1d的input sequence位置不一样，需要调整。
        
        # 设置初始状态h_0与c_0的状态是初始的状态，一般设置为0，尺寸是,x.size(0)
        h0 = Variable(torch.zeros(self.num_layers, out.size(0), self.hidden_size)).to(device)
        c0 = Variable(torch.zeros(self.num_layers, out.size(0), self.hidden_size)).to(device)

        # Forward propagate RNN
        out, (h_n, c_n) = self.lstm(out, (h0, c0))  # 送入一个初始的x值，作为输入以及(h0, c0)

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])  # output也是batch_first, 实际上h_n与c_n并不是batch_first
        return out


# In[6]:


# 搭建ResNet-LSTM Model (Many-to-One) 相较于简单的cnn-lstm，resnet可以防止梯度消失
#搭建第一类ResNet block
class BasicBlock(nn.Module):#基本残差网络的一个模块类
    expansion = 1#每一个residual block中不改变width,height,channel数，即增加的residual部分不需要做卷积处理

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)#stride=1,kernel_size=3,padding=1保证了data的
                                                                                   #size不变 
        self.bn1 = nn.BatchNorm1d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm1d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:  #BasicBlock内不需要调整residual的height,width,channel
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


# In[7]:


#组建block成ResNet
class ResNet1D_LSTM(nn.Module):
    def __init__(self, block, blocks_num, num_classes, feature_channel,hidden_size, num_layers,device ):
        super(ResNet1D_LSTM, self).__init__()
        self.in_channel = 64
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.conv1 = nn.Conv1d(in_channels=feature_channel, out_channels=self.in_channel, kernel_size=2, stride=2,
                               padding=0, bias=False)
        self.bn1 = nn.BatchNorm1d(self.in_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)  #channel数变为n/2(非整数时向下取整)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])         #按照已有结论按二次方形式增长ResNet不同阶段的channel
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)#stride=2表示想把上一个layer传过来的size缩减为1/2
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.avgpool = nn.AdaptiveAvgPool1d((1, 1))  # output size = (1, 1) 
        self.lstm = nn.LSTM(256, hidden_size, num_layers, batch_first=True)  # batch_first=True仅仅针对输入而言
        self.fc = nn.Linear(hidden_size, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.to(device), mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm1d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x) #预处理
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x) #残差网络
        x = self.layer2(x)
        x = self.layer3(x)
        
        x=torch.transpose(x,2,1)
        #因为pytorch里lstm和conv1d的input sequence位置不一样，需要调整。
        
        # 设置初始状态h_0与c_0的状态是初始的状态，一般设置为0，尺寸是,x.size(0)
        h0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)
        c0 = Variable(torch.zeros(self.num_layers, x.size(0), self.hidden_size)).to(device)

        # Forward propagate RNN
        out, (h_n, c_n) = self.lstm(x, (h0, c0))  # 送入一个初始的x值，作为输入以及(h0, c0)

        # Decode hidden state of last time step
        out = self.fc(out[:, -1, :])  # output也是batch_first, 实际上h_n与c_n并不是batch_first
        return out


# In[ ]:




