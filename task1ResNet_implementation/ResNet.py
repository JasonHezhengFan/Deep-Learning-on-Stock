#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch
import torch.nn as nn



#搭建第一类ResNet block
class BasicBlock(nn.Module):#基本残差网络的一个模块类
    expansion = 1#每一个residual block中不改变width,height,channel数，即增加的residual部分不需要做卷积处理

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)#stride=1,kernel_size=3,padding=1保证了data的
                                                                                   #size不变 
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
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




#搭建第二类ResNet block
class Bottleneck(nn.Module):
    expansion = 4 #channel数增加expansion倍

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=1, stride=1, bias=False)  # size不变
        self.bn1 = nn.BatchNorm2d(out_channel)

        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, bias=False, padding=1)#stride=1时size不变
        self.bn2 = nn.BatchNorm2d(out_channel)
 
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # height和width不变，channel变为out_channel*expansion
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample #将residual同步调整为data的size

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out



#组建block成ResNet
class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, feature_channel=3,device = torch.device("cpu")):
        super(ResNet, self).__init__()
        self.in_channel = 64

        self.conv1 = nn.Conv2d(in_channels=feature_channel, out_channels=self.in_channel, kernel_size=2, stride=2,
                               padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  #channel数变为n/2(非整数时向下取整)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])         #按照已有结论按二次方形式增长ResNet不同阶段的channel
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1) 
        self.fc = nn.Linear(512 * block.expansion, num_classes) #fully connected

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.to(device), mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

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
        x = self.layer4(x)
        
        x = self.avgpool(x)#fully connection
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# In[279]:


def resnet34(num_classes=1000,feature_channel=3,device = torch.device("cpu")):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, feature_channel=feature_channel,device = torch.device("cpu"))






from torch.utils.data import Dataset


# In[252]:


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

def model_optimize(num_epochs,train_loader,device,loss_function,optimizer):
    total_step=0
    for epoch in range(num_epochs):
        for data in train_loader:
            images, labels = data
            images=images.type(torch.FloatTensor)
            labels=labels.type(torch.FloatTensor)
            images=images.to(device)
            labels=labels.to(device)
            #forward pass
            outputs=model(images)
            loss=loss_function(outputs,labels.long())
        
            #Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_step+=1
            if (total_step)%10==0:#each 10 iterations is one epoch
                print("Epoch [{}/{}],step[{}] Loss:{:.4f}".format(epoch+1,num_epochs,total_step,loss.item()))




# In[334]:


def model_evaluation(test_loader):
    with torch.no_grad():
        correct=0
        total=0
        for images, labels in test_loader:
            images=images.type(torch.FloatTensor)
            labels=labels.type(torch.FloatTensor)
            images=images.to(device)
            labels=labels.to(device)
            outputs=model(images)
            _,predicted=torch.max(outputs.data,1)
            total+=labels.size(0)
            correct+=(predicted==labels).sum().item()
            print('accuracy of the model on the test images: {}%'.format(100*correct/total))
    return predicted



def predict(test_loader):
    with torch.no_grad():
        for images in test_loader:
            images=images.type(torch.FloatTensor)
            images=images.to(device)
            outputs=model(images)
            _,predicted=torch.max(outputs.data,1)
    return predicted