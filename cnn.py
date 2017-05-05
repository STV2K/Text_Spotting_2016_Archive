#! /usr/bin/env python
#input:512

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np

learning_rate=1e-3
batch_size=100
epoches=50

trans_img=transforms.Compose([
	transforms.ToTensor()
	])

trainset=MNIST('./data',train=True,transform=trans_img)
testset=MNIST('./data',train=False,transform=trans_img)

trainloader=DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=4)
testloader=DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=4)

def Maxout(inputs, num_units):
	shape = np.shape(inputs)
	if shape[0] is None:
		shape[0] = -1
	num_channels = shape[0]
	if num_channels % num_units:
		raise ValueError('number of features({}) is not a multiple of num_units({})'.format(num_channels, num_units))
	rshape = (num_units, num_channels // num_units,shape[1],shape[2])
	outputs = np.argmax(inputs.reshape(rshape), 0)
	return outputs

#build network
class Net(nn.Module):
	#build network
	def __init__(self):
		super(Net, self).__init__()
		self.max1=nn.MaxPool2d(2,stride=1)
		self.max2=nn.MaxPool2d(2,stride=2)
		self.conv1=nn.Conv2d(1,48,11,stride=5,padding=1)
		self.conv2=nn.Conv2d(48,96,9,stride=1,padding=0)
		self.conv3=nn.Conv2d(96,128,9,stride=1,padding=0)
		self.conv4=nn.Conv2d(128,512,8,stride=1,padding=0)
		self.conv5=nn.Conv2d(512,512,5,stride=1,padding=0)
		self.conv6=nn.Conv2d(512,148,1,stride=1,padding=0)

	def forward(self,x):
		x=self.max2(x)
		#x=Maxout(self.conv1(x),2)
		x=self.max2(F.dropout(self.conv2(x),p=0.5))
		x=F.dropout(self.conv3(x),p=0.5)
		x=self.max1(F.dropout(self.conv4(x),p=0.5))           
		x=F.dropout(self.conv5(x),p=0.5)
		x=F.dropout(self.conv6(x),p=0.5)
		x=x.view(-1,self.num_flat_features(x))
		x=F.softmax(out)
		return x

	def num_flat_features(self,x):
		size=x.size()[1:]
		num_features=1
		for s  in size:
			num_features *= s
		return num_features



net=Net()
net

criterian=nn.CrossEntropyLoss(size_average=False)
optimizer=optim.Adam(net.parameters(),lr=learning_rate)

#train
for i in range(epoches):
    running_loss=0.
    running_acc=0.
    for (img,label) in trainloader:
        img=Variable(img)
        label=Variable(label)

        optimizer.zero_grad()
        output=net(img)
        loss=criterian(output,label)
        #backward
        loss.backward()
        optimizer.step()

        running_loss += loss.data[0]
        _, predict=torch.max(output,1)
        correct_num=(predict==label).sum()
        running_acc += correct_num[0]

    running_loss /= len(trainset)
    running_acc /= len(trainset)
    print("[%d%d] Loss: %.5f, Acc: %.2f" %(i+1,epoches,running_loss,100*running_acc))



