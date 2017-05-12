#! /usr/bin/env python
#input:512

import torch
import torchvision
from torch.utils.data import DataLoader
# from torchvision.datasets import MNIST
from torchvision import transforms
from torch import optim
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data_utils
from torch.autograd import Variable
import numpy as np
import tqdm

import stTagBox as sttb

_TRAIN_SET = "./image_out-256-sdco/stv2k_train/"
_TEST_SET = "./image_out-256-sdco/stv2k_test/"

learning_rate=1e-3
batch_size=5
epoches=2

trans_img=transforms.Compose([
	transforms.ToTensor()
	])
torch.set_default_tensor_type('torch.DoubleTensor')

def get_set(path, randpick = None):
    features = []
    targets = []
    picset = sttb.walk_path(path, randpick)
    for fs in tqdm.tqdm(picset):
        imgarr, tagarr = sttb.get_data(fs, path)
        features.append(imgarr)
        targets.append(tagarr)
    return torch.from_numpy(np.array(features) * 1.).view(randpick, 1, 256, -1), torch.from_numpy(np.array(targets)).view(randpick, 1, 64, -1)
trainf, trarf = get_set(_TRAIN_SET, 50)
testf, tearf = get_set(_TEST_SET, 20)
trainset = data_utils.TensorDataset(trainf, trarf)
testset = data_utils.TensorDataset(testf, tearf)
trainloader = data_utils.DataLoader(trainset, batch_size = batch_size, shuffle = True, num_workers = 4)
testloader = data_utils.DataLoader(testset, batch_size = batch_size, shuffle = True, num_workers = 4)

# trainset=MNIST('./ministdata',train=True,transform=trans_img, download = True)
# testset=MNIST('./ministdata',train=False,transform=trans_img, download = True)

# trainloader=DataLoader(trainset,batch_size=batch_size,shuffle=True,num_workers=4)
# testloader=DataLoader(testset,batch_size=batch_size,shuffle=False,num_workers=4)

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

class Net(nn.Module):
    #build network
    def __init__(self):
        super(Net, self).__init__()
        self.max1=nn.MaxPool2d(4,stride=4)
        self.max2=nn.MaxPool2d(2,stride=2)
        self.max3=nn.MaxPool2d((8,1),stride=1)
        self.conv1=nn.Conv2d(1,720,9,stride=1,padding=0)
        self.conv2=nn.Conv2d(720,960,9,stride=1,padding=0)
        self.conv3=nn.Conv2d(960,1536,9,stride=1,padding=0)
        self.conv4=nn.Conv2d(1536,2048,8,stride=1,padding=0)
        self.conv5=nn.Conv2d(2048,3072,1,stride=1,padding=0)


    def forward(self,x):
        x=self.max1(x)
        x=self.conv1(x)
        x=self.max2(F.dropout(F.leaky_relu(self.conv2(x)),p=0.5))
        x=F.dropout(F.leaky_relu(self.conv3(x)),p=0.5)
        x=self.max3(F.dropout(F.leaky_relu(self.conv4(x)),p=0.5))
        x=F.dropout(F.leaky_relu(self.conv5(x)),p=0.5)
        x=x.view(-1,self.num_flat_features(x))
        x=F.softmax(x)
        return x

    def num_flat_features(self,x):
        size=x.size()[1:]
        num_features=1
        for s  in size:
            num_features *= s
        return num_features



net=Net()#.cuda()

criterian=nn.CrossEntropyLoss(size_average=False)
optimizer=optim.Adam(net.parameters(),lr=learning_rate)

#train
for i in range(epoches):
    running_loss=0
    running_acc=0
    for (img,label) in trainloader:
        img=Variable(img) #.cuda())
        label=Variable(label) #.cuda())

        optimizer.zero_grad()
        output=net(img)
        loss=criterian(output,label)
        #backward
        loss.backward()
        optimizer.step()

        running_loss+=loss.data[0]
        _, predict=torch.max(output,1)
        correct_num=(predict==label).sum()
        running_acc+=correct_num[0]

    running_loss/=len(trainset)
    running_acc/=len(trainset)
    print("Loss:")
    print(running_loss)
    print("Accuracy:")
    print(running_acc)
#    print("[%d%d] Loss: %.5f, Acc: %.2f" %(i+1,epoches,float(running_loss), 100 * float(running_acc)))


#test
net.eval()

testloss=0.
testacc=0.
for (img, label) in testloader:
    img = Variable(img.cuda())
    label = Variable(label.cuda())

    output = net(img)
    loss = criterian(output, label)
    testloss += loss.data[0]
    _, predict = torch.max(output, 1)
    num_correct = (predict == label).sum()
    testacc += num_correct.data[0]

testloss /= len(testset)
testacc /= len(testset)
print("Loss:")
print(testloss)
print("Accuracy:")
print(testacc)
