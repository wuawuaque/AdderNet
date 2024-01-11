#Copyright (C) 2019. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for more details.

import os
from resnet20 import resnet20
from vq_addernet_VAEnet import convVAE
import torch
from torch.autograd import Variable
from torchvision.datasets import CIFAR10
from torchvision.datasets import MNIST
import torchvision.transforms as transforms
from torch.utils.data import DataLoader 
import argparse
import math
from vars import *
parser = argparse.ArgumentParser(description='train-addernet')

# Basic model parameters.
parser.add_argument('--data', type=str, default='/cache/data/')
parser.add_argument('--output_dir', type=str, default='/cache/models/')
args = parser.parse_args()

os.makedirs(args.output_dir, exist_ok=True)  

acc = 0
acc_best = 0

if datasettype=='CIFAR10':
###以下为CIFAR10数据集的数据处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    data_train = CIFAR10(args.data,
                    transform=transform_train)
    data_test = CIFAR10(args.data,
                    train=False,
                    transform=transform_test)

if datasettype=='MNIST':   
    ###以下为MNIST数据集的数据处理
    transform_train = transforms.Compose([
        transforms.RandomCrop(28, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5), (0.5)),
    ])

    data_train = MNIST(args.data,
                    transform=transform_train)
    data_test = MNIST(args.data,
                    train=False,
                    transform=transform_test)



data_train_loader = DataLoader(data_train, batch_size=train_batch_size, shuffle=True, num_workers=8)#MINIST:60000//train_batch_size
data_test_loader = DataLoader(data_test, batch_size=test_batch_size, num_workers=0)##MINIST:10000//test_batch_size

# net = resnet20().cuda()
net =convVAE().cuda()
criterion = torch.nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)

def adjust_learning_rate(optimizer, epoch):
    """For resnet, the lr starts from 0.1, and is divided by 10 at 80 and 120 epochs"""
    # lr = 0.05 * (1+math.cos(float(epoch)/400*math.pi))
    lr = 0.05 * (1+math.cos(float(epoch)/epochsum*math.pi))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
def train(epoch):
    adjust_learning_rate(optimizer, epoch)
    global cur_batch_win
    net.train()
    loss_list, batch_list = [], []
    for i, (images, labels) in enumerate(data_train_loader):
        images, labels = Variable(images).cuda(), Variable(labels).cuda()
 
        optimizer.zero_grad()
        # imageshalf= images.half()
        output = net(images)
        loss = net.loss_function(images, output[0], output[1], output[2])
        # loss = criterion(output, labels)
        nmse= torch.sum((output[0] - images) ** 2) / torch.sum(images ** 2)

        loss_list.append(loss.data.item())
        batch_list.append(i+1)
 
        if (i == 1) or (i % 100 == 0):
            print('Train - Epoch %d, Batch: %d, Loss: %f, nmse: %f' % (epoch, i, loss.data.item(),nmse))
 
        loss.backward()
        optimizer.step()
 
def test(epoch):
    global acc, acc_best
    net.eval()
    total_loss = 0
    total_mse = 0
    avg_loss = 0.0
    avg_nmse = 0.0
    itertime = 0
    # mse = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(data_test_loader):
            images, labels = Variable(images).cuda(), Variable(labels).cuda()
            output = net(images)
            total_loss += net.loss_function(images, output[0], output[1], output[2])
            # pred = output.data.max(1)[1]
            #归一化mse求和
            total_mse += torch.sum((output[0] - images) ** 2) / torch.sum(images ** 2)
            itertime += 1
    # 
    avg_loss = total_loss / itertime
    avg_nmse = total_mse / itertime
    if acc_best < acc:
        acc_best = acc
    print('Test Avg. Loss: %f, mse: %f' % (avg_loss.data.item(), avg_nmse))
    #将下面的数据写入文档test.csv文件中
    with open(args.output_dir + 'addnet_sub_Encoder_all_noCNNtans_epoch50.csv', 'a') as f:
        f.write(str(epoch) + ' ' + str(avg_loss.data.item()) + ' ' + str(avg_nmse) + '\n')


 
 
def train_and_test(epoch):
    train(epoch)
    test(epoch)
 
 
def main():
    # epoch = 400
    epoch = epochsum
    for e in range(1, epoch):
        train_and_test(e)
    # torch.save(net,args.output_dir + 'cnn40_net')
    torch.save(net.state_dict(), args.output_dir + 'adder_net_dict')
    print('adder_net_dict_Weight saved ')
 
 
if __name__ == '__main__':
    main()
