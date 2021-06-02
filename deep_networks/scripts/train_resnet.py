#########################
# In this script we train
# resnet on CIFAR100
# then we will use the
# trained model for PC
# purposes.
# This code is the modified version of the following:
# https://github.com/weiaicunzai/pytorch-cifar100/blob/master/train.py
#########################

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

import torch
from   datetime import datetime
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from   torch.nn import Conv2d, MaxPool2d, Linear

from utils import get_cifar_test_dataloader, get_cifar_training_dataloader, WarmUpLR

########################
## GLOBAL CONFIGURATIONS
########################
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 200
MILESTONES = [60, 120, 160]

#tensorboard log dir
LOG_DIR = 'runs'

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10

#time of we run the script
TIME_NOW = datetime.now().isoformat()

########################
########################

def get_cifar_resnet18():
    resnet         = models.resnet18()
    resnet.conv1   = Conv2d(3, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
    resnet.maxpool = MaxPool2d(kernel_size=1, stride=1, ceil_mode=False)
    resnet.fc      = Linear(in_features=512, out_features=100, bias=True)
    return resnet

# TRAINING
def train(epoch):

    net.train()
    for batch_index, (images, labels) in enumerate(cifar100_training_loader):
        if epoch <= 1:
            warmup_scheduler.step()

        labels = labels.cuda()
        images = images.cuda()

        optimizer.zero_grad()
        outputs = net(images)
        loss = loss_function(outputs, labels)
        loss.backward()
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * 128 + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))

def eval_training(epoch):
    net.eval()

    test_loss = 0.0 # cost function error
    correct = 0.0

    for (images, labels) in cifar100_test_loader:

        images = images.cuda()
        labels = labels.cuda()

        outputs = net(images)
        loss = loss_function(outputs, labels)
        test_loss += loss.item()
        _, preds = outputs.max(1)
        correct += preds.eq(labels).sum()

    print('Test set: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
        test_loss / len(cifar100_test_loader.dataset),
        correct.float() / len(cifar100_test_loader.dataset)
    ))
    print()

    return correct.float() / len(cifar100_test_loader.dataset)

cifar100_training_loader = get_cifar_training_dataloader(
    CIFAR100_TRAIN_MEAN,
    CIFAR100_TRAIN_STD,
    num_workers=2,
    batch_size=128,
    shuffle=True,
    root='../datasets'
)

cifar100_test_loader = get_cifar_test_dataloader(
    CIFAR100_TRAIN_MEAN,
    CIFAR100_TRAIN_STD,
    num_workers=2,
    batch_size=128,
    shuffle=False,
    root='../datasets'
)

net = get_cifar_resnet18()
net.cuda()


loss_function = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
train_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=MILESTONES, gamma=0.2) #learning rate decay
iter_per_epoch = len(cifar100_training_loader)
warmup_scheduler = WarmUpLR(optimizer, iter_per_epoch * 1)
checkpoint_path = os.path.join(CHECKPOINT_PATH, 'resnet', TIME_NOW)

if not os.path.exists(checkpoint_path):
    os.makedirs(checkpoint_path)
checkpoint_path = os.path.join(checkpoint_path, '{net}-{epoch}-{type}.pth')


best_acc = 0.0
for epoch in range(1, EPOCH):
    if epoch > 1:
        train_scheduler.step(epoch)

    train(epoch)
    acc = eval_training(epoch)

    #start to save best performance model after learning rate decay to 0.01 
    if epoch > MILESTONES[1] and best_acc < acc:
        torch.save(net.state_dict(), checkpoint_path.format(net='resnet', epoch=epoch, type='best'))
        best_acc = acc
        continue

    if not epoch % SAVE_EPOCH:
        torch.save(net.state_dict(), checkpoint_path.format(net='resnet', epoch=epoch, type='regular'))