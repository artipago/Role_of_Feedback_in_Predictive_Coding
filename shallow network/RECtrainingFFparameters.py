import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np

from predCodNet import predCodNet

import os

# the following part ensures that only ONE GPU is seen
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

batchSize = 512
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, drop_last=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, shuffle=False, drop_last=True, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


featuresB = 12
featuresC = 18
featuresD = 24
width = 32
height = 32

gammaMem = 0.33
alphaRec = 0.01
betaFB = 0.33

#first FF pass
iterationNumber = 10
numberEpochs = 25

resFFall = np.empty((numberEpochs, iterationNumber))

for iterationIndex in range(0, iterationNumber):

    pcNet = predCodNet(featuresB, featuresC, featuresD, gammaMem, alphaRec, betaFB)
    pcNet = pcNet.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizerPCnet = optim.SGD(pcNet.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(0, numberEpochs):  
        print(iterationIndex)
        print(epoch)

        for i, data in enumerate(trainloader, 0):

            b = torch.randn(batchSize, featuresB, width, height).cuda()
            c = torch.randn(batchSize, featuresC, 16, 16).cuda()
            d = torch.randn(batchSize, featuresD, 8, 8).cuda()

            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizerPCnet.zero_grad()
            finalLoss = 0

            outputs, a, b, c, d, errorB, errorC, errorD = pcNet(inputs, b, c, d, 'forward')
            finalLoss = criterion(outputs, labels)

            finalLoss.backward(retain_graph=True)  
            optimizerPCnet.step()

        path = f"/models/pcNetREC_FF_E{epoch}_I{iterationIndex}.pth"
        torch.save({"module": pcNet.state_dict(), "epoch": epoch}, path)

        correct = 0
        total = 0

        for i, data in enumerate(testloader, 0):

            b = torch.randn(batchSize, featuresB, width, height).cuda()
            c = torch.randn(batchSize, featuresC, 16, 16).cuda()
            d = torch.randn(batchSize, featuresD, 8, 8).cuda()

            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs, a, b, c, d, errorB, errorC, errorD  = pcNet(inputs, b, c, d, 'forward')

            _, predicted = torch.max(outputs.data, 1)
            correct = correct + (predicted == labels).sum().item()

            total += labels.size(0)

        resFFall[epoch, iterationIndex] = (100 * correct / total)

np.save(f"/accuracies/accTrainingRECff.npy", resFFall)


