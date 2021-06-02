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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, drop_last=True, shuffle=True, num_workers=0)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, drop_last=True, shuffle=False, num_workers=0)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


featuresB = 12
featuresC = 18
featuresD = 24
width = 32
height = 32

gammaMem = 0.33
alphaRec = 0.01
betaFB = 0.33
memory = 0.33

numberEpochs = 25
timeSteps = 10

iterationNumber = 10
resCEall = np.empty((timeSteps, numberEpochs, iterationNumber))

for iterationIndex in range(0, iterationNumber):

    pcNet = predCodNet(featuresB, featuresC, featuresD, gammaMem, alphaRec, betaFB, memory)
    pcNet = pcNet.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizerPCnet = optim.SGD(pcNet.parameters(), lr=0.005, momentum=0.9)

    for epoch in range(0, numberEpochs):  # loop over the data set twice

        print(iterationIndex)
        print(epoch)

        #train
        for i, data in enumerate(trainloader, 0):

            bTemp = torch.zeros(batchSize, featuresB, width, height).cuda()
            cTemp = torch.zeros(batchSize, featuresC, 16, 16).cuda()
            dTemp = torch.zeros(batchSize, featuresD, 8, 8).cuda()

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            # zero the parameter gradients
            optimizerPCnet.zero_grad()
            finalLoss = 0

            bTemp.requires_grad = True
            cTemp.requires_grad = True
            dTemp.requires_grad = True

            outputs, aTemp, bTemp, cTemp, dTemp, errorB, errorC, errorD = pcNet(inputs, bTemp, cTemp, dTemp, 'forward')

            for tt in range(timeSteps):
                # print(tt)
                outputs, aTemp, bTemp, cTemp, dTemp, errorB, errorC, errorD = pcNet(inputs, bTemp, cTemp, dTemp, 'full')
                loss = criterion(outputs, labels)
                finalLoss += loss

            finalLoss.backward(retain_graph=True)  # makes sense? do we need the intermedary results?
            optimizerPCnet.step()

        path = f"/home/andrea/PycharmProjects/PredictiveCoding/models/pcNetCEnewFW2_E{epoch}_I{iterationIndex}.pth"
        torch.save({"module": pcNet.state_dict(), "epoch": epoch}, path)

        #compute test accuracy
        correct = np.zeros(timeSteps)
        total = 0
        for i, data in enumerate(testloader, 0):
            # print(i)

            bTemp = torch.zeros(batchSize, featuresB, width, height).cuda()
            cTemp = torch.zeros(batchSize, featuresC, 16, 16).cuda()
            dTemp = torch.zeros(batchSize, featuresD, 8, 8).cuda()

            bTemp.requires_grad = True
            cTemp.requires_grad = True
            dTemp.requires_grad = True

            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs, aTemp, bTemp, cTemp, dTemp, errorB, errorC, errorD = pcNet(inputs, bTemp, cTemp, dTemp, 'forward')

            for tt in range(timeSteps):
                outputs, aTemp, bTemp, cTemp, dTemp, errorB, errorC, errorD = pcNet(inputs, bTemp, cTemp, dTemp, 'full')

                _, predicted = torch.max(outputs.data, 1)
                correct[tt] = correct[tt] + (predicted == labels).sum().item()

            total += labels.size(0)

        resCEall[:, epoch, iterationIndex] = (100 * correct / total)

        np.save(f"/home/andrea/PycharmProjects/PredictiveCoding/accuracies/accTrainingCEnewFW2.npy", resCEall)
        print('Finished Training')





