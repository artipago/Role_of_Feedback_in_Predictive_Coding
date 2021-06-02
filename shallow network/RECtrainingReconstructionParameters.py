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


iterationNumber = 10
numberEpochs = 20

resRecLossAll = np.empty((3, numberEpochs, iterationNumber))
resRecAll = np.empty((numberEpochs, iterationNumber))

for iterationIndex in range(0, iterationNumber):

    print(iterationIndex)

    pcNet = predCodNet(featuresB, featuresC, featuresD, gammaMem, alphaRec, betaFB)
    pcNet = pcNet.cuda()

    checkpointPhase = torch.load(f"/models/pcNetREC_FF_E24_I{iterationIndex}.pth")
    pcNet.load_state_dict(checkpointPhase["module"])

    for name, p in pcNet.named_parameters():
        if name.split('.')[0] in ['convAB', 'convBC', 'convCD', 'fc1', 'fc2']:
            p.requires_grad_(False)

    criterionMSE = nn.functional.mse_loss
    optimizerPCnet = optim.SGD(pcNet.parameters(), lr=0.01, momentum=0.9)

    for epoch in range(0, numberEpochs):  

        for i, data in enumerate(trainloader, 0):

            b = torch.randn(batchSize, featuresB, width, height).cuda()
            c = torch.randn(batchSize, featuresC, 16, 16).cuda()
            d = torch.randn(batchSize, featuresD, 8, 8).cuda()

            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            optimizerPCnet.zero_grad()
            lossA = 0
            lossB = 0
            lossC = 0

            outputs, a, b, c, d, errorB, errorC, errorD = pcNet(inputs, b, c, d, 'forward')
            outputs, aR, bR, cR, dR, errorB, errorC, errorD = pcNet(inputs, b, c, d, 'reconstruction')

            lossA = criterionMSE(inputs, aR)
            lossB = criterionMSE(b, bR)
            lossC = criterionMSE(c, cR)

            finalLoss = lossA + lossB + lossC

            finalLoss.backward(retain_graph=True) 
            optimizerPCnet.step()

        path = f"/models/pcNetREC_FF_Rec_E{epoch}_I{iterationIndex}.pth"
        torch.save({"module": pcNet.state_dict(), "epoch": epoch}, path)

        finalLossA = 0
        finalLossB = 0
        finalLossC = 0

        correct = 0
        total = 0

        for i, data in enumerate(testloader, 0):

            b = torch.randn(batchSize, featuresB, width, height).cuda()
            c = torch.randn(batchSize, featuresC, 16, 16).cuda()
            d = torch.randn(batchSize, featuresD, 8, 8).cuda()

            inputs, labels = data
            inputs = inputs.cuda()
            labels = labels.cuda()

            outputs, a, b, c, d, errorB, errorC, errorD = pcNet(inputs, b, c, d, 'forward')
            outputs, aR, bR, cR, dR, errorB, errorC, errorD = pcNet(inputs, b, c, d, 'reconstruction')

            finalLossA = finalLossA + criterionMSE(inputs, aR)
            finalLossB = finalLossB + criterionMSE(b, bR)
            finalLossC = finalLossC + criterionMSE(c, cR)

            total += labels.size(0)

            _, predicted = torch.max(outputs.data, 1)
            correct = correct + (predicted == labels).sum().item()

        resRecAll[epoch, iterationIndex] = (100 * correct / total)

        resRecLossAll[0, epoch, iterationIndex] = finalLossA / total
        resRecLossAll[1, epoch, iterationIndex] = finalLossB / total
        resRecLossAll[2, epoch, iterationIndex] = finalLossC / total

        print(100 * correct / total)
        print(finalLossA / total)
        print(finalLossB / total)
        print(finalLossC / total)

print('Finished Training')
np.save(f"/accuracies/recLossTrainingRECff_rec.npy", resRecLossAll)
np.save(f"/accuracies/accTrainingRECff_rec.npy", resRecAll)


