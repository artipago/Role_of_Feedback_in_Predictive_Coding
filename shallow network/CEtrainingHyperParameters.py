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

batchSize = 128
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batchSize, shuffle=True, drop_last=True, num_workers=4)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchSize, drop_last=True, shuffle=False, num_workers=4)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def noisy(noise_typ, image, noiseParam):
    if noise_typ == "gauss":
        gaussNoise = torch.randn(image.shape) * noiseParam
        noisy = image + gaussNoise.cuda()
        return noisy

    elif noise_typ == "s&p":
        thUp = noiseParam/2 * torch.ones(image.shape[2],image.shape[3])
        thDown = (1 - noiseParam/2) * torch.ones(image.shape[2], image.shape[3])
        saltNpepper = torch.rand(image.shape[2],image.shape[3])
        unos = torch.max(image) * torch.ones(image.shape)
        menoUnos = - torch.max(image) * torch.ones(image.shape)
        noisy = image

        unos=unos.cuda()
        menoUnos=unos.cuda()
        saltNpepper=saltNpepper.cuda()
        thDown=thDown.cuda()
        thUp=thUp.cuda()

        noisy = torch.where(saltNpepper >= thUp, noisy, unos)
        noisy = torch.where(saltNpepper <= thDown, noisy, menoUnos)
        return noisy


featuresB = 12
featuresC = 18
featuresD = 24
width = 32
height = 32

criterion = nn.CrossEntropyLoss()

iterationNumber = 10
additionalEpochs = 10
timeSteps = 10

gammaFw = torch.rand(iterationNumber)
alphaRec = torch.rand(iterationNumber)
betaFB = torch.rand(iterationNumber)
memory = torch.rand(iterationNumber)

noiseLevelSP = [0, .02, .04, .06]
noiseLevelG = [0, .2, .4, .8]

resRECgauss = np.empty((timeSteps+1, len(noiseLevelG), iterationNumber))
resRECsnp = np.empty((timeSteps+1, len(noiseLevelSP), iterationNumber))

confRECgauss = np.empty((timeSteps, len(noiseLevelG), iterationNumber))
confRECsnp = np.empty((timeSteps, len(noiseLevelSP), iterationNumber))

recErrRECgauss = np.empty((3, timeSteps, len(noiseLevelG), iterationNumber))
recErrRECsnp = np.empty((3, timeSteps, len(noiseLevelG), iterationNumber))

softMaxFunk = torch.nn.Softmax(dim=1)

for iterationIndex in range(0, iterationNumber):
    for ii in range(len(noiseLevelSP)):

        pcNetG = predCodNet(featuresB, featuresC, featuresD, gammaFw[iterationIndex], alphaRec[iterationIndex], betaFB[iterationIndex], memory[iterationIndex])
        pcNetG = pcNetG.cuda()

        pcNetSnP = predCodNet(featuresB, featuresC, featuresD, gammaFw[iterationIndex], alphaRec[iterationIndex], betaFB[iterationIndex], memory[iterationIndex])
        pcNetSnP = pcNetSnP.cuda()

        checkpointPhase = torch.load(f"/models/pcNetCE_E24_I{iterationIndex}.pth") #here we load the parameters previously trained 
        pcNetG.load_state_dict(checkpointPhase["module"])
        pcNetSnP.load_state_dict(checkpointPhase["module"])

        for p in pcNetG.parameters():
            p.requires_grad = False

        for p in pcNetSnP.parameters():
            p.requires_grad = False

        pcNetG.gammaFw.requires_grad = True
        pcNetG.alphaRec.requires_grad = True
        pcNetG.betaFB.requires_grad = True
        pcNetG.memory.requires_grad = True
        optimizerPCnetG = optim.Adam([pcNetG.gammaFw, pcNetG.alphaRec, pcNetG.betaFB, pcNetG.memory], lr=0.001, weight_decay=0.00001)

        pcNetSnP.gammaFw.requires_grad = True
        pcNetSnP.alphaRec.requires_grad = True
        pcNetSnP.betaFB.requires_grad = True
        pcNetSnP.memory.requires_grad = True
        optimizerPCnetSnP = optim.Adam([pcNetSnP.gammaFw, pcNetSnP.alphaRec, pcNetSnP.betaFB, pcNetSnP.memory], lr=0.001, weight_decay=0.00001)

        for addEpoch in range(additionalEpochs):
            for i, data in enumerate(trainloader, 0):

                bTempG = torch.zeros(batchSize, featuresB, width, height).cuda()
                cTempG = torch.zeros(batchSize, featuresC, 16, 16).cuda()
                dTempG = torch.zeros(batchSize, featuresD, 8, 8).cuda()

                bTempSnP = torch.zeros(batchSize, featuresB, width, height).cuda()
                cTempSnP = torch.zeros(batchSize, featuresC, 16, 16).cuda()
                dTempSnP = torch.zeros(batchSize, featuresD, 8, 8).cuda()

                # get the inputs; data is a list of [inputs, labels]
                inputs, labels = data
                inputs = inputs.cuda()
                labels = labels.cuda()

                imageGauss = noisy("gauss", inputs, noiseLevelG[ii])
                imagesnp = noisy("s&p", inputs, noiseLevelSP[ii])

                # zero the parameter gradients
                optimizerPCnetG.zero_grad()
                optimizerPCnetSnP.zero_grad()

                finalLossG = 0
                finalLossSnP = 0

                outputsG, aTempG, bTempG, cTempG, dTempG, errorBG, errorCG, errorDG = pcNetG(imageGauss, bTempG, cTempG,dTempG, 'forward')
                outputsSnP, aTempSnP, bTempSnP, cTempSnP, dTempSnP, errorBSnP, errorCSnP, errorDSnP = pcNetSnP(imagesnp,bTempSnP,cTempSnP,dTempSnP,'forward')

                bTempG.requires_grad = True
                cTempG.requires_grad = True
                dTempG.requires_grad = True

                bTempSnP.requires_grad = True
                cTempSnP.requires_grad = True
                dTempSnP.requires_grad = True
                for tt in range(timeSteps):
                    outputsG, aTempG, bTempG, cTempG, dTempG, errorBG, errorCG, errorDG = pcNetG(imageGauss, bTempG, cTempG, dTempG, 'full')
                    outputsSnP, aTempSnP, bTempSnP, cTempSnP, dTempSnP, errorBSnP, errorCSnP, errorDSnP  = pcNetSnP(imagesnp, bTempSnP, cTempSnP, dTempSnP, 'full')
                    lossG = criterion(outputsG, labels)
                    lossSnP = criterion(outputsSnP, labels)
                    finalLossG += lossG
                    finalLossSnP += lossSnP

                finalLossG.backward(retain_graph=True)  
                finalLossSnP.backward(retain_graph=True)  

                optimizerPCnetG.step()
                optimizerPCnetSnP.step()

            paramG = [pcNetG.gammaFw, pcNetG.alphaRec, pcNetG.betaFB, pcNetG.memory]
            paramSnP = [pcNetSnP.gammaFw, pcNetSnP.alphaRec, pcNetSnP.betaFB, pcNetSnP.memory]

            np.save(f"/parameters/paramCEgauss_N{ii}_E{addEpoch}_I{iterationIndex}", paramG) #saving the hyperParams
            np.save(f"/parameters/paramCEsnp_N{ii}_E{addEpoch}_I{iterationIndex}", paramSnP) #saving the hyperParams

            if addEpoch == (additionalEpochs-1) :
                #compute test accuracy
                correctG = np.zeros(timeSteps+1)
                correctSNP = np.zeros(timeSteps+1)

                total = 0
                for i, data in enumerate(testloader, 0):
                    # print(i)

                    bTempG = torch.zeros(batchSize, featuresB, width, height).cuda()
                    cTempG = torch.zeros(batchSize, featuresC, 16, 16).cuda()
                    dTempG = torch.zeros(batchSize, featuresD, 8, 8).cuda()

                    bTempSnP = torch.zeros(batchSize, featuresB, width, height).cuda()
                    cTempSnP = torch.zeros(batchSize, featuresC, 16, 16).cuda()
                    dTempSnP = torch.zeros(batchSize, featuresD, 8, 8).cuda()

                    # get the inputs; data is a list of [inputs, labels]
                    inputs, labels = data
                    inputs = inputs.cuda()
                    labels = labels.cuda()

                    imageGauss = noisy("gauss", inputs, noiseLevelG[ii])
                    imagesnp = noisy("s&p", inputs, noiseLevelSP[ii])

                    outputsG, aTempG, bTempG, cTempG, dTempG, errorBG, errorCG, errorDG = pcNetG(imageGauss, bTempG, cTempG, dTempG,'forward')
                    outputsSnP, aTempSnP, bTempSnP, cTempSnP, dTempSnP, errorBSnP, errorCSnP, errorDSnP = pcNetSnP(imagesnp, bTempSnP, cTempSnP, dTempSnP, 'forward')

                    _, predictedG = torch.max(outputsG.data, 1)
                    _, predictedSNP = torch.max(outputsSnP.data, 1)
                    correctG[0] = correctG[0] + (predictedG == labels).sum().item()
                    correctSNP[0] = correctSNP[0] + (predictedSNP == labels).sum().item()

                    bTempG.requires_grad = True
                    cTempG.requires_grad = True
                    dTempG.requires_grad = True

                    bTempSnP.requires_grad = True
                    cTempSnP.requires_grad = True
                    dTempSnP.requires_grad = True

                    for tt in range(timeSteps):
                        # print(tt)
                        outputsG, aTempG, bTempG, cTempG, dTempG, errorBG, errorCG, errorDG  = pcNetG(imageGauss, bTempG, cTempG, dTempG, 'full')
                        outputsSnP, aTempSNP, bTempSnP, cTempSnP, dTempSnP, errorBSnP, errorCSnP, errorDSnP = pcNetSnP(imagesnp, bTempSnP, cTempSnP, dTempSnP, 'full')

                        _, predictedG = torch.max(outputsG.data, 1)
                        _, predictedSNP = torch.max(outputsSnP.data, 1)
                        correctG[tt+1] = correctG[tt+1] + (predictedG == labels).sum().item()
                        correctSNP[tt+1] = correctSNP[tt+1] + (predictedSNP == labels).sum().item()

                    total += labels.size(0)

                resRECgauss[:, ii, iterationIndex] = (100 * correctG / total)
                resRECsnp[:, ii, iterationIndex] = (100 * correctSNP / total)


np.save(f"/accuracies/accCEgauss.npy", resRECgauss) #saving the accuracies
np.save(f"/accuracies/accCEsnp.npy", resRECsnp) #saving the accuracies


print('Finished Training')
