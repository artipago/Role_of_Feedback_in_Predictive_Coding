import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class predCodNet(nn.Module):
    def __init__(self, featB, featC, featD, gammaFw, alphaRec, betaFB, memory):
        super(predCodNet, self).__init__()

        self.gammaFw = gammaFw * torch.ones(1).cuda()
        self.alphaRec = alphaRec * torch.ones(1).cuda()
        self.betaFB = betaFB * torch.ones(1).cuda()
        self.memory = memory * torch.ones(1).cuda()

        self.scalingL1 = np.round(np.sqrt(np.square(32*32*3) / (5*5*3)))
        self.scalingL2 = np.round(np.sqrt(np.square(16*16*featB) / (5*5*featB)))
        self.scalingL3 = np.round(np.sqrt(np.square(8*8*featC) / (5*5*featC)))

        self.poolingParameter = 2  # important to be the same everywhere

        self.convAB = nn.Conv2d(in_channels=3, out_channels=featB, kernel_size=5, stride=1, padding=2)
        self.convBC = nn.Conv2d(in_channels=featB, out_channels=featC, kernel_size=5, stride=1, padding=2)
        self.convCD = nn.Conv2d(in_channels=featC, out_channels=featD, kernel_size=5, stride=1, padding=2)
        self.convDC = nn.ConvTranspose2d(in_channels=featD, out_channels=featC, kernel_size=5, stride=1, padding=2)
        self.convCB = nn.ConvTranspose2d(in_channels=featC, out_channels=featB, kernel_size=5, stride=1, padding=2)
        self.convBA = nn.ConvTranspose2d(in_channels=featB, out_channels=3, kernel_size=5, stride=1, padding=2)
        self.pool2D = nn.MaxPool2d(self.poolingParameter, stride=2, return_indices=True)
        self.upsample = nn.Upsample(scale_factor = self.poolingParameter, mode='bilinear')
        self.unpool2D = nn.MaxUnpool2d(self.poolingParameter, stride=2)
        self.fc1 = nn.Linear(384, 120)
        self.fc2 = nn.Linear(120, 10)

    def forward(self, a, b, c, d, networkMode):

        errorB = []
        errorC = []
        errorD = []
        reconstructionB = []
        reconstructionC = []
        reconstructionD = []

        batchSize = a.shape[0]

        if networkMode == "forward":

            bNew = self.convAB(a)
            pooledB, indicesB = self.pool2D(F.relu(bNew))
            cNew =  self.convBC(pooledB)
            pooledC, indicesC = self.pool2D(F.relu(cNew))
            dNew = self.convCD(pooledC)

        elif networkMode == "reconstruction":

            a = self.convBA(b)  # feedback from B upsample
            bNew = self.convCB(self.upsample(c))  # feedback from C
            cNew = self.convDC(self.upsample(d))  # feedback from D
            dNew = d

        elif networkMode == "full":

            if self.betaFB == 0:
                den = torch.sigmoid(self.gammaFw) + torch.sigmoid(self.memory)
                gammaFw = torch.sigmoid(self.gammaFw) / den
                betaBw = 0
            else:
                den = torch.sigmoid(self.gammaFw) + torch.sigmoid(self.betaFB) + torch.sigmoid(self.memory)
                gammaFw = torch.sigmoid(self.gammaFw) / den
                betaBw = torch.sigmoid(self.betaFB) / den

            errorB = nn.functional.mse_loss(self.convBA(b), a)
            reconstructionB = torch.autograd.grad(errorB, b, retain_graph=True)[0]

            bNew = gammaFw * self.convAB(a) + (1 - gammaFw - betaBw) * b + betaBw * self.convCB(self.upsample(c)) - self.alphaRec * self.scalingL1 * batchSize * reconstructionB

            errorC = nn.functional.mse_loss(self.convCB(self.upsample(c)), b)
            reconstructionC = torch.autograd.grad(errorC, c, retain_graph=True)[0]

            pooledB, indicesB = self.pool2D(F.relu(bNew))
            cNew = gammaFw * self.convBC(pooledB) + (1 - gammaFw - betaBw) * c + betaBw * self.convDC(self.upsample(d)) - self.alphaRec * self.scalingL2 * batchSize * reconstructionC

            errorD = nn.functional.mse_loss(self.convDC(self.upsample(d)), c)
            reconstructionD = torch.autograd.grad(errorD, d, retain_graph=True)[0]
            pooledC, indicesC = self.pool2D(F.relu(cNew))

            dNew = gammaFw * self.convCD(pooledC) + (1 - gammaFw) * d - self.alphaRec * self.scalingL3 * batchSize * reconstructionD

        temp3, indices3 = self.pool2D(F.relu(dNew))

        flat3 = temp3.view(-1, temp3.shape[1] * temp3.shape[2] * temp3.shape[3])

        dense1 = F.relu(self.fc1(flat3))

        out = self.fc2(dense1)

        return out, a, bNew, cNew, dNew, reconstructionB, reconstructionC, reconstructionD