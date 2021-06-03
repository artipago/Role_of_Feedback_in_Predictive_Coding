import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
import pickle
from predCodNet import *





import foolbox ; print (foolbox.__version__)
from datetime import datetime

import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

device = torch.device('cuda:0')

torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)


random_seed = 42
batchsize = 128
path_to_weights = "./pcNetREC_FF_Rec_E19_I0.pth"

TARGETED = True

           ## gammaMem, alphaRec, betaFB, memory
params_list = [
              [1.0, 0.0, 0.0, 0.0],
              [0.3, 0.0, 0.3, 0.4],
              [0.3, 0.0, 0.5, 0.2],
              [1.0, 1.0, 0.0, 0.0],  
              [1.0, 2.0, 0.0, 0.0],
    
              [0.3, 1.0, 0.3, 0.4],
              [0.3, 1.0, 0.5, 0.2],
    
              [0.8, 0.0, 0.2, 0.0],
              [0.6, 0.0, 0.4, 0.0],
            
]

featuresB = 12
featuresC = 18
featuresD = 24
width = 32
height = 32




########################################################################################
##                                  get data loaders
########################################################################################
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.507, 0.486, 0.440), (0.267, 0.256, 0.276))])
testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=8,drop_last=True)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

########################################################################################
##                                 Definitions
########################################################################################
class picklable_adv():

    def __init__(self,x):
        self.distance = x._distance
        self.distance_value = x.distance.value
        self.unperturbed = x.unperturbed
        self.perturbed = x.perturbed
        self._total_prediction_calls = x._total_prediction_calls
        self.original_class = x.original_class
        self.target_class = x.target_class

        
class deeper_model(torch.nn.Module):

    def __init__(self,t,predCodNet,batchsize):
        super(deeper_andrea,self).__init__()
        self.predCodNet = predCodNet.to(device)
        self.t = t
        self.batchSize = batchsize

    def forward(self,x):

        bTempG = torch.randn(self.batchSize, 12, 32, 32).to(device)   #changed this to randn
        cTempG = torch.randn(self.batchSize, 18, 16, 16).to(device)
        dTempG = torch.randn(self.batchSize, 24, 8, 8).to(device)
        
        bTempG.requires_grad = cTempG.requires_grad = dTempG.requires_grad = True

        for _ in range(self.t):
            out, a, bTempG, cTempG, dTempG, errorB, errorC, errorD = self.predCodNet(x, bTempG, cTempG, dTempG, 'full')


        return out




def get_flag(sample_image,sample_target,allmodels):
    
    
    mean,std = (0.507, 0.486, 0.440), (0.267, 0.256, 0.276)
    
    x_image = torch.zeros(sample_image.shape)
    for i in range(3):
        x_image[:,i,:,:] = (sample_image[:,i,:,:] - torch.ones((1,32,32))*mean[i])/std[i]
    
    
    tlist = []
    
    for i in range(len(allmodels)):
        a = torch.max(allmodels[i](torch.Tensor(x_image).cuda()),1).indices
        b = sample_target.cuda()
        
        tlist.append(a==b)

    return all(tlist)






def get_correct_X(allmodels):
    
    total_images = 200
    
    transform = transforms.Compose([transforms.ToTensor()]) #, transforms.Normalize((0.507, 0.486, 0.440), (0.267, 0.256, 0.276))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=True, num_workers=8,drop_last=True)
    
    correct_X_test = []
    
    for ind,data in enumerate(testloader):
        if len(correct_X_test) < total_images:
            inputs,labels = data
            inputs.to(device)
            labels.to(device)

            if get_flag(inputs,labels,allmodels) == True:
                correct_X_test.append((inputs,labels))
                print (len(correct_X_test),'====>',ind)

        else:
            break
        
    return correct_X_test





def test_accuracy(xmodel):
    
    ##Sanity test...accuracy on test set...
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.507, 0.486, 0.440), (0.267, 0.256, 0.276))])
    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batchsize, shuffle=False, num_workers=8,drop_last=True)
    corrs = 0.
    total = 0.
    for (inputs,labels) in testloader:
        out = xmodel(torch.Tensor(inputs).cuda())
        _,preds= torch.max(out,1)
        corrs += torch.sum(preds == labels.cuda().data)
        total += labels.size(0)
    corrs = (float(corrs)*100.) / float(total)
#     print ('The accuracy is:', (float(corrs)*100.) / float(total))
    return corrs
        
checkpointPhase = torch.load(path_to_weights)


########################################################################################
##                             Get all the images
########################################################################################
all_models = []
for _param in params_list:

    pcnet = predCodNet(featB=12, featC=18, featD=24, gammaFw=_param[0],alphaRec=_param[1], betaFB=_param[2],memory=_param[3])
    pcnet.to(device)

    model = deeper_model(t=10,predCodNet=pcnet.eval(),batchsize=1)
    model.predCodNet.load_state_dict(checkpointPhase["module"])
    model.predCodNet.gammaFw.requires_grad = True
    model.predCodNet.alphaRec.requires_grad = True
    model.predCodNet.betaFB.requires_grad = True

    model.to(device)
    model.eval()
#     print (_param,'---',test_accuracy(model))
#     test_accuracy(model)
    all_models.append(model)
    

tstart = datetime.now()
correct_x_test = get_correct_X(all_models)
print (datetime.now() - tstart)

########################################################################################
##                             Attacks
########################################################################################
preprocessing = dict(mean=(0.507, 0.486, 0.440),std=(0.267, 0.256, 0.276),axis=-3)
fmodels = [foolbox.models.PyTorchModel(x,bounds=(0,1),num_classes=(10),preprocessing=preprocessing) for x in all_models]

del all_models

adversarial_dict = {}
targeted_attacks_list = {}
index = 0
for ind,data in enumerate(correct_x_test):
    image,label  = data

    if ind == len(correct_x_test)-1:
        target_image,target_class = correct_x_test[0]
    else:
        target_image,target_class = correct_x_test[ind+1]
    target_class = (label.item() + 5)%10
    
    if image.shape == (3,32,32):
        image.unsqueeze_(0)
    if target_image.shape == (3,32,32):
        target_image.unsqueeze_(0)





    print ("\n\n\nImage:",index)
    print (f"Label: {label.numpy()} =====> Target: {target_class}")

    targeted_attacks_list[index] = []
    adv_val = []
    for j in range(len(fmodels)):

        if TARGETED:
              criterion = foolbox.criteria.TargetClass(target_class)
              attack = foolbox.attacks.RandomProjectedGradientDescent(fmodels[j],criterion=criterion)
          else:
              print ('This attacks are not supported!')
              raise ValueError
          #foolbox2.3
          numpy_target_image = target_image.squeeze_(0).cpu().numpy()
          numpy_image = image.cpu().numpy()


          tstart = datetime.now()
          print ('starting attack...')
          adv_image = attack(inputs=numpy_image,labels=label.numpy(),
                             unpack=False)


          print ("Time taken:",datetime.now() - tstart)
          print (f"---Model{j}:{adv_image[0].distance.value}")

          targeted_attacks_list[index].append((j,picklable_adv(adv_image[0])))

      index = index + 1

########################################################################################
##                             Analyse
########################################################################################
param_based_dict = {}
for p_ind in range(len(params_list)):
    print (params_list[p_ind])
    param_based_dict[str(params_list[p_ind])] = []
    for image in targeted_attacks_list.keys():
        param_based_dict[str(params_list[p_ind])].append(targeted_attacks_list[image][p_ind][1].distance_value)

        
np.random.seed(17)
import scipy
from scipy import stats
from scipy.interpolate import make_interp_spline,BSpline


def mean_confidence_interval(data,confidence=0.95):

    a = 1.0 * np.array(data)
    n = len(a)
    m = np.mean(a)
    std = np.std(a)
    # se = scipy.stats.sem(a)
    h = std*scipy.stats.t.ppf((1 + confidence)/2., n-1)
    # return  m, m-h,m+h,h
    return m,std,h
NUMBER_BOOTSTRAPS = 1000


dict_of_medians = {}
# getting the bootstrap estimates for each timestep
for key in param_based_dict.keys():
    #bootstrap
    list_of_medians = []    
    for _ in range(NUMBER_BOOTSTRAPS):
#         x = np.random.choice (param_based_dict[key],size=len(param_based_dict[key]),replace=True)
        x = np.random.choice (param_based_dict[key],size=1000,replace=True)

        list_of_medians.append(np.median(x))        
    dict_of_medians[key] = list_of_medians


confidence_intervals = []
ydata = []
for key in param_based_dict.keys():
    mean,error,_ = mean_confidence_interval(dict_of_medians[key])
    ydata.append(mean)
    confidence_intervals.append(error)


print (ydata)
#plot 1 
xdata = [1,2,3,4,5,6,7]   #[0,2,4,6,10]
ydata = [ydata[i] for i in range(len(xdata))] #plot only timepoints you wish to plot
confidence_intervals = [confidence_intervals[i] for i in range(len(xdata))]


fp = {'fontsize':16}
plt.style.use('ggplot')
plt.figure(figsize=(7,7))
for i in range(len(xdata)):
    plt.bar(xdata[i],ydata[i],label=list(param_based_dict.keys())[i],yerr=confidence_intervals[i])
plt.title('RECmodel : BIM Attack Linf',**fp)

plt.xticks([])
plt.ylabel('Adversarial perturbation',**fp)



from matplotlib.lines import Line2D

colors = ['black']*7
lines = [Line2D([0], [0], color=c, linewidth=1) for c in colors]


labels = param_based_dict.keys()
plt.legend(bbox_to_anchor=(1,1),title='[ff,pc,fb,memory]',title_fontsize=14)

plt.show()
