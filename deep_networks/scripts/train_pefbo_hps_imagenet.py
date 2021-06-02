import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="0"

import torch
import torchvision.transforms as transforms
from   torchvision.datasets import ImageNet
from   datetime import datetime
import torch.optim as optim
import torch.nn as nn
import gc

import numpy as np
from   utils import AddGaussianNoise, AddSaltPepperNoise
from   timm.models import efficientnet_b0
from   peff_b0 import PEffN_b0SeparateHP_V1

from torch.utils.tensorboard import SummaryWriter

########################
## GLOBAL CONFIGURATIONS
########################
TRAIN_MEAN = [0.485, 0.456, 0.406]
TRAIN_STD  = [0.229, 0.224, 0.225]
dataset_root = '../datasets/imagenet'

#total training epoches
EPOCH = 8

SAME_PARAM = False           # to use the same parameters for all pcoders or not
FF_START = True             # to start from feedforward initialization
MAX_TIMESTEP = 5

#tensorboard log dir
LOG_DIR = '../tensorboards/' + f'runs_train_hps_{MAX_TIMESTEP}ts'
if FF_START:
    LOG_DIR += '_ff_start'
if not SAME_PARAM:
    LOG_DIR += '_sep'
LOG_DIR += '_imagenet'

TASK_NAME = 'pefb0_v1n'
WEIGHT_PATTERN_N = '../weights/PEffNetB0/pnetn_pretrained_pc*.pth'

#time of we run the script
TIME_NOW = datetime.now().isoformat()

########################
########################

def evaluate(net, epoch, dataloader, timesteps, writer=None, tag='Clean'):
    test_loss = np.zeros((timesteps+1,))
    correct   = np.zeros((timesteps+1,))
    for (images, labels) in dataloader:
        images = images.cuda()
        labels = labels.cuda()
        
        with torch.no_grad():
            for tt in range(timesteps+1):
                if tt == 0:
                    outputs = net(images)
                else:
                    outputs = net()
            
                loss = loss_function(outputs, labels)
                test_loss[tt] += loss.item()
                _, preds = outputs.max(1)
                correct[tt] += preds.eq(labels).sum()

    print()
    for tt in range(timesteps+1):
        test_loss[tt] /= len(dataloader.dataset)
        correct[tt] /= len(dataloader.dataset)
        print('Test set t = {:02d}: Average loss: {:.4f}, Accuracy: {:.4f}'.format(
            tt,
            test_loss[tt],
            correct[tt]
        ))
        if writer is not None:
            writer.add_scalar(f"{tag}Perf/Epoch#{epoch}", correct[tt], tt)
    print()

def train(net, epoch, dataloader, timesteps, writer=None):
    for batch_index, (images, labels) in enumerate(dataloader):
        net.reset()

        labels = labels.cuda()
        images = images.cuda()

        ttloss = np.zeros((timesteps+1))
        optimizer.zero_grad()

        for tt in range(timesteps+1):
            if tt == 0:
                outputs = net(images)
                loss = loss_function(outputs, labels)
                ttloss[tt] = loss.item()
            else:
                outputs = net()
                current_loss = loss_function(outputs, labels)
                ttloss[tt] = current_loss.item()
                loss += current_loss
        
        loss.backward()
        optimizer.step()
        net.update_hyperparameters()
            
        print(f"Training Epoch: {epoch} [{batch_index * 16 + len(images)}/{len(dataloader.dataset)}]\tLoss: {loss.item():0.4f}\tLR: {optimizer.param_groups[0]['lr']:0.6f}")
        for tt in range(timesteps+1):
            print(f'{ttloss[tt]:0.4f}\t', end='')
        print()
        if writer is not None:
            writer.add_scalar(f"TrainingLoss/CE", loss.item(), (epoch-1)*len(dataloader) + batch_index)

def load_pnet(net, weight_pattern, build_graph, random_init, ff_multiplier, fb_multiplier, er_multiplier, same_param, device='cuda:0'):
    if same_param:
        raise Exception('Not implemented!')
    else:
        pnet = PEffN_b0SeparateHP_V1(net, build_graph=build_graph, random_init=random_init, ff_multiplier=ff_multiplier, fb_multiplier=fb_multiplier, er_multiplier=er_multiplier)


    for pc in range(pnet.number_of_pcoders):
        pc_dict = torch.load(weight_pattern.replace('*',f'{pc+1}'), map_location='cpu')
        if 'C_sqrt' not in pc_dict:
            pc_dict['C_sqrt'] = torch.tensor(-1, dtype=torch.float)
        getattr(pnet, f'pcoder{pc+1}').load_state_dict(pc_dict)

    pnet.eval()
    pnet.to(device)
    return pnet

def log_hyper_parameters(net, epoch, sumwriter, same_param=True):
    if same_param:
        sumwriter.add_scalar(f"HyperparamRaw/feedforward", getattr(net,f'ff_part').item(), epoch)
        sumwriter.add_scalar(f"HyperparamRaw/feedback",    getattr(net,f'fb_part').item(), epoch)
        sumwriter.add_scalar(f"HyperparamRaw/error",       getattr(net,f'errorm').item(), epoch)
        sumwriter.add_scalar(f"HyperparamRaw/memory",      getattr(net,f'mem_part').item(), epoch)

        sumwriter.add_scalar(f"Hyperparam/feedforward", getattr(net,f'ffm').item(), epoch)
        sumwriter.add_scalar(f"Hyperparam/feedback",    getattr(net,f'fbm').item(), epoch)
        sumwriter.add_scalar(f"Hyperparam/error",       getattr(net,f'erm').item(), epoch)
        sumwriter.add_scalar(f"Hyperparam/memory",      1-getattr(net,f'ffm').item()-getattr(net,f'fbm').item(), epoch)
    else:
        for i in range(1, net.number_of_pcoders+1):
            sumwriter.add_scalar(f"Hyperparam/pcoder{i}_feedforward", getattr(net,f'ffm{i}').item(), epoch)
            if i < net.number_of_pcoders:
                sumwriter.add_scalar(f"Hyperparam/pcoder{i}_feedback", getattr(net,f'fbm{i}').item(), epoch)
            else:
                sumwriter.add_scalar(f"Hyperparam/pcoder{i}_feedback", 0, epoch)
            sumwriter.add_scalar(f"Hyperparam/pcoder{i}_error", getattr(net,f'erm{i}').item(), epoch)
            if i < net.number_of_pcoders:
                sumwriter.add_scalar(f"Hyperparam/pcoder{i}_memory",      1-getattr(net,f'ffm{i}').item()-getattr(net,f'fbm{i}').item(), epoch)
            else:
                sumwriter.add_scalar(f"Hyperparam/pcoder{i}_memory",      1-getattr(net,f'ffm{i}').item(), epoch)

all_noises = [
            "gaussian_noise",
            "impulse_noise",
            "none"]
noise_gens = [
    [
        AddGaussianNoise(std=0.50),
        AddGaussianNoise(std=0.75),
        AddGaussianNoise(std=1.00),
        AddGaussianNoise(std=1.25),
        AddGaussianNoise(std=1.50),
    ],
    [
        AddSaltPepperNoise(probability=0.05),
        AddSaltPepperNoise(probability=0.1),
        AddSaltPepperNoise(probability=0.15),
        AddSaltPepperNoise(probability=0.2),
        AddSaltPepperNoise(probability=0.3),
    ],
    [None],
]

for nt_idx, noise_type in enumerate(all_noises):
    for ng_idx, noise_gen in enumerate(noise_gens[nt_idx]):
        print(noise_gen)
        start = datetime.now()
        
        noise_level = 0
        transform_clean = [
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
        ]
        transform_noise = transform_clean[:]

        transform_clean.append(transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD))
        transform_noise.append(transforms.Normalize(mean=TRAIN_MEAN, std=TRAIN_STD))

        if noise_gen is not None:
            noise_level = ng_idx + 1
            transform_noise.append(noise_gen)

        clean_ds     = ImageNet(dataset_root, split='val', download=False, transform=transforms.Compose(transform_clean))
        clean_loader = torch.utils.data.DataLoader(clean_ds,  batch_size=16, shuffle=False, drop_last=False, num_workers=8)

        noise_ds     = ImageNet(dataset_root, split='val', download=False, transform=transforms.Compose(transform_noise))
        noise_loader = torch.utils.data.DataLoader(noise_ds,  batch_size=16, shuffle=True, drop_last=False, num_workers=8)

            
        sumwriter = SummaryWriter(f'{LOG_DIR}/net_{TASK_NAME}_type_{noise_type}_lvl_{noise_level}', filename_suffix=f'_{noise_type}_{noise_level}')
        
        backward_weight_patter = WEIGHT_PATTERN_N

        # feedforward for baseline
        net = efficientnet_b0(pretrained=True)
        pnet_fw = load_pnet(net, backward_weight_patter,
            build_graph=False, random_init=(not FF_START), ff_multiplier=1.0, fb_multiplier=0.0, er_multiplier=0.0, same_param=SAME_PARAM, device='cuda:0')
        
        loss_function = nn.CrossEntropyLoss()
        evaluate(pnet_fw, 0, noise_loader, timesteps=1, writer=sumwriter, tag='FeedForward')
        print(datetime.now() - start)
        del pnet_fw
        gc.collect()

        # train hps
        net = efficientnet_b0(pretrained=True)
        pnet = load_pnet(net, backward_weight_patter,
            build_graph=True, random_init=(not FF_START), ff_multiplier=0.33, fb_multiplier=0.33, er_multiplier=0.0, same_param=SAME_PARAM, device='cuda:0')

        loss_function = nn.CrossEntropyLoss()
        hyperparams = [*pnet.get_hyperparameters()]
        if SAME_PARAM:
            optimizer = optim.Adam([
                {'params': hyperparams[:-1], 'lr':0.01},
                {'params': hyperparams[-1:], 'lr':0.0001}], weight_decay=0.00001)
        else:
            fffbmem_hp = []
            erm_hp = []
            for pc in range(pnet.number_of_pcoders):
                fffbmem_hp.extend(hyperparams[pc*4:pc*4+3])
                erm_hp.append(hyperparams[pc*4+3])
            optimizer = optim.Adam([
                {'params': fffbmem_hp, 'lr':0.01},
                {'params': erm_hp, 'lr':0.0001}], weight_decay=0.00001)

        log_hyper_parameters(pnet, 0, sumwriter, same_param=SAME_PARAM)
        hps = pnet.get_hyperparameters_values()
        print(hps)

        evaluate(pnet, 0, noise_loader, timesteps=MAX_TIMESTEP, writer=sumwriter, tag='Noisy')
        print(datetime.now() - start)
        for epoch in range(1, EPOCH+1):
            train(pnet, epoch, noise_loader, timesteps=MAX_TIMESTEP, writer=sumwriter)
            print(datetime.now() - start)
            log_hyper_parameters(pnet, epoch, sumwriter, same_param=SAME_PARAM)

            hps = pnet.get_hyperparameters_values()
            print(hps)

            evaluate(pnet, epoch, noise_loader, timesteps=MAX_TIMESTEP, writer=sumwriter, tag='Noisy')
            print(datetime.now() - start)

        evaluate(pnet, epoch, clean_loader, timesteps=MAX_TIMESTEP, writer=sumwriter, tag='Clean')
        
        sumwriter.close()

        del pnet
        gc.collect()
        print(datetime.now() - start)
