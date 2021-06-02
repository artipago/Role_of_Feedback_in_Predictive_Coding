import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="2"

import torch
from   datetime import datetime
import torch.optim as optim
import torch.nn as nn
import gc

import numpy as np
from utils import get_cifar_test_dataloader, get_cifarc_dataloader
from presnet import get_cifar_resnet18, PResNet18V3NSameHP, PResNet18V3NSeparateHP

from torch.utils.tensorboard import SummaryWriter
########################
## GLOBAL CONFIGURATIONS
########################
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 40

SAME_PARAM = False           # to use the same parameters for all pcoders or not
FF_START = True             # to start from feedforward initialization
MAX_TIMESTEP = 5

#tensorboard log dir
LOG_DIR = '../tensorboards/' + f'runs_train_hps_{MAX_TIMESTEP}ts'
if FF_START:
    LOG_DIR += '_ff_start'
if not SAME_PARAM:
    LOG_DIR += '_sep'
LOG_DIR += '_cifarc'

TASK_NAME = 'presnet18_v3n'
WEIGHT_PATTERN = '../weights/PResNet18/presnet18n_pretrained*.pth'

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
            
        print(f"Training Epoch: {epoch} [{batch_index * 128 + len(images)}/{len(dataloader.dataset)}]\tLoss: {loss.item():0.4f}\tLR: {optimizer.param_groups[0]['lr']:0.6f}")
        for tt in range(timesteps+1):
            print(f'{ttloss[tt]:0.4f}\t', end='')
        print()
        if writer is not None:
            writer.add_scalar(f"TrainingLoss/CE", loss.item(), (epoch-1)*len(dataloader) + batch_index)

def load_presnet18(pre_trained_resnet, weight_pattern, build_graph, random_init, ff_multiplier, fb_multiplier, er_multiplier, same_param, device='cuda:0'):
    # loading PResNet with pretrained resnet on CIFAR100
    resnet = get_cifar_resnet18()
    resnet.load_state_dict(torch.load(pre_trained_resnet, map_location='cpu'))

    if same_param:
        presnet18 = PResNet18V3NSameHP(resnet, build_graph=build_graph, random_init=random_init, ff_multiplier=ff_multiplier, fb_multiplier=fb_multiplier, er_multiplier=er_multiplier)
    else:
        presnet18 = PResNet18V3NSeparateHP(resnet, build_graph=build_graph, random_init=random_init, ff_multiplier=ff_multiplier, fb_multiplier=fb_multiplier, er_multiplier=er_multiplier)

    # block 1
    pc_dict = torch.load(weight_pattern.replace('*','_pc1'), map_location='cpu')
    presnet18.pcoder1.load_state_dict(pc_dict)

    # block 2
    pc_dict = torch.load(weight_pattern.replace('*','_pc2'), map_location='cpu')
    presnet18.pcoder2.load_state_dict(pc_dict)

    # block 3
    pc_dict = torch.load(weight_pattern.replace('*','_pc3'), map_location='cpu')
    presnet18.pcoder3.load_state_dict(pc_dict)

    # block 4
    pc_dict = torch.load(weight_pattern.replace('*','_pc4'), map_location='cpu')
    presnet18.pcoder4.load_state_dict(pc_dict)

    # block 5
    pc_dict = torch.load(weight_pattern.replace('*','_pc5'), map_location='cpu')
    presnet18.pcoder5.load_state_dict(pc_dict)

    presnet18.eval()
    presnet18.to(device)
    return presnet18

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
            "speckle_noise",
            "pixelate",
            "saturate",
            "brightness",
            "contrast",
            "defocus_blur",
            "elastic_transform",
            "fog",
            "frost",
            "gaussian_blur",
            "glass_blur",
            "jpeg_compression",
            "motion_blur",
            "shot_noise",
            "snow",
            "spatter",
            "zoom_blur", "none"][:]

###################################################
### Training
###################################################

for noise_type in all_noises:
    noise_levels = range(5,0,-1)
    if noise_type == "none":
        noise_levels = [0]
    for rep in range(1):
        for noise_level in noise_levels:
            
            cifar100_testloader = get_cifar_test_dataloader(
                CIFAR100_TRAIN_MEAN,
                CIFAR100_TRAIN_STD,
                num_workers=4,
                batch_size=128,
                shuffle=False,
                cifar10=False,
                root='../datasets'
            )

            if noise_type == "none":
                cifar100c_trainloader = get_cifar_test_dataloader(
                    CIFAR100_TRAIN_MEAN,
                    CIFAR100_TRAIN_STD,
                    num_workers=4,
                    batch_size=128,
                    shuffle=True,
                    cifar10=False,
                    root='../datasets'
                )
            else:
                cifar100c_trainloader = get_cifarc_dataloader(
                    CIFAR100_TRAIN_MEAN,
                    CIFAR100_TRAIN_STD,
                    num_workers=4,
                    batch_size=128,
                    shuffle=True,
                    root='../datasets/CIFAR-100-C',
                    noise_level=noise_level,
                    noise_type=noise_type
                )
                
            sumwriter = SummaryWriter(f'{LOG_DIR}/net_{TASK_NAME}_type_{noise_type}_lvl_{noise_level}', filename_suffix=f'_{noise_type}_{rep}_{noise_level}')

            backward_weight_patter = WEIGHT_PATTERN

            # feedforward presnet for baseline
            presnet18_fw = load_presnet18('../weights/PResNet18/resnet18-193-best.pth',
                backward_weight_patter,
                build_graph=False, random_init=(not FF_START), ff_multiplier=1.0, fb_multiplier=0.0, er_multiplier=0.0, same_param=SAME_PARAM, device='cuda:0')
            
            loss_function = nn.CrossEntropyLoss()
            evaluate(presnet18_fw, 0, cifar100c_trainloader, timesteps=1, writer=sumwriter, tag='FeedForward')
            del presnet18_fw
            gc.collect()

            presnet18 = load_presnet18('../weights/PResNet18/resnet18-193-best.pth',
                backward_weight_patter,
                build_graph=True, random_init=(not FF_START), ff_multiplier=0.33, fb_multiplier=0.33, er_multiplier=0.0, same_param=SAME_PARAM, device='cuda:0')

            loss_function = nn.CrossEntropyLoss()
            hyperparams = [*presnet18.get_hyperparameters()]
            if SAME_PARAM:
                optimizer = optim.Adam([
                    {'params': hyperparams[:-1], 'lr':0.01},
                    {'params': hyperparams[-1:], 'lr':0.0001}], weight_decay=0.00001)
            else:
                fffbmem_hp = []
                erm_hp = []
                for pc in range(presnet18.number_of_pcoders):
                    fffbmem_hp.extend(hyperparams[pc*4:pc*4+3])
                    erm_hp.append(hyperparams[pc*4+3])
                optimizer = optim.Adam([
                    {'params': fffbmem_hp, 'lr':0.01},
                    {'params': erm_hp, 'lr':0.0001}], weight_decay=0.00001)

            log_hyper_parameters(presnet18, 0, sumwriter, same_param=SAME_PARAM)
            hps = presnet18.get_hyperparameters_values()
            print(hps)

            evaluate(presnet18, 0, cifar100c_trainloader, timesteps=MAX_TIMESTEP, writer=sumwriter, tag='Noisy')
            
            for epoch in range(1, EPOCH+1):
                train(presnet18, epoch, cifar100c_trainloader, timesteps=MAX_TIMESTEP, writer=sumwriter)
                log_hyper_parameters(presnet18, epoch, sumwriter, same_param=SAME_PARAM)

                hps = presnet18.get_hyperparameters_values()
                print(hps)

                evaluate(presnet18, epoch, cifar100c_trainloader, timesteps=MAX_TIMESTEP, writer=sumwriter, tag='Noisy')

            evaluate(presnet18, epoch, cifar100_testloader, timesteps=MAX_TIMESTEP, writer=sumwriter, tag='Clean')
            
            sumwriter.close()

            del presnet18
            gc.collect()
