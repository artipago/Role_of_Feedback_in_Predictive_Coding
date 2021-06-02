#########################
# In this script we train
# presnet on CIFAR100
# we use the pretrained
# model and only train
# feedback connections.
#########################

import torch
from   datetime import datetime
import torch.optim as optim
import torch.nn as nn

from utils import get_cifar_test_dataloader, get_cifar_training_dataloader
from presnet import get_cifar_resnet18, PResNet18V3NSeparateHP
from torch.utils.tensorboard import SummaryWriter

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"]="1"

########################
## GLOBAL CONFIGURATIONS
########################
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD  = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

CHECKPOINT_PATH = 'checkpoint'

#total training epoches
EPOCH = 200
MILESTONES = [60, 120, 160]

#initial learning rate
INIT_LR = 0.1

#tensorboard log dir
LOG_DIR   = '../tensorboards/runs_train_feedbacks'
TASK_NAME = 'resnet_v3n'
if not os.path.exists(TASK_NAME):
    os.mkdir(TASK_NAME)

#save weights file per SAVE_EPOCH epoch
SAVE_EPOCH = 10

#time of we run the script
TIME_NOW = datetime.now().isoformat()

########################
########################

def train_pcoders(net, epoch, writer):
    net.train()
    for batch_index, (images, _) in enumerate(cifar100_training_loader):
        net.reset()
        images = images.cuda()
        optimizer.zero_grad()
        outputs = net(images)
        for i in range(5):
            # print(outputs[i][1].shape)
            if i == 0:
                a = loss_function(net.pcoder1.prd, images)
                loss = a
            else:
                pcoder_pre = getattr(net, f"pcoder{i}")
                pcoder_cur = getattr(net, f"pcoder{i+1}")
                a = loss_function(pcoder_cur.prd, pcoder_pre.rep)
                loss += a
            sumwriter.add_scalar(f"MSE Train/PCoder{i+1}", a.item(), epoch * len(cifar100_training_loader) + batch_index)
        # return
        loss.backward()
        optimizer.step()

        print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
            loss.item(),
            optimizer.param_groups[0]['lr'],
            epoch=epoch,
            trained_samples=batch_index * 128 + len(images),
            total_samples=len(cifar100_training_loader.dataset)
        ))
        sumwriter.add_scalar(f"MSE Train/Sum", loss.item(), epoch * len(cifar100_training_loader) + batch_index)

def test_pcoders(net, epoch, writer):
    net.eval()
    final_loss = [0 for i in range(5)]
    for batch_index, (images, _) in enumerate(cifar100_test_loader):
        net.reset()
        images = images.cuda()
        with torch.no_grad():
            outputs = net(images)
        for i in range(5):
            if i == 0:
                final_loss[i] += loss_function(net.pcoder1.prd, images).item()
            else:
                pcoder_pre = getattr(net, f"pcoder{i}")
                pcoder_cur = getattr(net, f"pcoder{i+1}")
                final_loss[i] += loss_function(pcoder_cur.prd, pcoder_pre.rep).item()

    loss_sum = 0
    for i in range(5):
        final_loss[i] /= len(cifar100_test_loader)
        loss_sum += final_loss[i]
        sumwriter.add_scalar(f"MSE Test/PCoder{i+1}", final_loss[i], epoch * len(cifar100_test_loader))
    sumwriter.add_scalar(f"MSE Test/Sum", loss_sum, epoch * len(cifar100_test_loader))

    print('Training Epoch: {epoch} [{trained_samples}/{total_samples}]\tLoss: {:0.4f}\tLR: {:0.6f}'.format(
        loss_sum,
        optimizer.param_groups[0]['lr'],
        epoch=epoch,
        trained_samples=batch_index * 128 + len(images),
        total_samples=len(cifar100_test_loader.dataset)
    ))

cifar100_training_loader = get_cifar_training_dataloader(
    CIFAR100_TRAIN_MEAN,
    CIFAR100_TRAIN_STD,
    num_workers=4,
    batch_size=128,
    shuffle=True,
    root='../datasets'
)

cifar100_test_loader = get_cifar_test_dataloader(
    CIFAR100_TRAIN_MEAN,
    CIFAR100_TRAIN_STD,
    num_workers=4,
    batch_size=128,
    shuffle=False,
    root='../datasets'
)

net = get_cifar_resnet18()
net.load_state_dict(torch.load('../weights/PResNet18/resnet18-193-best.pth', map_location='cpu'))
pnet = PResNet18V3NSeparateHP(net, build_graph=True, random_init=False)
pnet.cuda()

print(pnet)
sumwriter = SummaryWriter(f'{LOG_DIR}/{TASK_NAME}', filename_suffix=f'')


loss_function = nn.MSELoss()
optimizer = optim.AdamW([{'params': getattr(pnet, f'pcoder{i}').pmodule.parameters()} for i in range(1, pnet.number_of_pcoders+1)], lr=0.001, weight_decay=5e-4)

for epoch in range(1, 50):
    train_pcoders(pnet, epoch, sumwriter)
    test_pcoders(pnet, epoch, sumwriter)

    for i in range(1, pnet.number_of_pcoders + 1):
        torch.save(getattr(pnet, f'pcoder{i}').state_dict(), f'{TASK_NAME}/presnet_v3n_pretrained_pc{i}_{epoch:03d}.pth')


