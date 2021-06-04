# Role_of_Feedback_in_Predictive_Coding
Code related to the paper "On the role of feedback in visual processing: a predictive coding perspective"

Here is the structure of the repository:

```
root
|
├── shallow network
|   |
│   └── CEtrainingHyperParameters.py
|   |   train the forward and backward parameters for the supervised network.
│   └── RECtrainingFFparameters.py
|   |   train the forward parameters for the unsupervised network.
│   └── RECtrainingReconstructionParamters.py
|   |   train the backward parameters for the unsupervised network.
│   └── CEtrainingHyperParameters.py
|   |   optimize the hyper-parameters of supervised networks for different levels of Gaussian and SaltNpepper noise. All scripts run 10 different initilizations, using CIFAR10.
|   └── RECtrainingHyperParameters.py
|   |   optimize the hyper-parameters of unsupervised networks for different levels of Gaussian and SaltNpepper noise. All scripts run 10 different initilizations, using CIFAR10.
│   └── adversarial_attack.py
|   |   test the network on adversarial attacks.
│   └── predCodNet.py
|       contains the predictive coding model. 
|
├── deep_networks
|   |
│   └── scripts
|       |
│       └── peff_b0.py
|       |   architecture of PEffNetB0
│       └── presnet.py
|       |   architecture of PResNet18
│       └── train_feedback_weights.py
|       |   general script to train feedback weights of predictive networks
│       └── train_pefbo_hps_imagenet.py
|       |   training hyper-parameters of PEffNetB0 on noisy ImageNet
│       └── train_presnet.py
|       |   training feedback weights of PResNet18 on CIFAR100
│       └── train_presnet_hps_cifarc.py
|       |   training hyper-parameters of PResNet18 on CIFAR100-C
│       └── train_resnet.py
|       |   training modified resnet18 (feedforward weights) on CIFAR100
│       └── utils.py
            some utility functions used in other scripts

```
