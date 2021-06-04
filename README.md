# Role_of_Feedback_in_Predictive_Coding
Code related to the paper "On the role of feedback in visual processing: a predictive coding perspective"

### shallow network/ subfolder: 
- CEtrainingHyperParameters.py: train the forward and backward parameters for the supervised network. 
- RECtrainingFFparameters.py and RECtrainingReconstructionParamters.py: train the forward and backward parameters respectively, for the unsupervised network. 
- CEtrainingHyperParameters.py and RECtrainingHyperParameters.py: optimize the hyper-parameters of supervised and unsupervised networks, respectively, for different levels of Gaussian and SaltNpepper noise. All scripts run 10 different initilizations, using CIFAR10. 
- adversarial_attack.py: test the network on adversarial attacks. 
- predCodNet.py: contains the predictive coding model. 
 
### deep_networks/ subfolder: 
- peff_b0.py and presnet.py contain the networks. 
- ..
