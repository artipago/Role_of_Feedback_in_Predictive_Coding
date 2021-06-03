# Role_of_Feedback_in_Predictive_Coding
Code related to the paper "On the role of feedback in visual processing: a predictive coding perspective"

Regarding the files in shallow network/: CEtrainingHyperParameters.py train the forward and backward parameters for the supervised network. RECtrainingFFparameters.py and RECtrainingReconstructionParamters.py train the forward and backward parameters respectively, for the unspervised network. CEtrainingHyperParameters and RECtrainingHyperParameters optimized the hyper-parameters of supervised and unsupervised networks, respectively, for diffrent levels of Gaussian and SaltNpepper noise. All scripts run 10 different initilizations, using CIFAR10. adversarial_attack test the network on adversarial attacks. predCodNet.py contains the predictive coding model.  

Regarding deep_networks/: peff_b0.py and presnet.py contain the networks. ..
