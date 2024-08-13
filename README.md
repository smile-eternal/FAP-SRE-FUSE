
This is the official implementation of Focus Affinity Perception and Super-Resolution Embedding for Multi-Focus Image Fusion, accepted in TNNLS'23.


## Dataset
For the training of FAPN, we generate the training set through a synthetic way.
For the training of SRN, being like existing super-resolution methods, we regard the Synthetic Training Data  set as the training set

## Train and Test
Please follow the training and evaluation steps:
```
first run train_net.py to train FAPN
then run train.py to train SRN
run test.py to evaluation our model
```
