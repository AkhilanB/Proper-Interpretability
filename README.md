# Interpretability

setup_*.py contains code to import datasets. Currently, there is code to import mnist, cifar, gtsrb and tinyimagenet.

load_model.py contains a function `load_model` that takes a model architecture specification and a model file, and loads it into tensorflow. The function returns a `Model` object and contains functions that compute prediction output, CAM, GradCAM++ and IG.

train.py contains functions to train networks with different training methods. Each function trains a network with a different method. Running a training function will save a trained model in `networks\`. The bottom of the file contains scripts to train all networks.

eval.py contains functions to evaluate networks. Each functuon corresponds to a different experiment. Functions either return numerical results or generates a plot in `gen_images\`.  Currently implemented experiments are attack: (one point tradeoff between eps and discrepancy, ISAA statistics), defense: (pgd accuracies, discrepancy AAI, top-k CAM AAI, top-k GradCAM++ AAI, top-k IG AAI). The bottom of the file contains scripts to run all evaluations.



