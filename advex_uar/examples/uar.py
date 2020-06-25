import importlib
import os

import pickle
import numpy as np
import torch

from setup_mnist import MNIST
from setup_cifar import CIFAR
from setup_restrictedimagenet import RestrictedImagenet as restImagenet

from advex_uar.eval import Evaluator
from advex_uar.common.pyt_common import *
from advex_uar.common import FlagHolder

import torch.nn as nn


def load_model(network, filters, kernels, strides, paddings, act_fn=torch.nn.functional.relu):
    with open('networks/' + network + '.pkl', 'rb') as file:
        param_vals = pickle.load(file)

    class Model(nn.Module):
        def __init__(self, param_vals, filters, kernels, strides, paddings):
            super(Model, self).__init__()
            self.param_vals = param_vals
            self.filters = filters
            self.kernels = kernels
            self.strides = strides
            self.paddings = paddings

        def forward(self, x):
            x = x / 255
            layers = [x]
            # Define network
            for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
                if type(s) is str:  # Residual
                    s = int(s[1:])
                    W, b = param_vals[i]
                    # HWIO to OIHW
                    W = np.transpose(W, (3, 2, 0, 1))
                    W, b = torch.from_numpy(W), torch.from_numpy(b)
                    W, b = W.cuda(), b.cuda()
                    out_size = 1 + (x.shape[2] - 1) // s
                    pad = (k - s) + s * out_size - x.shape[2]
                    pad_l = pad // 2
                    pad_r = pad - pad_l
                    x = torch.nn.functional.pad(x, (pad_l, pad_r, pad_l, pad_r))
                    x = torch.nn.functional.conv2d(act_fn(x), W, stride=s, bias=b)

                    if x.shape != layers[-2].shape:
                        last_x = layers[-2]
                        scale = int(last_x.shape[3]) // int(x.shape[3])
                        if scale != 1:
                            last_x = torch.nn.functional.avg_pool2d(last_x, kernel_size=scale, stride=scale)
                        last_x = torch.nn.functional.pad(x, (
                        0, 0, 0, 0, int(x.shape[3] - last_x.shape[3]) // 2, int(x.shape[3] - last_x.shape[3]) // 2))
                        x += last_x
                    else:
                        x += layers[-2]
                    layers.append(x)
                elif l == 'pool':
                    out_size = 1 + (x.shape[2] - 1) // s
                    pad = s * (x.shape[2] - 1) + k - out_size
                    pad_l = pad // 2
                    pad_r = pad - pad_l
                    x = torch.nn.functional.pad(x, (pad_l, pad_r, pad_l, pad_r))
                    x = torch.nn.functional.max_pool2d(x, kernel_size=k, stride=s)
                    layers.append(x)
                else:  # Conv
                    W, b = param_vals[i]
                    # HWIO to OIHW
                    W = np.transpose(W, (3, 2, 0, 1))
                    W, b = torch.from_numpy(W), torch.from_numpy(b)
                    W, b = W.cuda(), b.cuda()
                    out_size = 1 + (x.shape[2] - 1) // s
                    pad = (k - s) + s * out_size - x.shape[2]
                    pad_l = pad // 2
                    pad_r = pad - pad_l
                    x = torch.nn.functional.pad(x, (pad_l, pad_r, pad_l, pad_r))
                    if i == 0:
                        x = torch.nn.functional.conv2d(x, W, stride=s, bias=b)
                    else:
                        x = torch.nn.functional.conv2d(act_fn(x), W, stride=s, bias=b)
                    layers.append(x)
            pooled = torch.nn.functional.avg_pool2d(x, kernel_size=int(x.shape[2]), stride=1)

            W = param_vals[-1][0]
            # HWIO to OIHW
            W = np.transpose(W, (3, 2, 0, 1))
            W = torch.from_numpy(W).cuda()
            logits = torch.nn.functional.conv2d(pooled, W, stride=1)
            logits = torch.flatten(logits, start_dim=1)
            return logits

    return Model(param_vals, filters, kernels, strides, paddings)


def run(attack, file_name, filters, kernels, strides, paddings, num_image=200, cifar=False, restimagenet=False):
    model = load_model(file_name, filters, kernels, strides, paddings)

    dataset = 'mnist'
    if cifar:
        dataset = 'cifar-10'
    elif restimagenet:
        dataset = 'imagenet'

    if dataset == 'mnist' or dataset == 'cifar-10':
        if attack == 'jpeg_linf':
            eps = 0.25
        elif attack == 'jpeg_l2':
            eps = 8
        elif attack == 'jpeg_l1':
            eps = 256
        elif attack == 'fog':
            eps = 2048
        elif attack == 'snow':
            eps = 2
        elif attack == 'gabor':
            eps = 400
    elif dataset == 'imagenet':
        if attack == 'jpeg_linf':
            eps = 0.062
        elif attack == 'jpeg_l2':
            eps = 8
        elif attack == 'jpeg_l1':
            eps = 256
        elif attack == 'fog':
            eps = 2048
        elif attack == 'snow':
            eps = 2
        elif attack == 'gabor':
            eps = 400

    attack = get_attack(dataset, attack, eps,
                        200, eps / np.sqrt(200), False)

    classes = 10
    if cifar:
        data = CIFAR()
        x_test = (data.test_data + 0.5) * 255
        y_test = data.test_labels
    elif restimagenet:
        data = restImagenet()
        x_test = data.validation_data
        y_test = data.validation_labels
        classes = 9
    else:
        data = MNIST()
        x_test = (data.test_data + 0.5) * 255
        y_test = data.test_labels

    # NHWC to NCHW
    x_test = np.transpose(x_test, (0, 3, 1, 2))
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.float32)

    evaluator = Evaluator(model=model, attack=attack, data=(x_test, y_test),
                          nb_classes=classes,
                          batch_size=num_image)
    return evaluator.evaluate()
