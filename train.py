"""
train.py
Trains Network with Interpretability-aware training
Copyright (C) 2020, Akhilan Boopathy <akhilan@mit.edu>
                    Sijia Liu <Sijia.Liu@ibm.com>
                    Gaoyuan Zhang <Gaoyuan.Zhang@ibm.com>
                    Cynthia Liu <cynliu98@mit.edu>
                    Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
                    Shiyu Chang <Shiyu.Chang@ibm.com>
                    Luca Daniel <luca@mit.edu>
"""


import os

# Disable tf warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
from setup_mnist import MNIST
from setup_cifar import CIFAR
from setup_restrictedimagenet import RestrictedImagenet as restImagenet
import numpy as np
import pickle
import time
import scipy
import skimage
import sys

part = 1

if len(sys.argv) > 1:
    part = int(sys.argv[1])


def augment(data):
    rot = np.random.uniform(low=-5, high=5, size=(data.shape[0],))
    a = np.random.randint(0, 9, data.shape[0])
    b = np.random.randint(0, 9, data.shape[0])
    flip = np.random.randint(2, size=data.shape[0])
    gamma = np.random.uniform(low=0.9, high=1.08, size=(data.shape[0],))
    new_x = []
    for i in range(data.shape[0]):
        x = data[i, :, :, :]
        x = skimage.exposure.adjust_gamma(x, gamma[i])
        x = scipy.ndimage.rotate(x, rot[i])
        if flip[i] == 1:
            x = np.flipud(x)
        x = x[a[i]:a[i] + 56, b[i]:b[i] + 56, :]
        new_x.append(x)
    new_data = np.stack(new_x)
    return np.clip(new_data, 0, 1)


def crop(data):
    return data[:, 4:60, 4:60, :]


def save(data, name):
    with open('networks/' + str(name) + '.pkl', 'wb') as file:
        pickle.dump(data, file)


# Normal training
def train_normal(filters, kernels, strides, paddings, name, lr_val, batch_size=100, EPOCHS=25, cifar=False,
                 restimagenet=False, act=tf.nn.relu, device='/cpu:0'):
    if cifar:
        data = CIFAR()
        x_train = data.train_data + 0.5
        y_train = data.train_labels
        x_test = data.validation_data + 0.5
        y_test = data.validation_labels
    elif restimagenet:
        data = restImagenet()
        x_train = data.train_data / 255
        x_test = data.validation_data / 255
        y_train = data.train_labels
        y_test = data.validation_labels
    else:
        data = MNIST()
        x_train = data.train_data + 0.5
        y_train = data.train_labels
        x_test = data.validation_data + 0.5
        y_test = data.validation_labels

    np.random.seed(99)
    with tf.device(device):
        labels = tf.placeholder('float', shape=(None, 10))
        if cifar:
            inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
            last_shape = 3
        elif restimagenet:
            labels = tf.placeholder('float', shape=(None, 9))
            inputs = tf.placeholder('float', shape=(None, 224, 224, 3))
            last_shape = 3
        else:
            inputs = tf.placeholder('float', shape=(None, 28, 28, 1))
            last_shape = 1
        x0 = inputs

        eps = tf.placeholder('float', shape=())
        params = []
        x = x0
        layers = [x]
        weight_reg = 0
        np.random.seed(99)
        # Define network
        for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
            if type(s) is str:  # Residual
                s = int(s[1:])
                W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
                W_val = W_val.reshape((k, k, last_shape, l))
                W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
                b_val = np.zeros((l,))
                b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

                params.append((W, b))
                last_shape = l
                x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b

                if x.shape != layers[-2].shape:
                    last_x = layers[-2]
                    scale = int(last_x.shape[1]) // int(x.shape[1])
                    if scale != 1:
                        last_x = tf.nn.avg_pool(last_x, [1, scale, scale, 1], [1, scale, scale, 1], 'VALID')
                    last_x = tf.pad(last_x, [[0, 0], [0, 0], [0, 0],
                                             [int(x.shape[3] - last_x.shape[3]) // 2,
                                              int(x.shape[3] - last_x.shape[3]) // 2]])
                    x += last_x
                else:
                    x += layers[-2]
                layers.append(x)
            elif l == 'pool':
                x = tf.nn.max_pool(x, ksize=[1, k, k, 1],
                                   strides=[1, s, s, 1], padding=p)
                W = tf.fill([k, k], np.nan)
                b = tf.fill([], np.nan)
                params.append((W, b))
                layers.append(x)
            else:  # Conv
                W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
                W_val = W_val.reshape((k, k, last_shape, l))
                W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
                b_val = np.zeros((l,))
                b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

                params.append((W, b))
                last_shape = l
                if i == 0:
                    x = tf.nn.conv2d(x, W, [1, s, s, 1], p) + b
                else:
                    x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b
                layers.append(x)
        pooled = tf.nn.avg_pool(x, [1, x.shape[1], x.shape[2], 1], [1, 1, 1, 1], 'VALID')

        # Final layer
        W_val = np.random.normal(scale=1 / np.sqrt(last_shape), size=(labels.shape[-1], pooled.shape[-1])).T
        W_val = W_val.reshape((1, 1, pooled.shape[-1], labels.shape[-1]))
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        params.append((W,))
        logits = tf.nn.conv2d(pooled, W, [1, 1, 1, 1], 'SAME')
        logits = tf.layers.flatten(logits)

        # CAM
        cam = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'SAME')

        predicted_labels = tf.argmax(logits, 1)
        actual_labels = tf.argmax(labels, 1)
        accuracy = tf.contrib.metrics.accuracy(predicted_labels, actual_labels)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        lr = tf.placeholder('float', shape=())
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

    start = time.time()
    print('Time ' + str(time.time() - start))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        np.random.seed(99)
        for epoch in range(EPOCHS):
            print(epoch)
            indices = np.random.permutation(x_train.shape[0])
            for i in range(int(x_train.shape[0] / batch_size)):
                idx = indices[i * batch_size: (i + 1) * batch_size]
                feed_dict_train = {inputs: x_train[idx, :, :, :], labels: y_train[idx, :], lr: lr_val(step)}
                _, cross_entropy_value, accuracy_value = sess.run([optimizer, cross_entropy,
                                                                   accuracy], feed_dict=feed_dict_train)
                step += 1
            if (epoch + 1) % 1 == 0:
                print(step)
                print('Train set accuracy: ' + str(accuracy_value))
                feed_dict_test = {inputs: x_test[0:100, :, :, :], labels: y_test[0:100, :]}
                accuracy_value, logits_val, cam_val = sess.run([accuracy, logits, cam], feed_dict=feed_dict_test)
                print('Test set accuracy: ' + str(accuracy_value))
        save(sess.run(params), name)
    tf.reset_default_graph()
    print(name)
    return str(time.time() - start)


# TRADES
def train_trades(filters, kernels, strides, paddings, name, eps_val, lr_val, step_size=0.01, adv_steps=10,
                 batch_size=100, EPOCHS=25, restimagenet=False, cifar=False,
                 act=tf.nn.relu, device='/cpu:0'):
    if cifar:
        data = CIFAR()
        x_train = data.train_data + 0.5
        y_train = data.train_labels
        x_test = data.validation_data + 0.5
        y_test = data.validation_labels
    elif restimagenet:
        data = restImagenet()
        x_train = data.train_data / 255
        x_test = data.validation_data / 255
        y_train = data.train_labels
        y_test = data.validation_labels
    else:
        data = MNIST()
        x_train = data.train_data + 0.5
        y_train = data.train_labels
        x_test = data.validation_data + 0.5
        y_test = data.validation_labels

    np.random.seed(99)
    with tf.device(device):
        labels = tf.placeholder('float', shape=(None, 10))
        classes = 10
        if cifar:
            inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
            inputs_adv = tf.placeholder('float', shape=(None, 32, 32, 3))
            last_shape = 3
            dim = 3072
        elif restimagenet:
            inputs = tf.placeholder('float', shape=(None, 224, 224, 3))
            inputs_adv = tf.placeholder('float', shape=(None, 224, 224, 3))
            labels = tf.placeholder('float', shape=(None, 9))
            classes = 9
            last_shape = 3
            dim = 150528
        else:
            inputs = tf.placeholder('float', shape=(None, 28, 28, 1))
            inputs_adv = tf.placeholder('float', shape=(None, 28, 28, 1))
            last_shape = 1
            dim = 784
        x0 = inputs
        x0_adv = inputs_adv

        eps = tf.placeholder('float', shape=())
        params = []
        x = x0
        x_adv = x0_adv
        layers = [x]
        layers_adv = [x_adv]
        weight_reg = 0
        np.random.seed(99)
        # Define network
        for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
            if type(s) is str:  # Residual
                s = int(s[1:])
                W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
                W_val = W_val.reshape((k, k, last_shape, l))
                W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
                b_val = np.zeros((l,))
                b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

                params.append((W, b))
                last_shape = l
                x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b
                x_adv = tf.nn.conv2d(act(x_adv), W, [1, s, s, 1], p) + b

                if x.shape != layers[-2].shape:
                    last_x = layers[-2]
                    last_x_adv = layers_adv[-2]
                    scale = int(last_x.shape[1]) // int(x.shape[1])
                    if scale != 1:
                        last_x = tf.nn.avg_pool(last_x, [1, scale, scale, 1], [1, scale, scale, 1], 'VALID')
                        last_x_adv = tf.nn.avg_pool(last_x_adv, [1, scale, scale, 1], [1, scale, scale, 1], 'VALID')
                    last_x = tf.pad(last_x, [[0, 0], [0, 0], [0, 0],
                                             [int(x.shape[3] - last_x.shape[3]) // 2,
                                              int(x.shape[3] - last_x.shape[3]) // 2]])
                    last_x_adv = tf.pad(last_x_adv, [[0, 0], [0, 0], [0, 0],
                                                     [int(x.shape[3] - last_x_adv.shape[3]) // 2,
                                                      int(x.shape[3] - last_x_adv.shape[3]) // 2]])
                    x += last_x
                    x_adv += last_x_adv
                else:
                    x += layers[-2]
                    x_adv += layers_adv[-2]
                layers.append(x)
                layers_adv.append(x_adv)
            elif l == 'pool':
                x = tf.nn.max_pool(x, ksize=[1, k, k, 1],
                                   strides=[1, s, s, 1], padding=p)
                x_adv = tf.nn.max_pool(x_adv, ksize=[1, k, k, 1],
                                       strides=[1, s, s, 1], padding=p)
                W = tf.fill([k, k], np.nan)
                b = tf.fill([], np.nan)
                params.append((W, b))
                layers.append(x)
                layers_adv.append(x_adv)
            else:  # Conv
                W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
                W_val = W_val.reshape((k, k, last_shape, l))
                W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
                b_val = np.zeros((l,))
                b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

                params.append((W, b))
                last_shape = l
                if i == 0:
                    x = tf.nn.conv2d(x, W, [1, s, s, 1], p) + b
                    x_adv = tf.nn.conv2d(x_adv, W, [1, s, s, 1], p) + b
                else:
                    x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b
                    x_adv = tf.nn.conv2d(act(x_adv), W, [1, s, s, 1], p) + b
                layers.append(x)
                layers_adv.append(x_adv)
        pooled = tf.nn.avg_pool(x, [1, x.shape[1], x.shape[2], 1], [1, 1, 1, 1], 'VALID')
        pooled_adv = tf.nn.avg_pool(x_adv, [1, x.shape[1], x.shape[2], 1], [1, 1, 1, 1], 'VALID')

        # Final layer
        W_val = np.random.normal(scale=1 / np.sqrt(last_shape), size=(labels.shape[-1], pooled.shape[-1])).T
        W_val = W_val.reshape((1, 1, pooled.shape[-1], labels.shape[-1]))
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        params.append((W,))
        logits = tf.nn.conv2d(pooled, W, [1, 1, 1, 1], 'SAME')
        logits = tf.layers.flatten(logits)
        logits_adv = tf.nn.conv2d(pooled_adv, W, [1, 1, 1, 1], 'SAME')
        logits_adv = tf.layers.flatten(logits_adv)

        predicted_labels = tf.argmax(logits, 1)
        actual_labels = tf.argmax(labels, 1)
        accuracy = tf.contrib.metrics.accuracy(predicted_labels, actual_labels)

        normal_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        probs = tf.nn.softmax(logits)
        log_probs = tf.nn.log_softmax(logits)
        trades_loss = tf.einsum('ij,ij->i', probs, log_probs - tf.nn.log_softmax(logits_adv))
        cross_entropy = normal_cross_entropy + trades_loss

        # Code for attack
        grad = tf.gradients(cross_entropy, inputs_adv)[0]

        lr = tf.placeholder('float', shape=())
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

    start = time.time()
    print('Time ' + str(time.time() - start))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        np.random.seed(99)
        for epoch in range(EPOCHS):
            print(epoch)
            indices = np.random.permutation(x_train.shape[0])
            for i in range(int(x_train.shape[0] / batch_size)):
                idx = indices[i * batch_size: (i + 1) * batch_size]
                # Run attack
                x_adv = x_train[idx, :, :, :].copy()
                x_nat = x_train[idx, :, :, :].copy()
                if eps_val(step) > 0:
                    perturb = np.random.uniform(-eps_val(step), eps_val(step), x_nat.shape)
                    x_adv = x_nat + perturb
                    x_adv = np.clip(x_adv, 0, 1)

                    for j in range(adv_steps):
                        feed_dict_attack = {inputs: x_nat, inputs_adv: x_adv, labels: y_train[idx, :]}
                        grad_val = sess.run(grad, feed_dict=feed_dict_attack)
                        delta = step_size * np.sign(grad_val)
                        x_adv = x_adv + delta
                        x_adv = np.clip(x_adv, x_nat - eps_val(step), x_nat + eps_val(step))
                        x_adv = np.clip(x_adv, 0, 1)
                feed_dict_train = {inputs: x_nat, inputs_adv: x_adv, labels: y_train[idx, :], lr: lr_val(step)}
                _, cross_entropy_value, accuracy_value = sess.run([optimizer, cross_entropy,
                                                                   accuracy], feed_dict=feed_dict_train)

                step += 1
            if (epoch + 1) % 1 == 0:
                print(step)
                print('Train set accuracy: ' + str(accuracy_value))
                feed_dict_test = {inputs: x_test[0:100, :, :, :], labels: y_test[0:100, :]}
                accuracy_value, logits_val = sess.run([accuracy, logits], feed_dict=feed_dict_test)
                print('Test set accuracy: ' + str(accuracy_value))
        save(sess.run(params), name)
    tf.reset_default_graph()
    print(name)
    return str(time.time() - start)


# Adversarial training
def train_adv(filters, kernels, strides, paddings, name, eps_val, lr_val, step_size=0.01, adv_steps=10,
              batch_size=100, EPOCHS=25, restimagenet=False, cifar=False,
              act=tf.nn.relu, device='/cpu:0'):
    if cifar:
        data = CIFAR()
        x_train = data.train_data + 0.5
        y_train = data.train_labels
        x_test = data.validation_data + 0.5
        y_test = data.validation_labels
    elif restimagenet:
        data = restImagenet()
        x_train = data.train_data / 255
        x_test = data.validation_data / 255
        y_train = data.train_labels
        y_test = data.validation_labels
    else:
        data = MNIST()
        x_train = data.train_data + 0.5
        y_train = data.train_labels
        x_test = data.validation_data + 0.5
        y_test = data.validation_labels

    np.random.seed(99)
    with tf.device(device):
        labels = tf.placeholder('float', shape=(None, 10))
        if cifar:
            inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
            last_shape = 3
        elif restimagenet:
            inputs = tf.placeholder('float', shape=(None, 224, 224, 3))
            labels = tf.placeholder('float', shape=(None, 9))
            last_shape = 3
        else:
            inputs = tf.placeholder('float', shape=(None, 28, 28, 1))
            last_shape = 1
        x0 = inputs

        eps = tf.placeholder('float', shape=())
        params = []
        x = x0
        layers = [x]
        weight_reg = 0
        np.random.seed(99)
        # Define network
        for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
            if type(s) is str:  # Residual
                s = int(s[1:])
                W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
                W_val = W_val.reshape((k, k, last_shape, l))
                W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
                b_val = np.zeros((l,))
                b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

                params.append((W, b))
                last_shape = l
                x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b

                if x.shape != layers[-2].shape:
                    last_x = layers[-2]
                    scale = int(last_x.shape[1]) // int(x.shape[1])
                    if scale != 1:
                        last_x = tf.nn.avg_pool(last_x, [1, scale, scale, 1], [1, scale, scale, 1], 'VALID')
                    last_x = tf.pad(last_x, [[0, 0], [0, 0], [0, 0],
                                             [int(x.shape[3] - last_x.shape[3]) // 2,
                                              int(x.shape[3] - last_x.shape[3]) // 2]])
                    x += last_x
                else:
                    x += layers[-2]
                layers.append(x)
            elif l == 'pool':
                x = tf.nn.max_pool(x, ksize=[1, k, k, 1],
                                   strides=[1, s, s, 1], padding=p)
                W = tf.fill([k, k], np.nan)
                b = tf.fill([], np.nan)
                params.append((W, b))
                layers.append(x)
            else:
                W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
                W_val = W_val.reshape((k, k, last_shape, l))
                W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
                b_val = np.zeros((l,))
                b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

                params.append((W, b))
                last_shape = l
                if i == 0:
                    x = tf.nn.conv2d(x, W, [1, s, s, 1], p) + b
                else:
                    x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b
                layers.append(x)
        pooled = tf.nn.avg_pool(x, [1, x.shape[1], x.shape[2], 1], [1, 1, 1, 1], 'VALID')

        # Final layer
        W_val = np.random.normal(scale=1 / np.sqrt(last_shape), size=(labels.shape[-1], pooled.shape[-1])).T
        W_val = W_val.reshape((1, 1, pooled.shape[-1], labels.shape[-1]))
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        params.append((W,))
        logits = tf.nn.conv2d(pooled, W, [1, 1, 1, 1], 'SAME')
        logits = tf.layers.flatten(logits)

        # CAM
        cam = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'SAME')

        predicted_labels = tf.argmax(logits, 1)
        actual_labels = tf.argmax(labels, 1)
        accuracy = tf.contrib.metrics.accuracy(predicted_labels, actual_labels)

        cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        # Code for attack
        grad = tf.gradients(cross_entropy, inputs)[0]

        lr = tf.placeholder('float', shape=())
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

    start = time.time()
    print('Time ' + str(time.time() - start))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        np.random.seed(99)
        for epoch in range(EPOCHS):
            print(epoch)
            indices = np.random.permutation(x_train.shape[0])
            for i in range(int(x_train.shape[0] / batch_size)):
                idx = indices[i * batch_size: (i + 1) * batch_size]
                # Run attack
                x_adv = x_train[idx, :, :, :].copy()
                x_nat = x_train[idx, :, :, :].copy()
                if eps_val(step) > 0:
                    perturb = np.random.uniform(-eps_val(step), eps_val(step), x_nat.shape)
                    x_adv = x_nat + perturb
                    x_adv = np.clip(x_adv, 0, 1)
                    for j in range(adv_steps):
                        feed_dict_attack = {inputs: x_adv, labels: y_train[idx, :]}
                        grad_val = sess.run(grad, feed_dict=feed_dict_attack)
                        delta = step_size * np.sign(grad_val)
                        x_adv = x_adv + delta
                        x_adv = np.clip(x_adv, x_nat - eps_val(step), x_nat + eps_val(step))
                        x_adv = np.clip(x_adv, 0, 1)
                feed_dict_train = {inputs: x_adv, labels: y_train[idx, :], lr: lr_val(step)}
                _, cross_entropy_value, accuracy_value = sess.run([optimizer, cross_entropy,
                                                                   accuracy], feed_dict=feed_dict_train)
                step += 1
            if (epoch + 1) % 1 == 0:
                print(step)
                print('Train set accuracy: ' + str(accuracy_value))
                feed_dict_test = {inputs: x_test[0:100, :, :, :], labels: y_test[0:100, :]}
                accuracy_value, logits_val, cam_val = sess.run([accuracy, logits, cam], feed_dict=feed_dict_test)
                print('Test set accuracy: ' + str(accuracy_value))
        save(sess.run(params), name)
    tf.reset_default_graph()
    print(name)
    return str(time.time() - start)


# Int (and variants -1-class, -Adv)
def train_int(filters, kernels, strides, paddings, name, eps_val, lr_val, adv=False, step_size=0.01, adv_steps=10,
              lam=0.01, oneclass=False, batch_size=100, EPOCHS=25, restimagenet=False, cifar=False,
              gtsrb=False, tinyimagenet=False, act=tf.nn.relu, device='/cpu:0'):
    if cifar:
        data = CIFAR()
        x_train = data.train_data + 0.5
        y_train = data.train_labels
        x_test = data.validation_data + 0.5
        y_test = data.validation_labels
    elif restimagenet:
        data = restImagenet()
        x_train = data.train_data / 255
        x_test = data.validation_data / 255
        y_train = data.train_labels
        y_test = data.validation_labels
    else:
        data = MNIST()
        x_train = data.train_data + 0.5
        y_train = data.train_labels
        x_test = data.validation_data + 0.5
        y_test = data.validation_labels

    np.random.seed(99)
    with tf.device(device):
        labels = tf.placeholder('float', shape=(None, 10))
        classes = 10
        if cifar:
            inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
            inputs_adv = tf.placeholder('float', shape=(None, 32, 32, 3))
            last_shape = 3
            dim = 3072
        elif gtsrb:
            labels = tf.placeholder('float', shape=(None, 43))
            inputs = tf.placeholder('float', shape=(None, 28, 28, 3))
            inputs_adv = tf.placeholder('float', shape=(None, 28, 28, 3))
            classes = 43
            last_shape = 3
            dim = 2352
        elif tinyimagenet:
            labels = tf.placeholder('float', shape=(None, 200))
            inputs = tf.placeholder('float', shape=(None, 56, 56, 3))
            inputs_adv = tf.placeholder('float', shape=(None, 56, 56, 3))
            last_shape = 3
            classes = 200
            dim = 12288
        elif restimagenet:
            inputs = tf.placeholder('float', shape=(None, 224, 224, 3))
            inputs_adv = tf.placeholder('float', shape=(None, 224, 224, 3))
            last_shape = 3
            labels = tf.placeholder('float', shape=(None, 9))
            classes = 9
            dim = 150528
        else:
            inputs = tf.placeholder('float', shape=(None, 28, 28, 1))
            inputs_adv = tf.placeholder('float', shape=(None, 28, 28, 1))
            last_shape = 1
            dim = 784
        x0 = inputs
        x0_adv = inputs_adv

        eps = tf.placeholder('float', shape=())
        params = []
        x = x0
        x_adv = x0_adv
        layers = [x]
        layers_adv = [x_adv]
        weight_reg = 0
        np.random.seed(99)
        # Define network
        for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
            if type(s) is str:  # Residual
                s = int(s[1:])
                W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
                W_val = W_val.reshape((k, k, last_shape, l))
                W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
                b_val = np.zeros((l,))
                b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

                params.append((W, b))
                last_shape = l
                x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b
                x_adv = tf.nn.conv2d(act(x_adv), W, [1, s, s, 1], p) + b

                if x.shape != layers[-2].shape:
                    last_x = layers[-2]
                    last_x_adv = layers_adv[-2]
                    scale = int(last_x.shape[1]) // int(x.shape[1])
                    if scale != 1:
                        last_x = tf.nn.avg_pool(last_x, [1, scale, scale, 1], [1, scale, scale, 1], 'VALID')
                        last_x_adv = tf.nn.avg_pool(last_x_adv, [1, scale, scale, 1], [1, scale, scale, 1], 'VALID')
                    last_x = tf.pad(last_x, [[0, 0], [0, 0], [0, 0],
                                             [int(x.shape[3] - last_x.shape[3]) // 2,
                                              int(x.shape[3] - last_x.shape[3]) // 2]])
                    last_x_adv = tf.pad(last_x_adv, [[0, 0], [0, 0], [0, 0],
                                                     [int(x.shape[3] - last_x_adv.shape[3]) // 2,
                                                      int(x.shape[3] - last_x_adv.shape[3]) // 2]])
                    x += last_x
                    x_adv += last_x_adv
                else:
                    x += layers[-2]
                    x_adv += layers_adv[-2]
                layers.append(x)
                layers_adv.append(x_adv)
            elif l == 'pool':
                x = tf.nn.max_pool(x, ksize=[1, k, k, 1],
                                   strides=[1, s, s, 1], padding=p)
                x_adv = tf.nn.max_pool(x_adv, ksize=[1, k, k, 1],
                                       strides=[1, s, s, 1], padding=p)
                W = tf.fill([k, k], np.nan)
                b = tf.fill([], np.nan)
                params.append((W, b))
                layers.append(x)
                layers_adv.append(x_adv)
            else:
                W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
                W_val = W_val.reshape((k, k, last_shape, l))
                W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
                b_val = np.zeros((l,))
                b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

                params.append((W, b))
                last_shape = l
                if i == 0:
                    x = tf.nn.conv2d(x, W, [1, s, s, 1], p) + b
                    x_adv = tf.nn.conv2d(x_adv, W, [1, s, s, 1], p) + b
                else:
                    x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b
                    x_adv = tf.nn.conv2d(act(x_adv), W, [1, s, s, 1], p) + b
                layers.append(x)
                layers_adv.append(x_adv)
        pooled = tf.nn.avg_pool(x, [1, x.shape[1], x.shape[2], 1], [1, 1, 1, 1], 'VALID')
        pooled_adv = tf.nn.avg_pool(x_adv, [1, x.shape[1], x.shape[2], 1], [1, 1, 1, 1], 'VALID')

        # Final layer
        W_val = np.random.normal(scale=1 / np.sqrt(last_shape), size=(labels.shape[-1], pooled.shape[-1])).T
        W_val = W_val.reshape((1, 1, pooled.shape[-1], labels.shape[-1]))
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        params.append((W,))
        logits = tf.nn.conv2d(pooled, W, [1, 1, 1, 1], 'SAME')
        logits = tf.layers.flatten(logits)
        logits_adv = tf.nn.conv2d(pooled_adv, W, [1, 1, 1, 1], 'SAME')
        logits_adv = tf.layers.flatten(logits_adv)
        softmax_adv = tf.nn.softmax(logits_adv)
        softmax_adv = softmax_adv * (1 - labels)
        softmax_adv = softmax_adv / tf.reduce_sum(softmax_adv, axis=1, keepdims=True)  # Normalize over non-true classes

        # CAM
        cam = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'SAME')
        cam_adv = tf.nn.conv2d(x_adv, W, [1, 1, 1, 1], 'SAME')
        cam_true_adv = tf.reduce_sum(tf.reshape(labels, (-1, 1, 1, classes)) * cam_adv, axis=3)
        cam_true = tf.reduce_sum(tf.reshape(labels, (-1, 1, 1, classes)) * cam, axis=3)
        cam_targ_adv = tf.reshape(softmax_adv, (-1, 1, 1, classes)) * cam_adv
        cam_targ = tf.reshape(softmax_adv, (-1, 1, 1, classes)) * cam

        if oneclass:
            cam_diff = tf.reduce_sum(tf.abs(tf.layers.flatten(cam_true_adv - cam_true)), axis=1)
        else:
            cam_diff = tf.reduce_sum(tf.abs(tf.layers.flatten(cam_targ_adv - cam_targ)), axis=1) + tf.reduce_sum(
                tf.abs(tf.layers.flatten(cam_true_adv - cam_true)), axis=1)

        predicted_labels = tf.argmax(logits, 1)
        actual_labels = tf.argmax(labels, 1)
        accuracy = tf.contrib.metrics.accuracy(predicted_labels, actual_labels)

        if adv:
            normal_cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits_adv, labels=labels))
        else:
            normal_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        reg = tf.reduce_mean(cam_diff)
        cross_entropy = normal_cross_entropy + lam * reg

        # Code for attack
        grad = tf.gradients(cross_entropy, inputs_adv)[0]

        lr = tf.placeholder('float', shape=())
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

    start = time.time()
    print('Time ' + str(time.time() - start))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        np.random.seed(99)
        for epoch in range(EPOCHS):
            print(epoch)
            indices = np.random.permutation(x_train.shape[0])
            for i in range(int(x_train.shape[0] / batch_size)):
                idx = indices[i * batch_size: (i + 1) * batch_size]
                # Run attack
                if tinyimagenet:
                    x_adv = augment(x_train[idx, :, :, :])
                    x_nat = x_adv.copy()
                else:
                    x_adv = x_train[idx, :, :, :].copy()
                    x_nat = x_train[idx, :, :, :].copy()
                if eps_val(step) > 0:
                    perturb = np.random.uniform(-eps_val(step), eps_val(step), x_nat.shape)
                    x_adv = x_nat + perturb
                    x_adv = np.clip(x_adv, 0, 1)

                    for j in range(adv_steps):
                        feed_dict_attack = {inputs: x_nat, inputs_adv: x_adv, labels: y_train[idx, :]}
                        grad_val = sess.run(grad, feed_dict=feed_dict_attack)
                        delta = step_size * np.sign(grad_val)
                        x_adv = x_adv + delta
                        x_adv = np.clip(x_adv, x_nat - eps_val(step), x_nat + eps_val(step))
                        x_adv = np.clip(x_adv, 0, 1)
                feed_dict_train = {inputs: x_nat, inputs_adv: x_adv, labels: y_train[idx, :], lr: lr_val(step)}
                _, cross_entropy_value, accuracy_value = sess.run([optimizer, cross_entropy,
                                                                   accuracy], feed_dict=feed_dict_train)
                step += 1
            if (epoch + 1) % 1 == 0:
                print(step)
                print('Train set accuracy: ' + str(accuracy_value))
                if tinyimagenet:
                    feed_dict_test = {inputs: crop(x_test[0:100, :, :, :]), labels: y_test[0:100, :]}
                else:
                    feed_dict_test = {inputs: x_test[0:100, :, :, :], labels: y_test[0:100, :]}
                accuracy_value, logits_val, cam_val = sess.run([accuracy, logits, cam], feed_dict=feed_dict_test)
                print('Test set accuracy: ' + str(accuracy_value))
        save(sess.run(params), name)
    tf.reset_default_graph()
    print(name)
    return str(time.time() - start)


# Int2 (trains int objective with standard pgd attacks)
def train_int2(filters, kernels, strides, paddings, name, eps_val, lr_val, adv=False, step_size=0.01, adv_steps=10,
               lam=0.01, oneclass=False, batch_size=100, EPOCHS=25, restimagenet=False, cifar=False,
               act=tf.nn.relu, device='/cpu:0'):
    if cifar:
        data = CIFAR()
        x_train = data.train_data + 0.5
        y_train = data.train_labels
        x_test = data.validation_data + 0.5
        y_test = data.validation_labels
    elif restimagenet:
        data = restImagenet()
        x_train = data.train_data / 255
        x_test = data.validation_data / 255
        y_train = data.train_labels
        y_test = data.validation_labels
    else:
        data = MNIST()
        x_train = data.train_data + 0.5
        y_train = data.train_labels
        x_test = data.validation_data + 0.5
        y_test = data.validation_labels

    np.random.seed(99)
    with tf.device(device):
        labels = tf.placeholder('float', shape=(None, 10))
        classes = 10
        if cifar:
            inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
            inputs_adv = tf.placeholder('float', shape=(None, 32, 32, 3))
            last_shape = 3
            dim = 3072
        elif restimagenet:
            inputs = tf.placeholder('float', shape=(None, 224, 224, 3))
            inputs_adv = tf.placeholder('float', shape=(None, 224, 224, 3))
            last_shape = 3
            labels = tf.placeholder('float', shape=(None, 9))
            classes = 9
            dim = 150528
        else:
            inputs = tf.placeholder('float', shape=(None, 28, 28, 1))
            inputs_adv = tf.placeholder('float', shape=(None, 28, 28, 1))
            last_shape = 1
            dim = 784
        x0 = inputs
        x0_adv = inputs_adv

        eps = tf.placeholder('float', shape=())
        params = []
        x = x0
        x_adv = x0_adv
        layers = [x]
        layers_adv = [x_adv]
        weight_reg = 0
        np.random.seed(99)
        # Define network
        for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
            if type(s) is str:  # Residual
                s = int(s[1:])
                W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
                W_val = W_val.reshape((k, k, last_shape, l))
                W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
                b_val = np.zeros((l,))
                b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

                params.append((W, b))
                last_shape = l
                x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b
                x_adv = tf.nn.conv2d(act(x_adv), W, [1, s, s, 1], p) + b

                if x.shape != layers[-2].shape:
                    last_x = layers[-2]
                    last_x_adv = layers_adv[-2]
                    scale = int(last_x.shape[1]) // int(x.shape[1])
                    if scale != 1:
                        last_x = tf.nn.avg_pool(last_x, [1, scale, scale, 1], [1, scale, scale, 1], 'VALID')
                        last_x_adv = tf.nn.avg_pool(last_x_adv, [1, scale, scale, 1], [1, scale, scale, 1], 'VALID')
                    last_x = tf.pad(last_x, [[0, 0], [0, 0], [0, 0],
                                             [int(x.shape[3] - last_x.shape[3]) // 2,
                                              int(x.shape[3] - last_x.shape[3]) // 2]])
                    last_x_adv = tf.pad(last_x_adv, [[0, 0], [0, 0], [0, 0],
                                                     [int(x.shape[3] - last_x_adv.shape[3]) // 2,
                                                      int(x.shape[3] - last_x_adv.shape[3]) // 2]])
                    x += last_x
                    x_adv += last_x_adv
                else:
                    x += layers[-2]
                    x_adv += layers_adv[-2]
                layers.append(x)
                layers_adv.append(x_adv)
            elif l == 'pool':
                x = tf.nn.max_pool(x, ksize=[1, k, k, 1],
                                   strides=[1, s, s, 1], padding=p)
                x_adv = tf.nn.max_pool(x_adv, ksize=[1, k, k, 1],
                                       strides=[1, s, s, 1], padding=p)
                W = tf.fill([k, k], np.nan)
                b = tf.fill([], np.nan)
                params.append((W, b))
                layers.append(x)
                layers_adv.append(x_adv)
            else:
                W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
                W_val = W_val.reshape((k, k, last_shape, l))
                W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
                b_val = np.zeros((l,))
                b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

                params.append((W, b))
                last_shape = l
                if i == 0:
                    x = tf.nn.conv2d(x, W, [1, s, s, 1], p) + b
                    x_adv = tf.nn.conv2d(x_adv, W, [1, s, s, 1], p) + b
                else:
                    x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b
                    x_adv = tf.nn.conv2d(act(x_adv), W, [1, s, s, 1], p) + b
                layers.append(x)
                layers_adv.append(x_adv)
        pooled = tf.nn.avg_pool(x, [1, x.shape[1], x.shape[2], 1], [1, 1, 1, 1], 'VALID')
        pooled_adv = tf.nn.avg_pool(x_adv, [1, x.shape[1], x.shape[2], 1], [1, 1, 1, 1], 'VALID')

        # Final layer
        W_val = np.random.normal(scale=1 / np.sqrt(last_shape), size=(labels.shape[-1], pooled.shape[-1])).T
        W_val = W_val.reshape((1, 1, pooled.shape[-1], labels.shape[-1]))
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        params.append((W,))
        logits = tf.nn.conv2d(pooled, W, [1, 1, 1, 1], 'SAME')
        logits = tf.layers.flatten(logits)
        logits_adv = tf.nn.conv2d(pooled_adv, W, [1, 1, 1, 1], 'SAME')
        logits_adv = tf.layers.flatten(logits_adv)
        softmax_adv = tf.nn.softmax(logits_adv)
        softmax_adv = softmax_adv * (1 - labels)
        softmax_adv = softmax_adv / tf.reduce_sum(softmax_adv, axis=1, keepdims=True)  # Normalize over non-true classes

        # CAM
        cam = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'SAME')
        cam_adv = tf.nn.conv2d(x_adv, W, [1, 1, 1, 1], 'SAME')
        cam_true_adv = tf.reduce_sum(tf.reshape(labels, (-1, 1, 1, classes)) * cam_adv, axis=3)
        cam_true = tf.reduce_sum(tf.reshape(labels, (-1, 1, 1, classes)) * cam, axis=3)
        cam_targ_adv = tf.reshape(softmax_adv, (-1, 1, 1, classes)) * cam_adv
        cam_targ = tf.reshape(softmax_adv, (-1, 1, 1, classes)) * cam

        if oneclass:
            cam_diff = tf.reduce_sum(tf.abs(tf.layers.flatten(cam_true_adv - cam_true)), axis=1)
        else:
            cam_diff = tf.reduce_sum(tf.abs(tf.layers.flatten(cam_targ_adv - cam_targ)), axis=1) + tf.reduce_sum(
                tf.abs(tf.layers.flatten(cam_true_adv - cam_true)), axis=1)

        predicted_labels = tf.argmax(logits, 1)
        actual_labels = tf.argmax(labels, 1)
        accuracy = tf.contrib.metrics.accuracy(predicted_labels, actual_labels)

        adv_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits_adv, labels=labels))
        if adv:
            normal_cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits_adv, labels=labels))
        else:
            normal_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
        reg = tf.reduce_mean(cam_diff)
        cross_entropy = normal_cross_entropy + lam * reg

        # Code for attack
        grad = tf.gradients(adv_cross_entropy, inputs_adv)[0]

        lr = tf.placeholder('float', shape=())
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

    start = time.time()
    print('Time ' + str(time.time() - start))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        np.random.seed(99)
        for epoch in range(EPOCHS):
            print(epoch)
            indices = np.random.permutation(x_train.shape[0])
            for i in range(int(x_train.shape[0] / batch_size)):
                idx = indices[i * batch_size: (i + 1) * batch_size]
                # Run attack
                x_adv = x_train[idx, :, :, :].copy()
                x_nat = x_train[idx, :, :, :].copy()
                if eps_val(step) > 0:
                    perturb = np.random.uniform(-eps_val(step), eps_val(step), x_nat.shape)
                    x_adv = x_nat + perturb
                    x_adv = np.clip(x_adv, 0, 1)

                    for j in range(adv_steps):
                        feed_dict_attack = {inputs: x_nat, inputs_adv: x_adv, labels: y_train[idx, :]}
                        grad_val = sess.run(grad, feed_dict=feed_dict_attack)
                        delta = step_size * np.sign(grad_val)
                        x_adv = x_adv + delta
                        x_adv = np.clip(x_adv, x_nat - eps_val(step), x_nat + eps_val(step))
                        x_adv = np.clip(x_adv, 0, 1)
                feed_dict_train = {inputs: x_nat, inputs_adv: x_adv, labels: y_train[idx, :], lr: lr_val(step)}
                _, cross_entropy_value, accuracy_value = sess.run([optimizer, cross_entropy,
                                                                   accuracy], feed_dict=feed_dict_train)
                step += 1
            if (epoch + 1) % 1 == 0:
                print(step)
                print('Train set accuracy: ' + str(accuracy_value))
                feed_dict_test = {inputs: x_test[0:100, :, :, :], labels: y_test[0:100, :]}
                accuracy_value, logits_val, cam_val = sess.run([accuracy, logits, cam], feed_dict=feed_dict_test)
                print('Test set accuracy: ' + str(accuracy_value))
        save(sess.run(params), name)
    tf.reset_default_graph()
    print(name)
    return str(time.time() - start)


# Trains IG-Norm, IG-Sum-Norm
def train_ig(filters, kernels, strides, paddings, name, eps_val, lr_val, twoclass=False, lam=0.01, adv=False,
             step_size=0.01, adv_steps=10, batch_size=100, EPOCHS=25, restimagenet=False, cifar=False,
             gtsrb=False, tinyimagenet=False, act=tf.nn.relu, device='/cpu:0'):
    if cifar:
        data = CIFAR()
        x_train = data.train_data + 0.5
        y_train = data.train_labels
        x_test = data.validation_data + 0.5
        y_test = data.validation_labels
    elif restimagenet:
        data = restImagenet()
        x_train = data.train_data / 255
        x_test = data.validation_data / 255
        y_train = data.train_labels
        y_test = data.validation_labels
    else:
        data = MNIST()
        x_train = data.train_data + 0.5
        y_train = data.train_labels
        x_test = data.validation_data + 0.5
        y_test = data.validation_labels

    np.random.seed(99)
    with tf.device(device):
        labels = tf.placeholder('float', shape=(None, 10))
        classes = 10
        if cifar:
            inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
            inputs_adv = tf.placeholder('float', shape=(None, 32, 32, 3))
            last_shape = 3
            dim = 3072
        elif gtsrb:
            labels = tf.placeholder('float', shape=(None, 43))
            inputs = tf.placeholder('float', shape=(None, 28, 28, 3))
            inputs_adv = tf.placeholder('float', shape=(None, 28, 28, 3))
            classes = 43
            last_shape = 3
            dim = 2352
        elif tinyimagenet:
            labels = tf.placeholder('float', shape=(None, 200))
            inputs = tf.placeholder('float', shape=(None, 56, 56, 3))
            inputs_adv = tf.placeholder('float', shape=(None, 56, 56, 3))
            last_shape = 3
            classes = 200
            dim = 12288
        elif restimagenet:
            inputs = tf.placeholder('float', shape=(None, 224, 224, 3))
            inputs_adv = tf.placeholder('float', shape=(None, 224, 224, 3))
            last_shape = 3
            dim = 150528
            labels = tf.placeholder('float', shape=(None, 9))
            classes = 9
        else:
            inputs = tf.placeholder('float', shape=(None, 28, 28, 1))
            inputs_adv = tf.placeholder('float', shape=(None, 28, 28, 1))
            last_shape = 1
            dim = 784
        x0 = inputs
        x0_adv = inputs_adv

        eps = tf.placeholder('float', shape=())
        params = []
        x = x0
        x_adv = x0_adv
        layers = [x]
        layers_adv = [x_adv]
        weight_reg = 0
        np.random.seed(99)
        # Define network
        for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
            if type(s) is str:  # Residual
                s = int(s[1:])
                W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
                W_val = W_val.reshape((k, k, last_shape, l))
                W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
                b_val = np.zeros((l,))
                b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

                params.append((W, b))
                last_shape = l
                x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b
                x_adv = tf.nn.conv2d(act(x_adv), W, [1, s, s, 1], p) + b

                if x.shape != layers[-2].shape:
                    last_x = layers[-2]
                    last_x_adv = layers_adv[-2]
                    scale = int(last_x.shape[1]) // int(x.shape[1])
                    if scale != 1:
                        last_x = tf.nn.avg_pool(last_x, [1, scale, scale, 1], [1, scale, scale, 1], 'VALID')
                        last_x_adv = tf.nn.avg_pool(last_x_adv, [1, scale, scale, 1], [1, scale, scale, 1], 'VALID')
                    last_x = tf.pad(last_x, [[0, 0], [0, 0], [0, 0],
                                             [int(x.shape[3] - last_x.shape[3]) // 2,
                                              int(x.shape[3] - last_x.shape[3]) // 2]])
                    last_x_adv = tf.pad(last_x_adv, [[0, 0], [0, 0], [0, 0],
                                                     [int(x.shape[3] - last_x_adv.shape[3]) // 2,
                                                      int(x.shape[3] - last_x_adv.shape[3]) // 2]])
                    x += last_x
                    x_adv += last_x_adv
                else:
                    x += layers[-2]
                    x_adv += layers_adv[-2]
                layers.append(x)
                layers_adv.append(x_adv)
            elif l == 'pool':
                x = tf.nn.max_pool(x, ksize=[1, k, k, 1],
                                   strides=[1, s, s, 1], padding=p)
                x_adv = tf.nn.max_pool(x_adv, ksize=[1, k, k, 1],
                                       strides=[1, s, s, 1], padding=p)
                W = tf.fill([k, k], np.nan)
                b = tf.fill([], np.nan)
                params.append((W, b))
                layers.append(x)
                layers_adv.append(x_adv)
            else:
                W_val = np.random.normal(scale=1 / np.sqrt(k * k * last_shape), size=(l, k * k * last_shape)).T
                W_val = W_val.reshape((k, k, last_shape, l))
                W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
                b_val = np.zeros((l,))
                b = tf.Variable(tf.convert_to_tensor(b_val, dtype=tf.float32))

                params.append((W, b))
                last_shape = l
                if i == 0:
                    x = tf.nn.conv2d(x, W, [1, s, s, 1], p) + b
                    x_adv = tf.nn.conv2d(x_adv, W, [1, s, s, 1], p) + b
                else:
                    x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b
                    x_adv = tf.nn.conv2d(act(x_adv), W, [1, s, s, 1], p) + b
                layers.append(x)
                layers_adv.append(x_adv)
        pooled = tf.nn.avg_pool(x, [1, x.shape[1], x.shape[2], 1], [1, 1, 1, 1], 'VALID')
        pooled_adv = tf.nn.avg_pool(x_adv, [1, x.shape[1], x.shape[2], 1], [1, 1, 1, 1], 'VALID')

        # Final layer
        W_val = np.random.normal(scale=1 / np.sqrt(last_shape), size=(labels.shape[-1], pooled.shape[-1])).T
        W_val = W_val.reshape((1, 1, pooled.shape[-1], labels.shape[-1]))
        W = tf.Variable(tf.convert_to_tensor(W_val, dtype=tf.float32))
        params.append((W,))
        logits = tf.nn.conv2d(pooled, W, [1, 1, 1, 1], 'SAME')
        logits = tf.layers.flatten(logits)
        logits_adv = tf.nn.conv2d(pooled_adv, W, [1, 1, 1, 1], 'SAME')
        logits_adv = tf.layers.flatten(logits_adv)
        softmax_adv = tf.nn.softmax(logits_adv)
        softmax_adv = softmax_adv * (1 - labels)
        softmax_adv = softmax_adv / tf.reduce_sum(softmax_adv, axis=1, keepdims=True)  # Normalize over non-true classes

        def predict(inputs):
            x = inputs
            layers = [x]
            # Define network
            for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
                if type(s) is str:  # Residual
                    s = int(s[1:])
                    W, b = params[i]
                    x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b

                    if x.shape != layers[-2].shape:
                        last_x = layers[-2]
                        scale = int(last_x.shape[1]) // int(x.shape[1])
                        if scale != 1:
                            last_x = tf.nn.avg_pool(last_x, [1, scale, scale, 1], [1, scale, scale, 1], 'VALID')
                        last_x = tf.pad(last_x, [[0, 0], [0, 0], [0, 0],
                                                 [int(x.shape[3] - last_x.shape[3]) // 2,
                                                  int(x.shape[3] - last_x.shape[3]) // 2]])
                        x += last_x
                    else:
                        x += layers[-2]
                    layers.append(x)
                elif l == 'pool':
                    x = tf.nn.max_pool(x, ksize=[1, k, k, 1],
                                       strides=[1, s, s, 1], padding=p)
                    layers.append(x)
                else:  # Conv
                    W, b = params[i]
                    if i == 0:
                        x = tf.nn.conv2d(x, W, [1, s, s, 1], p) + b
                    else:
                        x = tf.nn.conv2d(act(x), W, [1, s, s, 1], p) + b
                    layers.append(x)
            pooled = tf.nn.avg_pool(x, [1, int(x.shape[1]), int(x.shape[2]), 1], [1, 1, 1, 1], 'VALID')

            W = params[-1][0]
            logits = tf.nn.conv2d(pooled, W, [1, 1, 1, 1], 'SAME')
            logits = tf.layers.flatten(logits)

            return logits

        predicted_labels = tf.argmax(logits, 1)
        actual_labels = tf.argmax(labels, 1)
        accuracy = tf.contrib.metrics.accuracy(predicted_labels, actual_labels)

        if adv:
            normal_cross_entropy = tf.reduce_mean(
                tf.nn.softmax_cross_entropy_with_logits(logits=logits_adv, labels=labels))
        else:
            normal_cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))

        if twoclass:
            ig_full = []
            for c in range(classes):
                grad_sum = tf.zeros_like(inputs)
                for k in range(5):
                    z = inputs + (inputs_adv - inputs) * (k + 1) / 5
                    f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict(z), labels=labels))
                    grad_sum += tf.gradients(f, z)[0]
                ig = (inputs_adv - inputs) * grad_sum / 5
                ig_full.append(ig)
            ig_full = tf.stack(ig_full, axis=-1)

            ig_true = tf.reduce_sum(tf.reshape(labels, (-1, 1, 1, 1, classes)) * ig_full, axis=3)
            ig_targ = tf.reshape(softmax_adv, (-1, 1, 1, 1, classes)) * ig_full

            reg = lam * tf.reduce_mean(tf.reduce_sum(tf.abs(tf.layers.flatten(ig_targ)), axis=1) + tf.reduce_sum(
                tf.abs(tf.layers.flatten(ig_true)), axis=1))
        else:
            grad_sum = tf.zeros_like(inputs)
            for k in range(5):
                z = inputs + (inputs_adv - inputs) * (k + 1) / 5
                f = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=predict(z), labels=labels))
                grad_sum += tf.gradients(f, z)[0]
            ig = (inputs_adv - inputs) * grad_sum / 5

            reg = tf.reduce_mean(tf.reduce_sum(tf.layers.flatten(tf.abs(ig)), axis=1))

        cross_entropy = normal_cross_entropy + reg

        # Code for attack
        grad = tf.gradients(cross_entropy, inputs_adv)[0]

        lr = tf.placeholder('float', shape=())
        optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cross_entropy)

    start = time.time()
    print('Time ' + str(time.time() - start))
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        step = 0
        np.random.seed(99)
        for epoch in range(EPOCHS):
            print(epoch)
            indices = np.random.permutation(x_train.shape[0])
            for i in range(int(x_train.shape[0] / batch_size)):
                idx = indices[i * batch_size: (i + 1) * batch_size]
                # Run attack
                if tinyimagenet:
                    x_adv = augment(x_train[idx, :, :, :])
                    x_nat = x_adv.copy()
                else:
                    x_adv = x_train[idx, :, :, :].copy()
                    x_nat = x_train[idx, :, :, :].copy()
                if eps_val(step) > 0:
                    perturb = np.random.uniform(-eps_val(step), eps_val(step), x_nat.shape)
                    x_adv = x_nat + perturb
                    x_adv = np.clip(x_adv, 0, 1)

                    for j in range(adv_steps):
                        feed_dict_attack = {inputs: x_nat, inputs_adv: x_adv, labels: y_train[idx, :]}
                        grad_val = sess.run(grad, feed_dict=feed_dict_attack)
                        delta = step_size * np.sign(grad_val)
                        x_adv = x_adv + delta
                        x_adv = np.clip(x_adv, x_nat - eps_val(step), x_nat + eps_val(step))
                        x_adv = np.clip(x_adv, 0, 1)
                feed_dict_train = {inputs: x_nat, inputs_adv: x_adv, labels: y_train[idx, :], lr: lr_val(step)}
                _, cross_entropy_value, accuracy_value = sess.run([optimizer, cross_entropy,
                                                                   accuracy], feed_dict=feed_dict_train)
                step += 1
            if (epoch + 1) % 1 == 0:
                print(step)
                print('Train set accuracy: ' + str(accuracy_value))
                if tinyimagenet:
                    feed_dict_test = {inputs: crop(x_test[0:100, :, :, :]), labels: y_test[0:100, :]}
                else:
                    feed_dict_test = {inputs: x_test[0:100, :, :, :], labels: y_test[0:100, :]}
                accuracy_value, logits_val = sess.run([accuracy, logits], feed_dict=feed_dict_test)
                print('Test set accuracy: ' + str(accuracy_value))
        save(sess.run(params), name)
    tf.reset_default_graph()
    print(name)
    return str(time.time() - start)


if __name__ == '__main__':

    times = []

    # Networks that train fast: MNIST Small, MNIST Pool, CIFAR Small
    if part == 1:
        def lr_val(step):
            return 0.0001


        def eps_val(eps):
            def f(step):
                if step <= 2000:
                    return 0
                elif step <= 10000:
                    return eps * (step - 2000) / 8000
                else:
                    return eps

            return f


        # Steps = 40 for MNIST
        # batch size = 50 for MNIST
        # step_size = 0.01 for MNIST

        # MNIST Small
        filters = [16, 32, 100]
        kernels = [4, 4, 7]
        strides = [2, 2, 1]
        paddings = ['SAME', 'SAME', 'SAME']

        for e in [100]:
            t = train_normal(filters, kernels, strides, paddings, 'mnist_smallgap_normal_' + str(e), lr_val,
                             batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_adv(filters, kernels, strides, paddings, 'mnist_smallgap_adv_' + str(e), eps_val(0.3), lr_val,
                          step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_trades(filters, kernels, strides, paddings, 'mnist_smallgap_trades_' + str(e), eps_val(0.3),
                             lr_val, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_int_' + str(e), eps_val(0.3), lr_val,
                          lam=0.01, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_intadv_' + str(e), eps_val(0.3), lr_val,
                          adv=True, lam=0.01, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_intone_' + str(e), eps_val(0.3), lr_val,
                          oneclass=True, lam=0.01, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        # MNIST Pool
        filters = [32, 'pool', 64, 'pool']
        kernels = [5, 2, 5, 2]
        strides = [1, 2, 1, 2]
        paddings = ['SAME', 'SAME', 'SAME', 'SAME']

        for e in [100]:
            t = train_normal(filters, kernels, strides, paddings, 'mnist_pool_normal_' + str(e), lr_val, batch_size=50,
                             EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_adv(filters, kernels, strides, paddings, 'mnist_pool_adv_' + str(e), eps_val(0.3), lr_val,
                          step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_trades(filters, kernels, strides, paddings, 'mnist_pool_trades_' + str(e), eps_val(0.3), lr_val,
                             step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_pool_int_' + str(e), eps_val(0.3), lr_val,
                          step_size=0.01, adv_steps=40, batch_size=50, lam=0.01, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_pool_intadv_' + str(e), eps_val(0.3), lr_val,
                          adv=True, step_size=0.01, adv_steps=40, batch_size=50, lam=0.01, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_pool_intone_' + str(e), eps_val(0.3), lr_val,
                          oneclass=True, step_size=0.01, adv_steps=40, batch_size=50, lam=0.01, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)


        def lr_val(step):
            if step <= 40000:
                return 0.001
            elif step <= 60000:
                return 0.0001
            else:
                return 0.00001


        def eps_val(eps):
            def f(step):
                if step <= 5000:
                    return 0
                elif step <= 70000:
                    return eps * (step - 5000) / 65000
                else:
                    return eps

            return f


        # Steps = 10 for CIFAR
        # batch size = 128 for CIFAR
        # step_size = 2/255 for CIFAR

        # CIFAR Small
        filters = [16, 32, 100]
        kernels = [4, 4, 7]
        strides = [2, 2, 1]
        paddings = ['SAME', 'SAME', 'SAME']

        for e in [200]:
            t = train_normal(filters, kernels, strides, paddings, 'cifar_smallgap_normal_' + str(e), lr_val,
                             batch_size=128, cifar=True, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_adv(filters, kernels, strides, paddings, 'cifar_smallgap_adv_' + str(e), eps_val(8 / 255), lr_val,
                          step_size=2 / 255, adv_steps=10, batch_size=128, cifar=True, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_trades(filters, kernels, strides, paddings, 'cifar_smallgap_trades_' + str(e), eps_val(8 / 255),
                             lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, cifar=True, EPOCHS=e,
                             device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_int_' + str(e), eps_val(8 / 255), lr_val,
                          step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.01, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_intadv_' + str(e), eps_val(8 / 255),
                          lr_val, adv=True, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.01, cifar=True,
                          EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_intone_' + str(e), eps_val(8 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.01, oneclass=True, cifar=True,
                          EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

    # IG-based methods, MNIST
    if part == 2:
        def lr_val(step):
            return 0.0001


        def eps_val(eps):
            def f(step):
                if step <= 2000:
                    return 0
                elif step <= 10000:
                    return eps * (step - 2000) / 8000
                else:
                    return eps

            return f


        # Steps = 40 for MNIST
        # batch size = 50 for MNIST
        # step_size = 0.01 for MNIST

        # MNIST Small
        filters = [16, 32, 100]
        kernels = [4, 4, 7]
        strides = [2, 2, 1]
        paddings = ['SAME', 'SAME', 'SAME']

        for e in [100]:
            t = train_ig(filters, kernels, strides, paddings, 'mnist_smallgap_ig_' + str(e), eps_val(0.3), lr_val,
                         step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_ig(filters, kernels, strides, paddings, 'mnist_smallgap_igsum_' + str(e), eps_val(0.3), lr_val,
                         adv=True, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_ig(filters, kernels, strides, paddings, 'mnist_smallgap_intig_' + str(e), eps_val(0.3), lr_val,
                         twoclass=True, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_ig(filters, kernels, strides, paddings, 'mnist_smallgap_intigadv_' + str(e), eps_val(0.3), lr_val,
                         twoclass=True, adv=True, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e,
                         device='/device:GPU:0')
            times.append(t)
            print(times)

        # MNIST Pool
        filters = [32, 'pool', 64, 'pool']
        kernels = [5, 2, 5, 2]
        strides = [1, 2, 1, 2]
        paddings = ['SAME', 'SAME', 'SAME', 'SAME']

        for e in [100]:
            t = train_ig(filters, kernels, strides, paddings, 'mnist_pool_ig_' + str(e), eps_val(0.3), lr_val,
                         step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_ig(filters, kernels, strides, paddings, 'mnist_pool_igsum_' + str(e), eps_val(0.3), lr_val,
                         adv=True, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_ig(filters, kernels, strides, paddings, 'mnist_pool_intig_' + str(e), eps_val(0.3), lr_val,
                         twoclass=True, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_ig(filters, kernels, strides, paddings, 'mnist_pool_intigadv_' + str(e), eps_val(0.3), lr_val,
                         twoclass=True, adv=True, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e,
                         device='/device:GPU:0')
            times.append(t)
            print(times)

    # IG-based methods, CIFAR Small, CIFAR WResnet
    elif part == 3:
        def lr_val(step):
            if step <= 40000:
                return 0.001
            elif step <= 60000:
                return 0.0001
            else:
                return 0.00001


        def eps_val(eps):
            def f(step):
                if step <= 5000:
                    return 0
                elif step <= 70000:
                    return eps * (step - 5000) / 65000
                else:
                    return eps

            return f


        # Steps = 10 for CIFAR
        # batch size = 128 for CIFAR
        # step_size = 2/255 for CIFAR

        # CIFAR Small
        filters = [16, 32, 100]
        kernels = [4, 4, 7]
        strides = [2, 2, 1]
        paddings = ['SAME', 'SAME', 'SAME']

        for e in [200]:
            t = train_ig(filters, kernels, strides, paddings, 'cifar_smallgap_ig_' + str(e), eps_val(8 / 255), lr_val,
                         step_size=2 / 255, adv_steps=10, batch_size=128, cifar=True, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_ig(filters, kernels, strides, paddings, 'cifar_smallgap_igsum_' + str(e), eps_val(8 / 255),
                         lr_val, adv=True, step_size=2 / 255, adv_steps=10, batch_size=128, cifar=True, EPOCHS=e,
                         device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_ig(filters, kernels, strides, paddings, 'cifar_smallgap_intig_' + str(e), eps_val(8 / 255),
                         lr_val, twoclass=True, step_size=2 / 255, adv_steps=10, batch_size=128, cifar=True, EPOCHS=e,
                         device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_ig(filters, kernels, strides, paddings, 'cifar_smallgap_intigadv_' + str(e), eps_val(8 / 255),
                         lr_val, twoclass=True, adv=True, step_size=2 / 255, adv_steps=10, batch_size=128, cifar=True,
                         EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        # CIFAR WResnet
        filters = [16] + 10 * [16] + 10 * [32] + 10 * [64]
        kernels = 31 * [3]
        strides = [1] + 5 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1']
        paddings = 31 * ['SAME']

        for e in [200]:
            t = train_normal(filters, kernels, strides, paddings, 'cifar_wresnet_normal_' + str(e), lr_val,
                             batch_size=128, cifar=True, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_adv(filters, kernels, strides, paddings, 'cifar_wresnet_adv_' + str(e), eps_val(8 / 255), lr_val,
                          step_size=2 / 255, adv_steps=10, batch_size=128, cifar=True, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_trades(filters, kernels, strides, paddings, 'cifar_wresnet_trades_' + str(e), eps_val(8 / 255),
                             lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, cifar=True, EPOCHS=e,
                             device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_wresnet_int_' + str(e), eps_val(8 / 255), lr_val,
                          step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.01, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_wresnet_intadv_' + str(e), eps_val(8 / 255),
                          lr_val, adv=True, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.01, cifar=True,
                          EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_wresnet_intone_' + str(e), eps_val(8 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.01, oneclass=True, cifar=True,
                          EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

    # Train networks with Int2 (standard attack in inner maximization)
    elif part == 4:
        def lr_val(step):
            return 0.0001


        def eps_val(eps):
            def f(step):
                if step <= 2000:
                    return 0
                elif step <= 10000:
                    return eps * (step - 2000) / 8000
                else:
                    return eps

            return f


        # Steps = 40 for MNIST
        # batch size = 50 for MNIST
        # step_size = 0.01 for MNIST

        # MNIST Small
        filters = [16, 32, 100]
        kernels = [4, 4, 7]
        strides = [2, 2, 1]
        paddings = ['SAME', 'SAME', 'SAME']

        for e in [100]:
            t = train_int2(filters, kernels, strides, paddings, 'mnist_smallgap_int2_' + str(e), eps_val(0.3),
                           lr_val, lam=0.01, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e,
                           device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int2(filters, kernels, strides, paddings, 'mnist_smallgap_int2adv_' + str(e),
                           eps_val(0.3), lr_val, adv=True, lam=0.01, step_size=0.01, adv_steps=40, batch_size=50,
                           EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        # MNIST Pool
        filters = [32, 'pool', 64, 'pool']
        kernels = [5, 2, 5, 2]
        strides = [1, 2, 1, 2]
        paddings = ['SAME', 'SAME', 'SAME', 'SAME']

        for e in [100]:
            t = train_int2(filters, kernels, strides, paddings, 'mnist_pool_int2_' + str(e), eps_val(0.3),
                           lr_val, step_size=0.01, adv_steps=40, batch_size=50, lam=0.01, EPOCHS=e,
                           device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int2(filters, kernels, strides, paddings, 'mnist_pool_int2adv_' + str(e), eps_val(0.3),
                           lr_val, adv=True, step_size=0.01, adv_steps=40, batch_size=50, lam=0.01, EPOCHS=e,
                           device='/device:GPU:0')
            times.append(t)
            print(times)


        def lr_val(step):
            if step <= 40000:
                return 0.001
            elif step <= 60000:
                return 0.0001
            else:
                return 0.00001


        def eps_val(eps):
            def f(step):
                if step <= 5000:
                    return 0
                elif step <= 70000:
                    return eps * (step - 5000) / 65000
                else:
                    return eps

            return f


        # Steps = 10 for CIFAR
        # batch size = 128 for CIFAR
        # step_size = 2/255 for CIFAR

        # CIFAR Small
        filters = [16, 32, 100]
        kernels = [4, 4, 7]
        strides = [2, 2, 1]
        paddings = ['SAME', 'SAME', 'SAME']

        for e in [200]:
            t = train_int2(filters, kernels, strides, paddings, 'cifar_smallgap_int2_' + str(e), eps_val(8 / 255),
                           lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.01, cifar=True, EPOCHS=e,
                           device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int2(filters, kernels, strides, paddings, 'cifar_smallgap_int2adv_' + str(e),
                           eps_val(8 / 255), lr_val, adv=True, step_size=2 / 255, adv_steps=10, batch_size=128,
                           lam=0.01, cifar=True, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        # CIFAR WResnet
        filters = [16] + 10 * [16] + 10 * [32] + 10 * [64]
        kernels = 31 * [3]
        strides = [1] + 5 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1']
        paddings = 31 * ['SAME']

        for e in [200]:
            t = train_int2(filters, kernels, strides, paddings, 'cifar_wresnet_int2_' + str(e), eps_val(8 / 255),
                           lr_val,
                           step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.01, cifar=True, EPOCHS=e,
                           device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int2(filters, kernels, strides, paddings, 'cifar_wresnet_int2adv_' + str(e),
                           eps_val(8 / 255), lr_val, adv=True, step_size=2 / 255, adv_steps=10, batch_size=128,
                           lam=0.01, cifar=True, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)
    # R-Imagenet
    elif part == 5:
        def lr_val(step):
            if step <= 8000:
                return 0.001
            elif step <= 16000:
                return 0.0001
            else:
                return 0.00001


        def eps_val(eps):
            def f(step):
                if step <= 5000:
                    return 0
                elif step <= 10000:
                    return eps * (step - 5000) / 5000
                else:
                    return eps

            return f


        # Steps = 7 for Restricted Imagenet
        # batch size = 128 for Restricted Imagenet
        # step_size = 0.1 for Restricted Imagenet

        # Restricted Imagenet WResnet
        filters = [16] + 10 * [16] + 10 * [32] + 10 * [64]
        kernels = 31 * [3]
        strides = [1] + 5 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1']
        paddings = 31 * ['SAME']

        for e in [35]:
            t = train_normal(filters, kernels, strides, paddings, 'restimagenet_wresnet_normal_' + str(e), lr_val,
                             batch_size=64, restimagenet=True, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [35]:
            t = train_normal(filters, kernels, strides, paddings, 'restimagenet_wresnet_normal_' + str(e), lr_val,
                             batch_size=64, restimagenet=True, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [35]:
            t = train_adv(filters, kernels, strides, paddings, 'restimagenet_wresnet_adv_' + str(e), eps_val(0.003),
                          lr_val, step_size=0.1, adv_steps=7, batch_size=64, cifar=True,
                          EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [35]:
            t = train_int(filters, kernels, strides, paddings, 'restimagenet_wresnet_int_' + str(e), eps_val(0.003),
                          lr_val, step_size=0.1, adv_steps=7, batch_size=64, lam=0.01, restimagenet=True,
                          EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [35]:
            t = train_int(filters, kernels, strides, paddings, 'restimagenet_wresnet_intadv_' + str(e), eps_val(0.003),
                          lr_val, adv=True, step_size=0.1, adv_steps=7, batch_size=64, lam=0.01, restimagenet=True,
                          EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [35]:
            t = train_int(filters, kernels, strides, paddings, 'restimagenet_wresnet_intone_' + str(e), eps_val(0.003),
                          lr_val, step_size=0.1, adv_steps=7, batch_size=128, lam=0.01, oneclass=True,
                          restimagenet=True, EPOCHS=e)
            times.append(t)
            print(times)

        for e in [35]:
            t = train_int2(filters, kernels, strides, paddings, 'restimagenet_wresnet_int2_' + str(e),
                           eps_val(0.003), lr_val, step_size=0.1, adv_steps=7, batch_size=64, lam=0.01,
                           restimagenet=True, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [35]:
            t = train_int2(filters, kernels, strides, paddings, 'restimagenet_wresnet_int2adv_' + str(e),
                           eps_val(0.003),
                           lr_val, adv=True, step_size=0.1, adv_steps=7, batch_size=128, lam=0.01, restimagenet=True,
                           EPOCHS=e)
            times.append(t)
            print(times)
    # Find training times by training for one epoch
    elif part == 6:
        # One epoch runtimes
        def lr_val(step):
            return 0.0001


        def eps_val(eps):
            def f(step):
                return eps

            return f


        # MNIST Small
        filters = [16, 32, 100]
        kernels = [4, 4, 7]
        strides = [2, 2, 1]
        paddings = ['SAME', 'SAME', 'SAME']

        for e in [1]:
            t = train_normal(filters, kernels, strides, paddings, 'mnist_smallgap_normal_' + str(e), lr_val,
                             batch_size=50, EPOCHS=e)
            times.append(t)
            print(times)

        for e in [1]:
            t = train_adv(filters, kernels, strides, paddings, 'mnist_smallgap_adv_' + str(e), eps_val(0.3), lr_val,
                          step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e)
            times.append(t)
            print(times)

        for e in [1]:
            t = train_trades(filters, kernels, strides, paddings, 'mnist_smallgap_trades_' + str(e), eps_val(0.3),
                             lr_val, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e)
            times.append(t)
            print(times)

        for e in [1]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_int_' + str(e), eps_val(0.3), lr_val,
                          lam=0.01, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e)
            times.append(t)
            print(times)

        for e in [1]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_intadv_' + str(e), eps_val(0.3), lr_val,
                          adv=True, lam=0.01, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e)
            times.append(t)
            print(times)

        for e in [1]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_intone_' + str(e), eps_val(0.3), lr_val,
                          oneclass=True, lam=0.01, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e)
            times.append(t)
            print(times)

        for e in [1]:
            t = train_ig(filters, kernels, strides, paddings, 'mnist_smallgap_ig_' + str(e), eps_val(0.3), lr_val,
                         step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e)
            times.append(t)
            print(times)

        for e in [1]:
            t = train_ig(filters, kernels, strides, paddings, 'mnist_smallgap_igsum_' + str(e), eps_val(0.3), lr_val,
                         adv=True, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e)
            times.append(t)
            print(times)

        for e in [1]:
            t = train_ig(filters, kernels, strides, paddings, 'mnist_smallgap_intig_' + str(e), eps_val(0.3), lr_val,
                         twoclass=True, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e)
            times.append(t)
            print(times)

        for e in [1]:
            t = train_ig(filters, kernels, strides, paddings, 'mnist_smallgap_intigadv_' + str(e), eps_val(0.3), lr_val,
                         twoclass=True, adv=True, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e)
            times.append(t)
            print(times)

        for e in [1]:
            t = train_int2(filters, kernels, strides, paddings, 'mnist_smallgap_int2_' + str(e), eps_val(0.3),
                           lr_val, lam=0.01, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e,
                           device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_int2(filters, kernels, strides, paddings, 'mnist_smallgap_int2adv_' + str(e),
                           eps_val(0.3), lr_val, adv=True, lam=0.01, step_size=0.01, adv_steps=40, batch_size=50,
                           EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        # MNIST Pool
        filters = [32, 'pool', 64, 'pool']
        kernels = [5, 2, 5, 2]
        strides = [1, 2, 1, 2]
        paddings = ['SAME', 'SAME', 'SAME', 'SAME']

        for e in [1]:
            t = train_normal(filters, kernels, strides, paddings, 'mnist_pool_normal_' + str(e), lr_val, batch_size=50,
                             EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_adv(filters, kernels, strides, paddings, 'mnist_pool_adv_' + str(e), eps_val(0.3), lr_val,
                          step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_trades(filters, kernels, strides, paddings, 'mnist_pool_trades_' + str(e), eps_val(0.3), lr_val,
                             step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_pool_int_' + str(e), eps_val(0.3), lr_val,
                          step_size=0.01, adv_steps=40, batch_size=50, lam=0.01, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_pool_intadv_' + str(e), eps_val(0.3), lr_val,
                          adv=True, step_size=0.01, adv_steps=40, batch_size=50, lam=0.01, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_pool_intone_' + str(e), eps_val(0.3), lr_val,
                          oneclass=True, step_size=0.01, adv_steps=40, batch_size=50, lam=0.01, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_ig(filters, kernels, strides, paddings, 'mnist_pool_ig_' + str(e), eps_val(0.3), lr_val,
                         step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_ig(filters, kernels, strides, paddings, 'mnist_pool_igsum_' + str(e), eps_val(0.3), lr_val,
                         adv=True, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_ig(filters, kernels, strides, paddings, 'mnist_pool_intig_' + str(e), eps_val(0.3), lr_val,
                         twoclass=True, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_ig(filters, kernels, strides, paddings, 'mnist_pool_intigadv_' + str(e), eps_val(0.3), lr_val,
                         twoclass=True, adv=True, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e,
                         device='/device:GPU:0')
            times.append(t)
            print(times)

        # CIFAR Small
        filters = [16, 32, 100]
        kernels = [4, 4, 7]
        strides = [2, 2, 1]
        paddings = ['SAME', 'SAME', 'SAME']

        for e in [1]:
            t = train_normal(filters, kernels, strides, paddings, 'cifar_smallgap_normal_' + str(e), lr_val,
                             batch_size=128, cifar=True, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_adv(filters, kernels, strides, paddings, 'cifar_smallgap_adv_' + str(e), eps_val(8 / 255), lr_val,
                          step_size=2 / 255, adv_steps=10, batch_size=128, cifar=True, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_trades(filters, kernels, strides, paddings, 'cifar_smallgap_trades_' + str(e), eps_val(8 / 255),
                             lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, cifar=True, EPOCHS=e,
                             device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_int_' + str(e), eps_val(8 / 255), lr_val,
                          step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.01, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_intadv_' + str(e), eps_val(8 / 255),
                          lr_val, adv=True, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.01, cifar=True,
                          EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_intone_' + str(e), eps_val(8 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.01, oneclass=True, cifar=True,
                          EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_ig(filters, kernels, strides, paddings, 'cifar_smallgap_ig_' + str(e), eps_val(8 / 255), lr_val,
                         step_size=2 / 255, adv_steps=10, batch_size=128, cifar=True, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_ig(filters, kernels, strides, paddings, 'cifar_smallgap_igsum_' + str(e), eps_val(8 / 255),
                         lr_val, adv=True, step_size=2 / 255, adv_steps=10, batch_size=128, cifar=True, EPOCHS=e,
                         device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_ig(filters, kernels, strides, paddings, 'cifar_smallgap_intig_' + str(e), eps_val(8 / 255),
                         lr_val, twoclass=True, step_size=2 / 255, adv_steps=10, batch_size=128, cifar=True, EPOCHS=e,
                         device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_ig(filters, kernels, strides, paddings, 'cifar_smallgap_intigadv_' + str(e), eps_val(8 / 255),
                         lr_val, twoclass=True, adv=True, step_size=2 / 255, adv_steps=10, batch_size=128, cifar=True,
                         EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        # CIFAR WResnet
        filters = [16] + 10 * [16] + 10 * [32] + 10 * [64]
        kernels = 31 * [3]
        strides = [1] + 5 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1']
        paddings = 31 * ['SAME']

        for e in [1]:
            t = train_normal(filters, kernels, strides, paddings, 'cifar_wresnet_normal_' + str(e), lr_val,
                             batch_size=128, cifar=True, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_adv(filters, kernels, strides, paddings, 'cifar_wresnet_adv_' + str(e), eps_val(8 / 255), lr_val,
                          step_size=2 / 255, adv_steps=10, batch_size=128, cifar=True, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_trades(filters, kernels, strides, paddings, 'cifar_wresnet_trades_' + str(e), eps_val(8 / 255),
                             lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, cifar=True, EPOCHS=e,
                             device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_wresnet_int_' + str(e), eps_val(8 / 255), lr_val,
                          step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.01, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_wresnet_intadv_' + str(e), eps_val(8 / 255),
                          lr_val, adv=True, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.01, cifar=True,
                          EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [1]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_wresnet_intone_' + str(e), eps_val(8 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.01, oneclass=True, cifar=True,
                          EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        # Restricted Imagenet WResnet
        filters = [16] + 10 * [16] + 10 * [32] + 10 * [64]
        kernels = 31 * [3]
        strides = [1] + 5 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1']
        paddings = 31 * ['SAME']

        for e in [1]:
            t = train_normal(filters, kernels, strides, paddings, 'restimagenet_wresnet_normal_' + str(e), lr_val,
                             batch_size=64, restimagenet=True, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)
    elif part == 7:  # Varying eps + gamma experiments
        def lr_val(step):
            return 0.0001


        def eps_val(eps):
            def f(step):
                if step <= 2000:
                    return 0
                elif step <= 10000:
                    return eps * (step - 2000) / 8000
                else:
                    return eps

            return f


        # Steps = 40 for MNIST
        # batch size = 50 for MNIST
        # step_size = 0.01 for MNIST

        # MNIST Small
        filters = [16, 32, 100]
        kernels = [4, 4, 7]
        strides = [2, 2, 1]
        paddings = ['SAME', 'SAME', 'SAME']

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_int001_2_' + str(e), eps_val(0.2),
                          lr_val, lam=0.001, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_int005_2_' + str(e), eps_val(0.2),
                          lr_val, lam=0.005, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_int01_2_' + str(e), eps_val(0.2), lr_val,
                          lam=0.01, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_int015_2_' + str(e), eps_val(0.2),
                          lr_val, lam=0.015, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_int02_2_' + str(e), eps_val(0.2), lr_val,
                          lam=0.02, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_adv(filters, kernels, strides, paddings, 'mnist_smallgap_adv_2_' + str(e), eps_val(0.2), lr_val,
                          step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_int001_3_' + str(e), eps_val(0.3),
                          lr_val, lam=0.001, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_int005_3_' + str(e), eps_val(0.3),
                          lr_val, lam=0.005, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_int01_3_' + str(e), eps_val(0.3), lr_val,
                          lam=0.01, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_int015_3_' + str(e), eps_val(0.3),
                          lr_val, lam=0.015, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_int02_3_' + str(e), eps_val(0.3), lr_val,
                          lam=0.02, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_adv(filters, kernels, strides, paddings, 'mnist_smallgap_adv_3_' + str(e), eps_val(0.3), lr_val,
                          step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_int001_4_' + str(e), eps_val(0.4),
                          lr_val, lam=0.001, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_int005_4_' + str(e), eps_val(0.4),
                          lr_val, lam=0.005, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_int01_4_' + str(e), eps_val(0.4), lr_val,
                          lam=0.01, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_int015_4_' + str(e), eps_val(0.4),
                          lr_val, lam=0.015, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_int(filters, kernels, strides, paddings, 'mnist_smallgap_int02_4_' + str(e), eps_val(0.4), lr_val,
                          lam=0.02, step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [100]:
            t = train_adv(filters, kernels, strides, paddings, 'mnist_smallgap_adv_4_' + str(e), eps_val(0.4), lr_val,
                          step_size=0.01, adv_steps=40, batch_size=50, EPOCHS=e, device='/device:GPU:0')
            times.append(t)
            print(times)


        def lr_val(step):
            if step <= 40000:
                return 0.001
            elif step <= 60000:
                return 0.0001
            else:
                return 0.00001


        def eps_val(eps):
            def f(step):
                if step <= 5000:
                    return 0
                elif step <= 70000:
                    return eps * (step - 5000) / 65000
                else:
                    return eps

            return f


        # Steps = 10 for CIFAR
        # batch size = 128 for CIFAR
        # step_size = 2/255 for CIFAR

        # CIFAR Small
        filters = [16, 32, 100]
        kernels = [4, 4, 7]
        strides = [2, 2, 1]
        paddings = ['SAME', 'SAME', 'SAME']

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_int001_6_' + str(e), eps_val(6 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.001, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_int005_6_' + str(e), eps_val(6 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.005, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_int01_6_' + str(e), eps_val(6 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.01, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_int015_6_' + str(e), eps_val(6 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.015, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_int02_6_' + str(e), eps_val(6 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.02, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_adv(filters, kernels, strides, paddings, 'cifar_smallgap_adv_6_' + str(e), eps_val(6 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_int001_8_' + str(e), eps_val(8 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.001, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_int005_8_' + str(e), eps_val(8 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.005, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_int01_8_' + str(e), eps_val(8 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.01, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_int015_8_' + str(e), eps_val(8 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.015, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_int02_8_' + str(e), eps_val(8 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.02, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_adv(filters, kernels, strides, paddings, 'cifar_smallgap_adv_8_' + str(e), eps_val(8 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_int001_10_' + str(e), eps_val(10 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.001, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_int005_10_' + str(e), eps_val(10 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.005, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_int01_10_' + str(e), eps_val(10 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.01, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_int015_10_' + str(e), eps_val(10 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.015, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_int(filters, kernels, strides, paddings, 'cifar_smallgap_int02_10_' + str(e), eps_val(10 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, lam=0.02, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)

        for e in [200]:
            t = train_adv(filters, kernels, strides, paddings, 'cifar_smallgap_adv_10_' + str(e), eps_val(10 / 255),
                          lr_val, step_size=2 / 255, adv_steps=10, batch_size=128, cifar=True, EPOCHS=e,
                          device='/device:GPU:0')
            times.append(t)
            print(times)
