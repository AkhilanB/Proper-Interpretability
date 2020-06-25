"""
eval.py
Evaluate networks on aai, uar and pgd attacks
Copyright (C) 2020, Akhilan Boopathy <akhilan@mit.edu>
                    Sijia Liu <Sijia.Liu@ibm.com>
                    Gaoyuan Zhang <Gaoyuan.Zhang@ibm.com>
                    Cynthia Liu <cynliu98@mit.edu>
                    Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
                    Shiyu Chang <Shiyu.Chang@ibm.com>
                    Luca Daniel <luca@mit.edu>
"""

# Run pgd attack

import tensorflow as tf
import numpy as np
from setup_mnist import MNIST
from setup_cifar import CIFAR
from setup_restrictedimagenet import RestrictedImagenet as restImagenet
from load_model import load_model
import time as timer
import tensorflow as tf
import random
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib
import sys

part = 1

if len(sys.argv) > 1:
    part = int(sys.argv[1])


def crop(data):
    return data[:, 4:60, 4:60, :]


def display_cifar(x, name=None):
    dim = int(np.sqrt(x.size // 3))
    x = np.reshape(x, (dim, dim, 3))
    plt.imshow(x, vmin=0.0, vmax=1.0)
    plt.axis('off')
    if name is None:
        plt.show()
    else:
        plt.savefig('gen_images/' + name + '.png', bbox_inches='tight')
        plt.close()


def display_mnist(x, name=None):
    dim = int(np.sqrt(x.size))
    two_d = (np.reshape(x, (dim, dim)) * 255)
    plt.imshow(two_d, cmap='gray', vmin=0, vmax=255)
    if name is None:
        plt.show()
    else:
        plt.savefig('gen_images/' + name + '.png', bbox_inches='tight')
        plt.close()


def display_restimagenet(x, name=None):
    dim = int(np.sqrt(x.size // 3))
    x = np.reshape(x, (dim, dim, 3))
    plt.imshow(x, vmin=0.0, vmax=1.0)
    plt.axis('off')
    if name is None:
        plt.show()
    else:
        plt.savefig('gen_images/' + name + '.png', bbox_inches='tight')
        plt.close()


def display_cam_restimagenet(x, cam, name=None):
    x = np.transpose(x, (1, 0, 2))
    cam = np.transpose(cam, (1, 0))
    scale = x.shape[1] / cam.shape[1]
    upsample_cam = np.repeat(np.repeat(cam, scale, axis=0), scale, axis=1)

    cmap = plt.get_cmap('coolwarm')
    norm = matplotlib.colors.DivergingNorm(vmin=-1, vcenter=0, vmax=1)
    cam_rgb = cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(upsample_cam)[:, :, :3]

    plt.imshow(0.5 * cam_rgb + 0.5 * x)
    if name is None:
        plt.show()
    else:
        plt.savefig('gen_images/' + name + '.png', bbox_inches='tight')
        plt.close()


def display_cam_cifar(x, cam, name=None):
    scale = x.shape[1] / cam.shape[1]
    upsample_cam = np.repeat(np.repeat(cam, scale, axis=0), scale, axis=1)

    scale = np.maximum(np.max(upsample_cam), -np.min(upsample_cam))
    cmap = plt.get_cmap('coolwarm')
    norm = matplotlib.colors.DivergingNorm(vmin=-1, vcenter=0, vmax=1)
    cam_rgb = cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(upsample_cam)[:, :, :3]
    plt.axis('off')
    plt.imshow(0.5 * cam_rgb + 0.5 * x)
    if name is None:
        plt.show()
    else:
        plt.savefig('gen_images/' + name + '.png', bbox_inches='tight')
        plt.close()


def display_cam_mnist(x, cam, name=None):
    scale = x.shape[1] / cam.shape[1]
    upsample_cam = np.repeat(np.repeat(cam, scale, axis=0), scale, axis=1)

    x = x * np.ones((x.shape[1], x.shape[2], 3))

    cmap = plt.get_cmap('coolwarm')
    norm = matplotlib.colors.DivergingNorm(vmin=-60, vcenter=0, vmax=60)
    cam_rgb = cm.ScalarMappable(norm=norm, cmap=cmap).to_rgba(upsample_cam)[:, :, :3]

    plt.imshow(0.5 * cam_rgb + 0.5 * x)
    if name is None:
        plt.show()
    else:
        plt.savefig('gen_images/' + name + '.png', bbox_inches='tight')
        plt.close()


# Visualizes cam
def vis_cam(file_name, sess, filters, kernels, strides, paddings, num_image=5, cifar=False,
            restimagenet=False):
    np.random.seed(99)
    tf.set_random_seed(99)
    random.seed(99)

    classes = 10
    targets = tf.placeholder('float', shape=(None, 10))
    labels = tf.placeholder('float', shape=(None, 10))
    if cifar:
        data = CIFAR()
        inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
        x_test = data.test_data + 0.5
        y_test = data.test_labels
    elif restimagenet:
        data = restImagenet()
        inputs = tf.placeholder('float', shape=(None, 224, 224, 3))
        labels = tf.placeholder('float', shape=(None, 9))
        targets = tf.placeholder('float', shape=(None, 9))
        x_test = data.validation_data / 255
        y_test = data.validation_labels
        classes = 9
    else:
        data = MNIST()
        inputs = tf.placeholder('float', shape=(None, 28, 28, 1))
        x_test = data.test_data + 0.5
        y_test = data.test_labels
    model = load_model(file_name, sess, filters, kernels, strides, paddings)

    out, rep, cam = model.predict(inputs)

    cam_true = tf.reduce_sum(tf.reshape(labels, (-1, 1, 1, classes)) * cam, axis=3)

    np.random.seed(99)
    idx = np.random.permutation(x_test.shape[0])[:num_image]

    start_time = timer.time()
    true = np.argmax(y_test[idx], axis=1)

    x_nat = x_test[idx]

    feed_dict_attack = {inputs: x_nat, labels: y_test[idx, :]}
    cam_val = sess.run(cam_true, feed_dict=feed_dict_attack)

    if cifar:
        for i in range(num_image):
            display_cifar(x_nat[i, :, :, :], name='img_' + str(i) + '_' + file_name)  # Natural Image
            display_cam_cifar(x_nat[i, :, :, :], cam_val[i, :, :],
                              name='cam_' + str(i) + '_' + file_name)  # Natural image, true label CAM
    elif restimagenet:
        for i in range(num_image):
            display_restimagenet(x_nat[i, :, :, :], name='img_' + str(i) + '_' + file_name)  # Natural Image
            display_cam_restimagenet(x_nat[i, :, :, :], cam_val[i, :, :],
                                     name='cam_' + str(i) + '_' + file_name)  # Natural image, true label CAM
    else:
        for i in range(num_image):
            display_mnist(x_nat[i, :, :, :], name='img_' + str(i) + '_' + file_name)  # Natural Image
            display_cam_mnist(x_nat[i, :, :, :], cam_val[i, :, :],
                              name='cam_' + str(i) + '_' + file_name)  # Natural image, true label CAM


def intr_topk(network, base_net, sess, filters, kernels, strides, paddings, k=8, num_image=200, cifar=False,
              restimagenet=False):
    np.random.seed(99)
    tf.set_random_seed(99)
    random.seed(99)

    classes = 10
    targets = tf.placeholder('float', shape=(None, 10))
    labels = tf.placeholder('float', shape=(None, 10))
    if cifar:
        data = CIFAR()
        inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
        x_test = data.test_data + 0.5
        y_test = data.test_labels
    elif restimagenet:
        data = restImagenet()
        inputs = tf.placeholder('float', shape=(None, 224, 224, 3))
        labels = tf.placeholder('float', shape=(None, 9))
        targets = tf.placeholder('float', shape=(None, 9))
        x_test = data.validation_data / 255
        y_test = data.validation_labels
        classes = 9
    else:
        data = MNIST()
        inputs = tf.placeholder('float', shape=(None, 28, 28, 1))
        x_test = data.test_data + 0.5
        y_test = data.test_labels
    base_model = load_model(base_net, sess, filters, kernels, strides, paddings)
    model = load_model(network, sess, filters, kernels, strides, paddings)

    _, _, cam = model.predict(inputs)
    cam_true = tf.reduce_sum(tf.reshape(labels, (-1, 1, 1, classes)) * cam, axis=3)

    _, _, base_cam = base_model.predict(inputs)
    base_cam_true = tf.reduce_sum(tf.reshape(labels, (-1, 1, 1, classes)) * base_cam, axis=3)

    np.random.seed(99)
    idx = np.random.permutation(x_test.shape[0])[:num_image]

    start_time = timer.time()
    true = np.argmax(y_test[idx], axis=1)

    x_nat = x_test[idx]

    feed_dict_attack = {inputs: x_nat, labels: y_test[idx, :]}
    cam_true_val, base_cam_true_val = sess.run([cam_true, base_cam_true], feed_dict=feed_dict_attack)

    # Get top-k intersection
    cam_true_val = np.reshape(cam_true_val, (cam_true_val.shape[0], -1))
    base_cam_true_val = np.reshape(base_cam_true_val, (base_cam_true_val.shape[0], -1))

    cam_kth_largest = np.partition(cam_true_val, -k)[-k]
    base_cam_kth_largest = np.partition(base_cam_true_val, -k)[-k]

    cam_kth_largest = cam_kth_largest[:, np.newaxis]
    base_cam_kth_largest = base_cam_kth_largest[:, np.newaxis]

    cam_k_largest = np.where(cam_true_val > cam_kth_largest, 1, 0)
    base_cam_k_largest = np.where(base_cam_true_val > base_cam_kth_largest, 1, 0)

    cam_size = cam_k_largest.shape[1]

    return np.mean(np.sum(cam_k_largest * base_cam_k_largest, axis=1)) / cam_size


from scipy.stats import kendalltau


def tau_corr(true, adv):
    corr_sum = 0
    for i in range(true.shape[2]):
        corr_sum += kendalltau(true[:, :, i], adv[:, :, i])[0]
    return corr_sum / int(true.shape[2])


# Runs topk AAI, returns Tau rank correlation
def aai_topk(file_name, sess, filters, kernels, strides, paddings, eps, k=8, steps=200, kappa=0.1, num_image=200,
             ig=False, plusplus=False, cifar=False,
             restimagenet=False):
    if eps == 0:
        return 0

    np.random.seed(99)
    tf.set_random_seed(99)
    random.seed(99)

    classes = 10
    targets = tf.placeholder('float', shape=(None, 10))
    labels = tf.placeholder('float', shape=(None, 10))
    if cifar:
        data = CIFAR()
        inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
        inputs_adv = tf.placeholder('float', shape=(None, 32, 32, 3))
        x_test = data.test_data + 0.5
        y_test = data.test_labels
    elif restimagenet:
        data = restImagenet()
        inputs = tf.placeholder('float', shape=(None, 224, 224, 3))
        inputs_adv = tf.placeholder('float', shape=(None, 224, 224, 3))
        labels = tf.placeholder('float', shape=(None, 9))
        targets = tf.placeholder('float', shape=(None, 9))
        x_test = data.validation_data / 255
        y_test = data.validation_labels
        classes = 9
    else:
        data = MNIST()
        inputs = tf.placeholder('float', shape=(None, 28, 28, 1))
        inputs_adv = tf.placeholder('float', shape=(None, 28, 28, 1))
        x_test = data.test_data + 0.5
        y_test = data.test_labels
    model = load_model(file_name, sess, filters, kernels, strides, paddings)

    if ig:
        out, rep, cam = model.predict(inputs)
        ig_true, ig_targ = model.ig(inputs, labels), model.ig(inputs, targets)
        out_adv, rep_adv, cam_adv = model.predict(inputs_adv)
        ig_true_adv, ig_targ_adv = model.ig(inputs_adv, labels), model.ig(inputs_adv, targets)

        _, targ_idx = tf.nn.top_k(tf.layers.flatten(ig_targ), k)
        _, true_idx = tf.nn.top_k(tf.layers.flatten(ig_true), k)

        ig_targ_adv_topk = tf.gather(tf.layers.flatten(ig_targ_adv), targ_idx, batch_dims=1)
        ig_true_adv_topk = tf.gather(tf.layers.flatten(ig_true_adv), true_idx, batch_dims=1)

        loss = -tf.reduce_mean(tf.reduce_sum(ig_true_adv_topk, axis=1))
    else:
        out, rep, cam = model.predict(inputs)
        if plusplus:
            cam = model.campp(out, rep)
        out_adv, rep_adv, cam_adv = model.predict(inputs_adv)
        if plusplus:
            cam_adv = model.campp(out_adv, rep_adv)

        cam_targ_adv = tf.reduce_sum(tf.reshape(targets, (-1, 1, 1, classes)) * cam_adv, axis=3)
        cam_targ = tf.reduce_sum(tf.reshape(targets, (-1, 1, 1, classes)) * cam, axis=3)
        cam_true_adv = tf.reduce_sum(tf.reshape(labels, (-1, 1, 1, classes)) * cam_adv, axis=3)
        cam_true = tf.reduce_sum(tf.reshape(labels, (-1, 1, 1, classes)) * cam, axis=3)

        _, targ_idx = tf.nn.top_k(tf.layers.flatten(cam_targ), k)
        _, true_idx = tf.nn.top_k(tf.layers.flatten(cam_true), k)

        cam_targ_adv_topk = tf.gather(tf.layers.flatten(cam_targ_adv), targ_idx, batch_dims=1)
        cam_true_adv_topk = tf.gather(tf.layers.flatten(cam_true_adv), true_idx, batch_dims=1)

        loss = -tf.reduce_mean(tf.reduce_sum(cam_true_adv_topk, axis=1))

    grad = tf.gradients(loss, inputs_adv)[0]

    np.random.seed(99)
    idx = np.random.permutation(x_test.shape[0])[:num_image]

    start_time = timer.time()

    true = np.argmax(y_test[idx], axis=1)
    targ = (true + 1 + np.random.randint(classes - 1, size=num_image)) % classes
    targets_val = np.eye(classes)[targ]

    y_nat = y_test[idx].copy()
    x_adv = x_test[idx].copy()
    x_nat = x_test[idx].copy()

    perturb = np.random.uniform(-eps, eps, x_nat.shape)
    x_adv = x_nat + perturb
    x_adv = np.clip(x_adv, 0, 1)

    batch_size = 40

    x_adv_all = []
    for i in range(num_image // batch_size):
        x_nat_b = x_nat[batch_size * i:batch_size * (i + 1)]
        x_adv_b = x_adv[batch_size * i:batch_size * (i + 1)].copy()
        y_nat_b = y_nat[batch_size * i:batch_size * (i + 1)]
        targets_val_b = targets_val[batch_size * i:batch_size * (i + 1)]

        for j in range(steps):
            feed_dict_attack = {inputs: x_nat_b, inputs_adv: x_adv_b, labels: y_nat_b, targets: targets_val_b}
            grad_val = sess.run(grad, feed_dict=feed_dict_attack)
            delta = 0.01 * np.sign(grad_val)
            x_adv_b = x_adv_b + delta
            x_adv_b = np.clip(x_adv_b, x_nat_b - eps, x_nat_b + eps)
            x_adv_b = np.clip(x_adv_b, 0, 1)
        x_adv_all.append(x_adv_b)
    x_adv = np.concatenate(x_adv_all)
    feed_dict_attack = {inputs: x_nat, inputs_adv: x_adv, labels: y_nat, targets: targets_val}

    if not ig:
        cam_val, cam_adv_val, clean_out_val, adv_out_val = sess.run([cam, cam_adv, out, out_adv],
                                                                    feed_dict=feed_dict_attack)

        # rank corr
        corr_sum = 0
        total = 0
        for i in range(num_image):
            tr = true[i]
            trg = targ[i]
            if np.argmax(clean_out_val[i, :]) == np.argmax(adv_out_val[i, :]):  # Ignore if prediction changed
                c = tau_corr(cam_val[i, :, :, tr][:, :, np.newaxis], cam_adv_val[i, :, :, tr][:, :, np.newaxis])
                corr_sum += c
                total += 1
    else:
        ig_true_val, ig_true_adv_val, ig_targ_val, ig_targ_adv_val, clean_out_val, adv_out_val = sess.run(
            [ig_true, ig_targ, ig_true_adv, ig_targ_adv, out, out_adv], feed_dict=feed_dict_attack)

        # rank corr
        corr_sum = 0
        total = 0
        for i in range(num_image):
            tr = true[i]
            trg = targ[i]
            if np.argmax(clean_out_val[i, :]) == np.argmax(adv_out_val[i, :]):  # Ignore if prediction changed
                c = tau_corr(ig_true_val[i, :, :][:, :, np.newaxis], ig_true_adv_val[i, :, :][:, :, np.newaxis])
                corr_sum += c
                total += 1
    return corr_sum / max(total, 0.01)


# Runs minimum eps adversarial attack, returns margins and discrepancies
def min_pgd_attack(file_name, sess, filters, kernels, strides, paddings, kappa=0.1, num_image=100, plusplus=False,
                   rep_l=False, all_class=False, one_class=False, l1=True, cifar=False,
                   restimagenet=False):
    np.random.seed(99)
    tf.set_random_seed(99)
    random.seed(99)

    classes = 10
    targets = tf.placeholder('float', shape=(None, 10))
    labels = tf.placeholder('float', shape=(None, 10))
    if cifar:
        data = CIFAR()
        inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
        inputs_adv = tf.placeholder('float', shape=(None, 32, 32, 3))
        x_test = data.test_data + 0.5
        y_test = data.test_labels
    elif restimagenet:
        data = restImagenet()
        inputs = tf.placeholder('float', shape=(None, 224, 224, 3))
        inputs_adv = tf.placeholder('float', shape=(None, 224, 224, 3))
        labels = tf.placeholder('float', shape=(None, 9))
        targets = tf.placeholder('float', shape=(None, 9))
        x_test = data.validation_data / 255
        y_test = data.validation_labels
        classes = 9
    else:
        data = MNIST()
        inputs = tf.placeholder('float', shape=(None, 28, 28, 1))
        inputs_adv = tf.placeholder('float', shape=(None, 28, 28, 1))
        x_test = data.test_data + 0.5
        y_test = data.test_labels
    model = load_model(file_name, sess, filters, kernels, strides, paddings)

    out, rep, cam = model.predict(inputs)
    if plusplus:
        cam = model.campp(out, rep)
    out_adv, rep_adv, cam_adv = model.predict(inputs_adv)
    if plusplus:
        cam_adv = model.campp(out_adv, rep_adv)
    f_target = tf.reduce_sum(out_adv * targets, axis=1, keepdims=True)
    clean_f_target = tf.reduce_sum(out * targets, axis=1, keepdims=True)
    f_range = tf.stop_gradient(tf.reduce_max(out_adv) - tf.reduce_min(out_adv))
    clean_f_range = tf.reduce_max(out) - tf.reduce_min(out)
    all_margin = out_adv - f_range * targets - f_target
    margin = tf.maximum(tf.reduce_max(out_adv - f_range * targets - f_target, axis=1), -kappa)
    clean_margin = tf.reduce_max(out - clean_f_range * targets - clean_f_target, axis=1)

    if l1:
        rep_diff = tf.reduce_sum(tf.abs(tf.layers.flatten(rep_adv - rep)), axis=1)
        cam_diff = tf.reduce_sum(tf.abs(tf.layers.flatten(cam_adv - cam)), axis=1) / classes
    else:
        rep_diff = tf.reduce_sum(tf.layers.flatten(rep_adv - rep) ** 2, axis=1)
        cam_diff = tf.reduce_sum(tf.layers.flatten(cam_adv - cam) ** 2, axis=1) / classes

    pred = tf.argmax(out, axis=1)
    cam_targ_adv = tf.reduce_sum(tf.reshape(targets, (-1, 1, 1, classes)) * cam_adv, axis=3)
    cam_targ = tf.reduce_sum(tf.reshape(targets, (-1, 1, 1, classes)) * cam, axis=3)
    cam_true_adv = tf.reduce_sum(tf.reshape(labels, (-1, 1, 1, classes)) * cam_adv, axis=3)
    cam_true = tf.reduce_sum(tf.reshape(labels, (-1, 1, 1, classes)) * cam, axis=3)

    if not all_class:
        if one_class:
            if l1:
                cam_diff = tf.reduce_sum(tf.abs(tf.layers.flatten(cam_true_adv - cam_true)), axis=1)
            else:
                cam_diff = tf.reduce_sum(tf.layers.flatten((cam_true_adv - cam_true) ** 2), axis=1)
        else:
            if l1:
                cam_diff = 0.5 * tf.reduce_sum(tf.abs(tf.layers.flatten(cam_targ_adv - cam_targ)) + tf.abs(
                    tf.layers.flatten(cam_true_adv - cam_true)), axis=1)
            else:
                cam_diff = 0.5 * tf.reduce_sum(tf.layers.flatten((cam_targ_adv - cam_targ) ** 2) + tf.layers.flatten(
                    (cam_true_adv - cam_true) ** 2), axis=1)

    margin_loss = tf.reduce_mean(margin)
    gamma = tf.placeholder('float', shape=(None,))
    if rep_l:
        loss = margin_loss + tf.reduce_mean(gamma * rep_diff)
    else:
        loss = margin_loss + tf.reduce_mean(gamma * cam_diff)

    grad = tf.gradients(loss, inputs_adv)[0]

    np.random.seed(99)
    idx = np.random.permutation(x_test.shape[0])[:num_image]

    start_time = timer.time()
    true = np.argmax(y_test[idx], axis=1)
    targ = (true + 1 + np.random.randint(classes - 1, size=num_image)) % classes
    targets_val = np.eye(classes)[targ]

    attacks = []

    # Adversarial attack
    eps = 0.05 * np.ones(num_image)
    eps_min = np.zeros(num_image)
    eps_max = np.ones(num_image)
    for i in range(10):
        x_adv = x_test[idx].copy()
        x_nat = x_test[idx].copy()
        for j in range(10):
            feed_dict_attack = {inputs: x_nat, inputs_adv: x_adv, targets: targets_val, labels: y_test[idx, :],
                                gamma: np.zeros(num_image)}
            grad_val = sess.run(grad, feed_dict=feed_dict_attack)
            delta = 0.2 * np.clip(1000 * grad_val, -1, 1) * eps[:, np.newaxis, np.newaxis, np.newaxis]
            x_adv = x_adv - delta
            x_adv = np.clip(x_adv, x_nat - eps[:, np.newaxis, np.newaxis, np.newaxis],
                            x_nat + eps[:, np.newaxis, np.newaxis, np.newaxis])
            x_adv = np.clip(x_adv, 0, 1)
        feed_dict_attack = {inputs: x_nat, inputs_adv: x_adv, targets: targets_val, labels: y_test[idx, :],
                            gamma: np.zeros(num_image)}

        margin_val = sess.run(margin, feed_dict=feed_dict_attack)
        for k in range(num_image):
            if margin_val[k] < 0:  # Success: Decrease eps
                eps_max[k] = eps[k]
                eps[k] = (eps_min[k] + eps_max[k]) / 2
            else:  # Failure: Increase eps
                eps_min[k] = eps[k]
                eps[k] = min(2 * eps[k], (eps_min[k] + eps_max[k]) / 2)
    x_adv = x_test[idx].copy()
    x_nat = x_test[idx].copy()
    eps = eps_max
    for j in range(10):
        feed_dict_attack = {inputs: x_nat, inputs_adv: x_adv, targets: targets_val, labels: y_test[idx, :],
                            gamma: np.zeros(num_image)}
        grad_val = sess.run(grad, feed_dict=feed_dict_attack)
        delta = 0.2 * np.clip(1000 * grad_val, -1, 1) * eps[:, np.newaxis, np.newaxis, np.newaxis]
        x_adv = x_adv - delta
        x_adv = np.clip(x_adv, x_nat - eps[:, np.newaxis, np.newaxis, np.newaxis],
                        x_nat + eps[:, np.newaxis, np.newaxis, np.newaxis])
        x_adv = np.clip(x_adv, 0, 1)
    eps_adv = eps
    attacks.append(x_adv)

    feed_dict_attack = {inputs: x_nat, inputs_adv: x_adv, targets: targets_val, labels: y_test[idx, :],
                        gamma: np.zeros(num_image)}
    cam_clean, cam_atk, cam_diff_val, clean_out_val, atk_out_val = sess.run([cam, cam_adv, cam_diff, out, out_adv],
                                                                            feed_dict=feed_dict_attack)

    f_t = np.sum(clean_out_val * y_test[idx, :], axis=1)
    f_tp = np.sum(clean_out_val * targets_val, axis=1)
    margin = 0.5 * (f_t - f_tp)
    discrepancy = cam_diff_val / (cam_clean.shape[1] * cam_clean.shape[
        2])  # Divide by shape dims since cam actually uses average pooling instead of sum
    return margin, discrepancy


# Runs standard pgd attack, returns PGD accuracy
def pgd_attack(file_name, sess, filters, kernels, strides, paddings, eps, steps=200, num_image=200,
               cifar=False, restimagenet=False):
    np.random.seed(99)
    tf.set_random_seed(99)
    random.seed(99)

    classes = 10
    targets = tf.placeholder('float', shape=(None, 10))
    labels = tf.placeholder('float', shape=(None, 10))
    if cifar:
        data = CIFAR()
        inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
        inputs_adv = tf.placeholder('float', shape=(None, 32, 32, 3))
        x_test = data.test_data + 0.5
        y_test = data.test_labels
    elif restimagenet:
        data = restImagenet()
        inputs = tf.placeholder('float', shape=(None, 224, 224, 3))
        inputs_adv = tf.placeholder('float', shape=(None, 224, 224, 3))
        labels = tf.placeholder('float', shape=(None, 9))
        targets = tf.placeholder('float', shape=(None, 9))
        x_test = data.validation_data / 255
        y_test = data.validation_labels
        classes = 9
    else:
        data = MNIST()
        inputs = tf.placeholder('float', shape=(None, 28, 28, 1))
        inputs_adv = tf.placeholder('float', shape=(None, 28, 28, 1))
        x_test = data.test_data + 0.5
        y_test = data.test_labels
    model = load_model(file_name, sess, filters, kernels, strides, paddings)

    out, rep, cam = model.predict(inputs)
    out_adv, rep_adv, cam_adv = model.predict(inputs_adv)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=out_adv, labels=labels))

    grad = tf.gradients(loss, inputs_adv)[0]

    np.random.seed(99)
    idx = np.random.permutation(x_test.shape[0])[:num_image]

    start_time = timer.time()

    y_nat = y_test[idx].copy()
    x_adv = x_test[idx].copy()
    x_nat = x_test[idx].copy()
    if eps != 0:
        for j in range(steps):
            feed_dict_attack = {inputs_adv: x_adv, labels: y_nat}
            grad_val = sess.run(grad, feed_dict=feed_dict_attack)
            delta = 0.01 * np.sign(grad_val)
            x_adv = x_adv + delta
            x_adv = np.clip(x_adv, x_nat - eps, x_nat + eps)
            x_adv = np.clip(x_adv, 0, 1)
        feed_dict_attack = {inputs_adv: x_adv, labels: y_nat}
    else:
        feed_dict_attack = {inputs_adv: x_nat, labels: y_nat}
    out_adv_val = sess.run(out_adv, feed_dict=feed_dict_attack)

    success = 0
    for k in range(num_image):
        if np.argmax(out_adv_val[k, :]) != np.argmax(y_nat[k, :]):  # Success
            success += 1
    return 1 - (success / num_image)


from advex_uar.examples.uar import run


# Runs unforseen attacks, returns accuracies
def uar_attack(file_name, filters, kernels, strides, paddings, num_image=200, cifar=False,
               restimagenet=False):
    np.random.seed(99)
    tf.set_random_seed(99)
    random.seed(99)

    results = []
    for attack in ['fog', 'gabor', 'snow', 'jpeg_linf', 'jpeg_l2', 'jpeg_l1']:
        r = run(attack, file_name, filters, kernels, strides, paddings, num_image=num_image, cifar=cifar,
                restimagenet=restimagenet)
        results.append(r)
    return ', '.join([str(r) for r in results])


if __name__ == '__main__':
    with tf.Session() as sess:
        if part == 1:  # Main evaluations
            # Defense: PGD Accuracy

            print('PGD Accuracy')
            try:
                for steps in [1, 10, 100, 200]:
                    networks = ['mnist_smallgap_normal_100', 'mnist_smallgap_adv_100',
                                'mnist_smallgap_trades_100', 'mnist_smallgap_int_100',
                                'mnist_smallgap_intadv_100', 'mnist_smallgap_intone_100',
                                'mnist_smallgap_ig_100', 'mnist_smallgap_igsum_100',
                                'mnist_smallgap_intig_100', 'mnist_smallgap_intigadv_100',
                                'mnist_smallgap_int2_100', 'mnist_smallgap_int2adv_100']
                    filters = [16, 32, 100]
                    kernels = [4, 4, 7]
                    strides = [2, 2, 1]
                    paddings = ['SAME', 'SAME', 'SAME']
                    epss = [0, 0.05, 0.1, 0.2, 0.3, .35, .4]
                    for network in networks:
                        results = []
                        for eps in epss:
                            results.append(
                                str(pgd_attack(network, sess, filters, kernels, strides, paddings, eps, steps=steps)))
                        print(', '.join(results))

                    networks = ['mnist_pool_normal_100', 'mnist_pool_adv_100',
                                'mnist_pool_trades_100', 'mnist_pool_int_100',
                                'mnist_pool_intadv_100', 'mnist_pool_intone_100',
                                'mnist_pool_ig_100', 'mnist_pool_igsum_100',
                                'mnist_pool_intig_100', 'mnist_pool_intigadv_100',
                                'mnist_pool_int2_100', 'mnist_pool_int2adv_100']
                    filters = [32, 'pool', 64, 'pool']
                    kernels = [5, 2, 5, 2]
                    strides = [1, 2, 1, 2]
                    paddings = ['SAME', 'SAME', 'SAME', 'SAME']
                    epss = [0, 0.05, 0.1, 0.2, 0.3, .35, .4]
                    for network in networks:
                        results = []
                        for eps in epss:
                            results.append(
                                str(pgd_attack(network, sess, filters, kernels, strides, paddings, eps, steps=steps)))
                        print(', '.join(results))

                    networks = ['cifar_smallgap_normal_200', 'cifar_smallgap_adv_200',
                                'cifar_smallgap_trades_200', 'cifar_smallgap_int_200',
                                'cifar_smallgap_intadv_200', 'cifar_smallgap_intone_200',
                                'cifar_smallgap_ig_200', 'cifar_smallgap_igsum_200',
                                'cifar_smallgap_intig_200', 'cifar_smallgap_intigadv_200',
                                'cifar_smallgap_int2_200', 'cifar_smallgap_int2adv_200']
                    filters = [16, 32, 100]
                    kernels = [4, 4, 7]
                    strides = [2, 2, 1]
                    paddings = ['SAME', 'SAME', 'SAME']
                    epss = [0, 2 / 255, 4 / 255, 6 / 255, 8 / 255, 9 / 255, 10 / 255]
                    for network in networks:
                        results = []
                        for eps in epss:
                            results.append(str(
                                pgd_attack(network, sess, filters, kernels, strides, paddings, eps, cifar=True,
                                           steps=steps)))
                        print(', '.join(results))

                    networks = ['cifar_wresnet_normal_200', 'cifar_wresnet_adv_200',
                                'cifar_wresnet_trades_200', 'cifar_wresnet_int_200',
                                'cifar_wresnet_intadv_200', 'cifar_wresnet_intone_200',
                                'cifar_wresnet_int2_200', 'cifar_wresnet_int2adv_200']
                    filters = [16] + 10 * [16] + 10 * [32] + 10 * [64]
                    kernels = 31 * [3]
                    strides = [1] + 5 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1']
                    paddings = 31 * ['SAME']
                    epss = [0, 2 / 255, 4 / 255, 6 / 255, 8 / 255, 9 / 255, 10 / 255]
                    for network in networks:
                        results = []
                        for eps in epss:
                            results.append(str(
                                pgd_attack(network, sess, filters, kernels, strides, paddings, eps, cifar=True,
                                           steps=steps)))
                        print(', '.join(results))

                    networks = ['restimagenet_wresnet_normal_35', 'restimagenet_wresnet_adv_35',
                                'restimagenet_wresnet_int_35', 'restimagenet_wresnet_int2_35']
                    filters = [16] + 10 * [16] + 10 * [32] + 10 * [64]
                    kernels = 31 * [3]
                    strides = [1] + 5 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1']
                    paddings = 31 * ['SAME']
                    epss = [0, 2 / 255, 4 / 255, 6 / 255, 8 / 255, 9 / 255, 10 / 255]
                    for network in networks:
                        results = []
                        for eps in epss:
                            results.append(str(
                                pgd_attack(network, sess, filters, kernels, strides, paddings, eps, restimagenet=True,
                                           steps=steps)))
                        print(', '.join(results))
            except:
                print('Failed')

            # Defense: UAR Attack
            print('UAR Attack')
            try:
                networks = ['cifar_smallgap_normal_200', 'cifar_smallgap_adv_200',
                            'cifar_smallgap_trades_200', 'cifar_smallgap_int_200',
                            'cifar_smallgap_intadv_200', 'cifar_smallgap_intone_200',
                            'cifar_smallgap_ig_200', 'cifar_smallgap_igsum_200',
                            'cifar_smallgap_intig_200', 'cifar_smallgap_intigadv_200',
                            'cifar_smallgap_int2_200', 'cifar_smallgap_int2adv_200']
                filters = [16, 32, 100]
                kernels = [4, 4, 7]
                strides = [2, 2, 1]
                paddings = ['SAME', 'SAME', 'SAME']
                for network in networks:
                    print(uar_attack(network, filters, kernels, strides, paddings, cifar=True))

                networks = ['cifar_wresnet_normal_200', 'cifar_wresnet_adv_200',
                            'cifar_wresnet_trades_200', 'cifar_wresnet_int_200',
                            'cifar_wresnet_intadv_200', 'cifar_wresnet_intone_200',
                            'cifar_wresnet_int2_200', 'cifar_wresnet_int2adv_200']
                filters = [16] + 10 * [16] + 10 * [32] + 10 * [64]
                kernels = 31 * [3]
                strides = [1] + 5 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1']
                paddings = 31 * ['SAME']
                for network in networks:
                    print(uar_attack(network, filters, kernels, strides, paddings, cifar=True))

                    networks = ['restimagenet_wresnet_normal_100', 'restimagenet_wresnet_adv_100',
                                'restimagenet_wresnet_trades_100', 'restimagenet_wresnet_int_100',
                                'restimagenet_wresnet_intadv_100', 'restimagenet_wresnet_intone_100']
                    filters = [16] + 10 * [16] + 10 * [32] + 10 * [64]
                    kernels = 31 * [3]
                    strides = [1] + 5 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1']
                    paddings = 31 * ['SAME']
                    for network in networks:
                        print(uar_attack(network, filters, kernels, strides, paddings, restimagenet=True))
            except:
                print('Failed')

            # Defense: AAI, top-k CAM
            print('Top-k AAI CAM')
            try:
                networks = ['mnist_smallgap_normal_100', 'mnist_smallgap_adv_100',
                            'mnist_smallgap_trades_100', 'mnist_smallgap_int_100',
                            'mnist_smallgap_intadv_100', 'mnist_smallgap_intone_100',
                            'mnist_smallgap_ig_100', 'mnist_smallgap_igsum_100',
                            'mnist_smallgap_intig_100', 'mnist_smallgap_intigadv_100']
                networks = ['mnist_smallgap_int2_100', 'mnist_smallgap_int2adv_100']
                filters = [16, 32, 100]
                kernels = [4, 4, 7]
                strides = [2, 2, 1]
                paddings = ['SAME', 'SAME', 'SAME']
                epss = [0, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4]
                for network in networks:
                    results = []
                    for eps in epss:
                        results.append(str(aai_topk(network, sess, filters, kernels, strides, paddings, eps)))
                    print(', '.join(results))

                networks = ['mnist_pool_normal_100', 'mnist_pool_adv_100',
                            'mnist_pool_trades_100', 'mnist_pool_int_100',
                            'mnist_pool_intadv_100', 'mnist_pool_intone_100',
                            'mnist_pool_ig_100', 'mnist_pool_igsum_100']  # ,
                # 'mnist_pool_intig_100','mnist_pool_intigadv_100']
                networks = ['mnist_pool_int2_100', 'mnist_pool_int2adv_100']
                filters = [32, 'pool', 64, 'pool']
                kernels = [5, 2, 5, 2]
                strides = [1, 2, 1, 2]
                paddings = ['SAME', 'SAME', 'SAME', 'SAME']
                epss = [0, 0.05, 0.1, 0.2, 0.3, 0.35, 0.4]
                for network in networks:
                    results = []
                    for eps in epss:
                        results.append(str(aai_topk(network, sess, filters, kernels, strides, paddings, eps)))
                    print(', '.join(results))

                networks = ['cifar_smallgap_normal_200', 'cifar_smallgap_adv_200',
                            'cifar_smallgap_trades_200', 'cifar_smallgap_int_200',
                            'cifar_smallgap_intadv_200', 'cifar_smallgap_intone_200',
                            'cifar_smallgap_ig_200', 'cifar_smallgap_igsum_200',
                            'cifar_smallgap_intig_200', 'cifar_smallgap_intigadv_200']
                networks = ['cifar_smallgap_int2_200', 'cifar_smallgap_int2adv_200']
                filters = [16, 32, 100]
                kernels = [4, 4, 7]
                strides = [2, 2, 1]
                paddings = ['SAME', 'SAME', 'SAME']
                epss = [0, 2 / 255, 4 / 255, 6 / 255, 8 / 255, 9 / 255, 10 / 255]
                for network in networks:
                    results = []
                    for eps in epss:
                        results.append(
                            str(aai_topk(network, sess, filters, kernels, strides, paddings, eps, cifar=True)))
                    print(', '.join(results))

                networks = ['cifar_wresnet_normal_200', 'cifar_wresnet_adv_200',
                            'cifar_wresnet_trades_200', 'cifar_wresnet_int_200',
                            'cifar_wresnet_intadv_200', 'cifar_wresnet_intone_200']
                networks = ['cifar_wresnet_int2_200', 'cifar_wresnet_int2adv_200']
                filters = [16] + 10 * [16] + 10 * [32] + 10 * [64]
                kernels = 31 * [3]
                strides = [1] + 5 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1']
                paddings = 31 * ['SAME']
                epss = [0, 2 / 255, 4 / 255, 6 / 255, 8 / 255, 9 / 255, 10 / 255]
                for network in networks:
                    results = []
                    for eps in epss:
                        results.append(
                            str(aai_topk(network, sess, filters, kernels, strides, paddings, eps, cifar=True)))
                    print(', '.join(results))

                networks = ['restimagenet_wresnet_normal_35', 'restimagenet_wresnet_adv_35',
                            'restimagenet_wresnet_int_35', 'restimagenet_wresnet_int2_35']
                filters = [16] + 10 * [16] + 10 * [32] + 10 * [64]
                kernels = 31 * [3]
                strides = [1] + 5 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1']
                paddings = 31 * ['SAME']
                epss = [0, 2 / 255, 4 / 255, 6 / 255, 8 / 255, 9 / 255, 10 / 255]
                for network in networks:
                    results = []
                    for eps in epss:
                        results.append(
                            str(aai_topk(network, sess, filters, kernels, strides, paddings, eps, restimagenet=True)))
                    print(', '.join(results))
            except:
                print('Failed')

            # Defense: Visualize CAM
            try:
                networks = ['cifar_smallgap_normal_200', 'cifar_smallgap_adv_200',
                            'cifar_smallgap_int_200', 'cifar_smallgap_ig_200']
                filters = [16, 32, 100]
                kernels = [4, 4, 7]
                strides = [2, 2, 1]
                paddings = ['SAME', 'SAME', 'SAME']
                for network in networks:
                    vis_cam(network, sess, filters, kernels, strides, paddings, cifar=True)

                networks = ['cifar_wresnet_normal_200', 'cifar_wresnet_adv_200',
                            'cifar_wresnet_int_200']
                filters = [16] + 10 * [16] + 10 * [32] + 10 * [64]
                kernels = 31 * [3]
                strides = [1] + 5 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1']
                paddings = 31 * ['SAME']
                for network in networks:
                    vis_cam(network, sess, filters, kernels, strides, paddings, cifar=True)

                networks = ['restimagenet_wresnet_normal_100', 'restimagenet_wresnet_adv_100',
                            'restimagenet_wresnet_int_100']
                filters = [16] + 10 * [16] + 10 * [32] + 10 * [64]
                kernels = 31 * [3]
                strides = [1] + 5 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1']
                paddings = 31 * ['SAME']
                for network in networks:
                    vis_cam(network, sess, filters, kernels, strides, paddings, restimagenet=True)
            except:
                print('Failed')

            # Defense: Compute top-k intersection with normal CAM
            try:
                base_net = 'cifar_smallgap_normal_200'
                networks = ['cifar_smallgap_adv_200', 'cifar_smallgap_int_200', 'cifar_smallgap_ig_200']
                filters = [16, 32, 100]
                kernels = [4, 4, 7]
                strides = [2, 2, 1]
                paddings = ['SAME', 'SAME', 'SAME']
                for network in networks:
                    r = intr_topk(network, base_net, sess, filters, kernels, strides, paddings, cifar=True)
                    print(r)

                base_net = 'cifar_wresnet_normal_200'
                networks = ['cifar_wresnet_adv_200', 'cifar_wresnet_int_200']
                filters = [16] + 10 * [16] + 10 * [32] + 10 * [64]
                kernels = 31 * [3]
                strides = [1] + 5 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1']
                paddings = 31 * ['SAME']
                for network in networks:
                    r = intr_topk(network, base_net, sess, filters, kernels, strides, paddings, cifar=True)
                    print(r)

                base_net = 'restimagenet_wresnet_normal_100'
                networks = ['restimagenet_wresnet_adv_100', 'restimagenet_wresnet_int_100']
                filters = [16] + 10 * [16] + 10 * [32] + 10 * [64]
                kernels = 31 * [3]
                strides = [1] + 5 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1']
                paddings = 31 * ['SAME']
                for network in networks:
                    r = intr_topk(network, base_net, sess, filters, kernels, strides, paddings, restimagenet=True)
                    print(r)
            except:
                print('Failed')

        elif part == 2:  # Additional evaluations
            print('PGD Accuracy varying gamma/eps')
            try:
                for steps in [200]:
                    networks = ['mnist_smallgap_int001_2_100',
                                'mnist_smallgap_int005_2_100',
                                'mnist_smallgap_int01_2_100',
                                'mnist_smallgap_int015_2_100',
                                'mnist_smallgap_int02_2_100',
                                'mnist_smallgap_adv_2_100',
                                'mnist_smallgap_int001_3_100',
                                'mnist_smallgap_int005_3_100',
                                'mnist_smallgap_int01_3_100',
                                'mnist_smallgap_int015_3_100',
                                'mnist_smallgap_int02_3_100',
                                'mnist_smallgap_adv_3_100',
                                'mnist_smallgap_int001_4_100',
                                'mnist_smallgap_int005_4_100',
                                'mnist_smallgap_int01_4_100',
                                'mnist_smallgap_int015_4_100',
                                'mnist_smallgap_int02_4_100',
                                'mnist_smallgap_adv_4_100']
                    filters = [16, 32, 100]
                    kernels = [4, 4, 7]
                    strides = [2, 2, 1]
                    paddings = ['SAME', 'SAME', 'SAME']
                    epss = [0, 0.3]
                    for network in networks:
                        results = []
                        for eps in epss:
                            results.append(
                                str(pgd_attack(network, sess, filters, kernels, strides, paddings, eps, steps=steps)))
                        print(', '.join(results))

                    networks = ['cifar_smallgap_int001_6_200',
                                'cifar_smallgap_int005_6_200',
                                'cifar_smallgap_int01_6_200',
                                'cifar_smallgap_int015_6_200',
                                'cifar_smallgap_int02_6_200',
                                'cifar_smallgap_adv_6_200',
                                'cifar_smallgap_int001_8_200',
                                'cifar_smallgap_int005_8_200',
                                'cifar_smallgap_int01_8_200',
                                'cifar_smallgap_int015_8_200',
                                'cifar_smallgap_int02_8_200',
                                'cifar_smallgap_adv_8_200',
                                'cifar_smallgap_int001_10_200',
                                'cifar_smallgap_int005_10_200',
                                'cifar_smallgap_int01_10_200',
                                'cifar_smallgap_int015_10_200',
                                'cifar_smallgap_int02_10_200',
                                'cifar_smallgap_adv_10_200']
                    filters = [16, 32, 100]
                    kernels = [4, 4, 7]
                    strides = [2, 2, 1]
                    paddings = ['SAME', 'SAME', 'SAME']
                    epss = [0, 8 / 255]
                    for network in networks:
                        results = []
                        for eps in epss:
                            results.append(str(
                                pgd_attack(network, sess, filters, kernels, strides, paddings, eps, cifar=True,
                                           steps=steps)))
                        print(', '.join(results))
            except:
                print('Failed')

            print('Bound tightness experiments')
            try:
                networks = ['mnist_smallgap_normal_100']
                filters = [16, 32, 100]
                kernels = [4, 4, 7]
                strides = [2, 2, 1]
                paddings = ['SAME', 'SAME', 'SAME']
                for network in networks:
                    margin, disc = min_pgd_attack(network, sess, filters, kernels, strides, paddings, cifar=False,
                                                  one_class=False, all_class=False, l1=True, rep_l=False)
                    print(margin)
                    print(disc)

                networks = ['cifar_smallgap_normal_200']
                filters = [16, 32, 100]
                kernels = [4, 4, 7]
                strides = [2, 2, 1]
                paddings = ['SAME', 'SAME', 'SAME']
                for network in networks:
                    margin, disc = min_pgd_attack(network, sess, filters, kernels, strides, paddings, cifar=True,
                                                  one_class=False, all_class=False, l1=True, rep_l=False)
                    print(margin)
                    print(disc)
            except:
                print('Failed')
