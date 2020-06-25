"""
funcs.py
Contains utility functions
Copyright (C) 2020, Akhilan Boopathy <akhilan@mit.edu>
                    Sijia Liu <Sijia.Liu@ibm.com>
                    Gaoyuan Zhang <Gaoyuan.Zhang@ibm.com>
                    Cynthia Liu <cynliu98@mit.edu>
                    Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
                    Shiyu Chang <Shiyu.Chang@ibm.com>
                    Luca Daniel <luca@mit.edu>
"""
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
from sklearn.linear_model import LinearRegression
from scipy.stats import kendalltau
import cv2


def tau_corr(true, adv):
    corr_sum = 0
    for i in range(true.shape[2]):
        corr_sum += kendalltau(true[:, :, i], adv[:, :, i])[0]
    return corr_sum / int(true.shape[2])


from scipy.stats import pearsonr


def pearson_corr(true, adv):
    corr_sum = 0
    for i in range(true.shape[2]):
        corr_sum += pearsonr(true[:, :, i].flatten(), adv[:, :, i].flatten())[0]
    return corr_sum / int(true.shape[2])


def display_cifar(xs, name=None):
    for i in range(xs.shape[0]):
        x = xs[i]
        dim = 224
        x = cv2.resize(x, (dim, dim))
        plt.imshow(x, vmin=0.0, vmax=1.0)
        plt.axis('off')
        if name is None:
            plt.show()
        else:
            plt.savefig('gen_images/' + f"{i}_" + name + '.png', bbox_inches='tight')
            plt.close()


def save_cam_batch(xs, cams, names):
    cams_np = np.asarray(cams)
    m = np.mean(cams_np)
    v = np.std(cams_np)
    upper_limit = m + 2 * v
    lower_limit = m - 3 * v

    cams_np[cams_np > upper_limit] = upper_limit
    cams_np[cams_np < lower_limit] = lower_limit

    for i in range(cams_np.shape[0]):
        cam = cams_np[i][0]
        x = xs[0]
        name = names[i]
        image = np.uint8(x[:, :, ::-1] * 255.0)  # RGB -> BGR
        image = cv2.resize(image, (224, 224))

        cam = cv2.resize(cam, (224, 224))  # enlarge heatmap
        cam = np.maximum(cam, 0)
        heatmap = cam / np.max(cams_np)  # normalize

        cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)  # balck-and-white to color
        cam = np.float32(cam) + np.float32(image)  # everlay heatmap onto the image
        cam = 255 * cam / np.max(cam)
        cam = np.uint8(cam)

        cam_path = 'gen_images/' + f"{6}_" + name + '.png'

        # write images
        cv2.imwrite(cam_path, cam)
    a = 1

    pass


def save_cam(xs, cams, name=None, normalize_factor=None):
    """
    save Grad-CAM images
    """

    max_value = np.max(cams, axis=(1, 2))

    for i in range(xs.shape[0]):
        x = xs[i]
        cam = cams[i]

        image = np.uint8(x[:, :, ::-1] * 255.0)  # RGB -> BGR
        image = cv2.resize(image, (224, 224))
        cam = cv2.resize(cam, (224, 224))  # enlarge heatmap
        cam = np.maximum(cam, 0)
        heatmap = cam / normalize_factor[i]  # normalize

        cam = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)  # balck-and-white to color
        cam = np.float32(cam) + np.float32(image)  # everlay heatmap onto the image
        cam = 255 * cam / np.max(cam)
        cam = np.uint8(cam)

        cam_path = 'gen_images/' + f"{i}_" + name + '.png'

        # write images
        cv2.imwrite(cam_path, cam)


def save_ig(igss):
    a = np.array(igss)

    th = np.max(np.sum(np.abs(a), -1), axis=(0, 2, 3))

    for j, igs in enumerate(igss):

        for i in range(igs.shape[0]):
            ig = igs[i]
            saliency = np.sum(np.abs(ig), -1)
            saliency = cv2.resize(saliency, (224, 224))
            plt.imsave(f'gen_images/ig_{j}_{i}.png', saliency * 255, cmap="seismic", vmin=0, vmax=th[i] * 255)


def network_utils(network_type):
    if network_type == "small":
        filters = [16, 32, 100]
        kernels = [4, 4, 7]
        strides = [2, 2, 1]
        paddings = ['SAME', 'SAME', 'SAME']
    elif network_type == "wresnet":
        filters = [16] + 10 * [16] + 10 * [32] + 10 * [64]
        kernels = 31 * [3]
        strides = [1] + 5 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1'] + [2, 'r1'] + 4 * [1, 'r1']
        paddings = 31 * ['SAME']
    elif network_type == "pool":
        filters = [32, 'pool', 64, 'pool']
        kernels = [5, 2, 5, 2]
        strides = [1, 2, 1, 2]
        paddings = ['SAME', 'SAME', 'SAME', 'SAME']

    return filters, kernels, strides, paddings


def data_utils(dataset):
    if dataset == "cifar":
        data = CIFAR()
        inputs = tf.placeholder('float', shape=(None, 32, 32, 3))
        inputs_adv = tf.placeholder('float', shape=(None, 32, 32, 3))
        x_test = data.test_data + 0.5
        y_test = data.test_labels
        classes = 10
        targets = tf.placeholder('float', shape=(None, 10))
        labels = tf.placeholder('float', shape=(None, 10))
    elif dataset == "restimagenet":
        data = restImagenet()
        inputs = tf.placeholder('float', shape=(None, 224, 224, 3))
        inputs_adv = tf.placeholder('float', shape=(None, 224, 224, 3))
        x_test = data.validation_data / 255
        y_test = data.validation_labels
        classes = 9
        targets = tf.placeholder('float', shape=(None, 9))
        labels = tf.placeholder('float', shape=(None, 9))
    elif dataset == "mnist":
        data = MNIST()
        inputs = tf.placeholder('float', shape=(None, 28, 28, 1))
        inputs_adv = tf.placeholder('float', shape=(None, 28, 28, 1))
        x_test = data.test_data + 0.5
        y_test = data.test_labels
        classes = 10
        targets = tf.placeholder('float', shape=(None, 10))
        labels = tf.placeholder('float', shape=(None, 10))

    return data, inputs, inputs_adv, x_test, y_test, classes, targets, labels


def display_cam_cifar(xs, cams, name=None):
    for i in range(xs.shape[0]):
        x = xs[i]
        cam = cams[i]
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
            plt.savefig('gen_images/' + f"{i}_" + name + '.png', bbox_inches='tight')
            plt.close()


# Plot single point tradeoff, CAM (attack version)
def tradeoff_cam(file_name, sess, network_type, penalties, dataset, num_class, norm, inp_method, rep_l, kappa,
                 num_image, output_file):
    seed = 52
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)
    data, inputs, inputs_adv, x_test, y_test, classes, targets, labels = data_utils(dataset)

    filters, kernels, strides, paddings = network_utils(network_type)
    model = load_model(file_name, sess, filters, kernels, strides, paddings)

    out, rep, cam = model.predict(inputs)
    if inp_method == "cam++":
        cam = model.campp(out, rep)
    out_adv, rep_adv, cam_adv = model.predict(inputs_adv)
    if inp_method == "cam++":
        cam_adv = model.campp(out_adv, rep_adv)
    f_target = tf.reduce_sum(out_adv * targets, axis=1, keepdims=True)
    clean_f_target = tf.reduce_sum(out * targets, axis=1, keepdims=True)
    f_range = tf.stop_gradient(tf.reduce_max(out_adv) - tf.reduce_min(out_adv))
    clean_f_range = tf.reduce_max(out) - tf.reduce_min(out)
    all_margin = out_adv - f_range * targets - f_target
    margin = tf.maximum(tf.reduce_max(out_adv - f_range * targets - f_target, axis=1), -kappa)
    clean_margin = tf.reduce_max(out - clean_f_range * targets - clean_f_target, axis=1)

    if norm == "l1":
        cam_diff_all = tf.reduce_sum(tf.abs(tf.layers.flatten(cam_adv - cam)), axis=1) / classes
    elif norm == "l2":
        cam_diff_all = tf.reduce_sum(tf.layers.flatten(cam_adv - cam) ** 2, axis=1) / classes

    pred = tf.argmax(out, axis=1)
    cam_targ_adv = tf.reduce_sum(tf.reshape(targets, (-1, 1, 1, classes)) * cam_adv, axis=3)
    cam_targ = tf.reduce_sum(tf.reshape(targets, (-1, 1, 1, classes)) * cam, axis=3)
    cam_true_adv = tf.reduce_sum(tf.reshape(labels, (-1, 1, 1, classes)) * cam_adv, axis=3)
    cam_true = tf.reduce_sum(tf.reshape(labels, (-1, 1, 1, classes)) * cam, axis=3)

    if norm == "l1":
        cam_diff_one = tf.reduce_sum(tf.abs(tf.layers.flatten(cam_true_adv - cam_true)), axis=1)
        cam_diff_two = 0.5 * tf.reduce_sum(
            tf.abs(tf.layers.flatten(cam_targ_adv - cam_targ)) + tf.abs(tf.layers.flatten(cam_true_adv - cam_true)),
            axis=1)
    elif norm == "l2":
        cam_diff_one = tf.reduce_sum(tf.layers.flatten((cam_true_adv - cam_true) ** 2), axis=1)
        cam_diff_two = 0.5 * tf.reduce_sum(
            tf.layers.flatten((cam_targ_adv - cam_targ) ** 2) + tf.layers.flatten((cam_true_adv - cam_true) ** 2),
            axis=1)

    margin_loss = tf.reduce_mean(margin)
    gamma = tf.placeholder('float', shape=(None,))
    loss_all = margin_loss + tf.reduce_mean(gamma * cam_diff_all)
    loss_two = margin_loss + tf.reduce_mean(gamma * cam_diff_two)
    loss_one = margin_loss + tf.reduce_mean(gamma * cam_diff_one)

    grad_all = tf.gradients(loss_all, inputs_adv)[0]
    grad_two = tf.gradients(loss_two, inputs_adv)[0]
    grad_one = tf.gradients(loss_one, inputs_adv)[0]

    np.random.seed(seed)
    idx = np.random.permutation(x_test.shape[0])[37:42]
    start_time = timer.time()
    true = np.argmax(y_test[idx], axis=1)
    targ = (true + 1 + np.random.randint(classes - 1, size=num_image)) % classes
    targets_val = np.eye(classes)[targ]

    attacks_all = []
    attacks_two = []
    attacks_one = []

    # Adversarial attack
    print('Adversarial Attack')
    eps = 0.05 * np.ones(num_image)
    eps_min = np.zeros(num_image)
    eps_max = np.ones(num_image)
    for i in range(10):
        x_adv = x_test[idx].copy()
        x_nat = x_test[idx].copy()
        for j in range(10):
            feed_dict_attack = {inputs: x_nat, inputs_adv: x_adv, targets: targets_val, labels: y_test[idx, :],
                                gamma: np.zeros(num_image)}
            grad_val = sess.run(grad_all, feed_dict=feed_dict_attack)
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
        grad_val = sess.run(grad_all, feed_dict=feed_dict_attack)
        delta = 0.2 * np.clip(1000 * grad_val, -1, 1) * eps[:, np.newaxis, np.newaxis, np.newaxis]
        x_adv = x_adv - delta
        x_adv = np.clip(x_adv, x_nat - eps[:, np.newaxis, np.newaxis, np.newaxis],
                        x_nat + eps[:, np.newaxis, np.newaxis, np.newaxis])
        x_adv = np.clip(x_adv, 0, 1)
    eps_adv = eps
    attacks_all.append(x_adv)
    attacks_two.append(x_adv)
    attacks_one.append(x_adv)

    grads = [grad_all, grad_two, grad_one]
    attacks_list = [attacks_all, attacks_two, attacks_one]

    for grad, attacks in zip(grads, attacks_list):
        for penalty in penalties:
            # Int attack with penalty*eps_adv perturbation size
            print('Interprebility Attack, eps penalty ' + str(penalty))
            # Fix eps, search over gamma
            eps = penalty * eps_adv
            gamma_val = 0.00005 * np.ones(num_image)
            gamma_min = np.zeros(num_image)
            gamma_max = np.ones(num_image)
            for i in range(10):
                x_int = x_test[idx].copy()
                x_nat = x_test[idx].copy()
                for j in range(10):
                    feed_dict_attack = {inputs: x_nat, inputs_adv: x_int, targets: targets_val, labels: y_test[idx, :],
                                        gamma: gamma_val}
                    grad_val = sess.run(grad, feed_dict=feed_dict_attack)
                    delta = 0.2 * np.clip(1000 * grad_val, -1, 1) * eps[:, np.newaxis, np.newaxis, np.newaxis]
                    x_int = x_int - delta
                    x_int = np.clip(x_int, x_nat - eps[:, np.newaxis, np.newaxis, np.newaxis],
                                    x_nat + eps[:, np.newaxis, np.newaxis, np.newaxis])
                    x_int = np.clip(x_int, 0, 1)
                feed_dict_attack = {inputs: x_nat, inputs_adv: x_int, targets: targets_val, labels: y_test[idx, :],
                                    gamma: gamma_val}

                margin_val = sess.run(margin, feed_dict=feed_dict_attack)
                for k in range(num_image):
                    if margin_val[k] < 0:  # Success: Increase gamma
                        gamma_min[k] = gamma_val[k]
                        gamma_val[k] = min(2 * gamma_val[k], (gamma_min[k] + gamma_max[k]) / 2)
                    else:  # Failure: Decrease gamma
                        gamma_max[k] = gamma_val[k]
                        gamma_val[k] = (gamma_min[k] + gamma_max[k]) / 2
            gamma_val = gamma_min
            x_int = x_test[idx].copy()
            x_nat = x_test[idx].copy()
            for j in range(10):
                feed_dict_attack = {inputs: x_nat, inputs_adv: x_int, targets: targets_val, labels: y_test[idx, :],
                                    gamma: gamma_val}
                grad_val = sess.run(grad, feed_dict=feed_dict_attack)
                delta = 0.2 * np.clip(1000 * grad_val, -1, 1) * eps[:, np.newaxis, np.newaxis, np.newaxis]
                x_int = x_int - delta
                x_int = np.clip(x_int, x_nat - eps[:, np.newaxis, np.newaxis, np.newaxis],
                                x_nat + eps[:, np.newaxis, np.newaxis, np.newaxis])
                x_int = np.clip(x_int, 0, 1)
            attacks.append(x_int)

    # Process attacks: find discrepancy and eps
    results = []
    cam_diffs = [cam_diff_all, cam_diff_two, cam_diff_one]
    cam_names = ["all_class", "two_class", "one_class"]

    normalize_factor = 0
    baseline = 0
    baseline_targ = 0

    vals = []
    vals_targ = []

    def truncate(cam_val, factor):
        m = np.mean(cam_val)
        v = np.std(cam_val)
        upper_limit = m + factor * v
        lower_limit = m - factor * v

        cam_val[cam_val > upper_limit] = upper_limit
        cam_val[cam_val < lower_limit] = lower_limit

        return cam_val

    factor = 0.5
    cams = []
    names = []
    for attacks, cam_diff, cam_name in zip(attacks_list, cam_diffs, cam_names):
        epss = []
        ds = []
        for k, x_atk in enumerate(attacks):
            feed_dict_attack = {inputs: x_nat, inputs_adv: x_atk, targets: targets_val, labels: y_test[idx, :]}
            cam_diff_val, clean_out_val, atk_out_val, cm_true, cm_true_adv, cm_targ, cm_targ_adv = sess.run(
                [cam_diff, out, out_adv, cam_true, cam_true_adv, cam_targ, cam_targ_adv], feed_dict=feed_dict_attack)

            if k == 0 and cam_name == "all_class":
                baseline = cm_true[0][:, :, np.newaxis]
                baseline_targ = cm_targ[0][:, :, np.newaxis]

                vals_targ.append(tau_corr(baseline_targ, cm_targ_adv[0][:, :, np.newaxis]))
                vals.append(tau_corr(baseline, cm_true_adv[0][:, :, np.newaxis]))

                cams.append(cm_true)
                names.append(f"cam++_true")
                cams.append(cm_true_adv)
                names.append(f"cam++_true_adv_no_class")
                cams.append(cm_targ)
                names.append(f"cam++_targ")
                cams.append(cm_targ_adv)
                names.append(f"cam++_targ_adv")
            if k == 1:
                vals_targ.append(tau_corr(baseline_targ, cm_targ_adv[0][:, :, np.newaxis]))
                vals.append(tau_corr(baseline, cm_true_adv[0][:, :, np.newaxis]))
                cams.append(cm_true_adv)
                names.append(f"cam++_true_adv_{cam_name}")
                cams.append(cm_targ_adv)
                names.append(f"cam++_targ_adv_{cam_name}")
                pass
            epss.append(eps_adv * (1 if k == 0 else penalties[k - 1]))

            # discrepancy
            d = []
            for i in range(num_image):
                tr = true[i]
                trg = targ[i]
                if atk_out_val[i, trg] < atk_out_val[i, tr]:  # Attack failed
                    d.append(-1)
                elif clean_out_val[i, trg] > clean_out_val[i, tr]:  # Clean preferred target over true
                    d.append(-1)
                else:
                    d.append(cam_diff_val[i])
            d = np.asarray(d)
            ds.append(d)
        results.append((epss, ds))

    # Plot!!!
    save_cam_batch(x_nat, cams, names)
    plt.rc('font', size=18)
    plt.ticklabel_format(axis='y', style='sci', scilimits=(-2, 2))
    print(vals)
    print(vals_targ)
    i = 0

    linestyles = ['-', '--', ':']
    labels = ['all class', 'two class', 'one class']
    for (epss, ds), label, ls in zip(results, labels, linestyles):
        xs = []
        ys = []
        for eps, d in zip(epss, ds):
            if d[i] >= 0:
                xs.append(d[i])
                ys.append(eps[i])

        # If cam_d becomes larger with larger epsilon, can just use a previous attack
        if len(xs) >= 1:
            new_xs = []
            last = xs[0]
            for x in xs:
                last = min(x, last)
                new_xs.append(last)
            xs = new_xs

        plt.plot(ys, xs, label=label, ls=ls, marker='o')

    if inp_method == "cam++":
        if norm == "l1":
            plt.ylabel(r'$\ell_1$ GradCAM++ Discrepancy')
        elif norm == "l2":
            plt.ylabel(r'Squared $\ell_2$ GradCAM++ Discrepancy')
    else:
        if norm == "l1":
            plt.ylabel(r'$\ell_1$ CAM Discrepancy')
        elif norm == "l2":
            plt.ylabel(r'Squared $\ell_2$ CAM Discrepancy')

    plt.xlabel(r'$\epsilon$')
    plt.legend()

    if norm == "l1":
        name = "l1_"
    elif norm == "l2":
        name = "l2_"

    if inp_method == "cam++":
        plt.savefig('gen_images/tradeoff_oneptcampp_' + name + str(file_name[:-4]) + '.png', bbox_inches='tight')
    else:
        plt.savefig('gen_images/tradeoff_oneptcam_' + name + str(file_name[:-4]) + '.png', bbox_inches='tight')
    plt.close()


# Plot single point tradeoff, repr (attack version)
def tradeoff_repr(file_name, sess, network_type, penalties, dataset, num_class, norm, inp_method, rep_l, kappa,
                  num_image):
    np.random.seed(99)
    tf.set_random_seed(99)
    random.seed(99)

    data, inputs, inputs_adv, x_test, y_test, classes, targets, labels = data_utils(dataset)

    filters, kernels, strides, paddings = network_utils(network_type)
    model = load_model(file_name, sess, filters, kernels, strides, paddings)

    out, rep, cam = model.predict(inputs)
    out_adv, rep_adv, cam_adv = model.predict(inputs_adv)
    f_target = tf.reduce_sum(out_adv * targets, axis=1, keepdims=True)
    clean_f_target = tf.reduce_sum(out * targets, axis=1, keepdims=True)
    f_range = tf.stop_gradient(tf.reduce_max(out_adv) - tf.reduce_min(out_adv))
    clean_f_range = tf.reduce_max(out) - tf.reduce_min(out)
    all_margin = out_adv - f_range * targets - f_target
    margin = tf.maximum(tf.reduce_max(out_adv - f_range * targets - f_target, axis=1), -kappa)
    clean_margin = tf.reduce_max(out - clean_f_range * targets - clean_f_target, axis=1)

    if norm == "l1":
        rep_diff = tf.reduce_sum(tf.abs(tf.layers.flatten(rep_adv - rep)), axis=1)
    elif norm == "l2":
        rep_diff = tf.reduce_sum(tf.layers.flatten(rep_adv - rep) ** 2, axis=1)

    margin_loss = tf.reduce_mean(margin)
    gamma = tf.placeholder('float', shape=(None,))
    loss = margin_loss + tf.reduce_mean(gamma * rep_diff)

    grad = tf.gradients(loss, inputs_adv)[0]

    np.random.seed(99)
    idx = np.random.permutation(x_test.shape[0])[:num_image]

    start_time = timer.time()
    true = np.argmax(y_test[idx], axis=1)
    targ = (true + 1 + np.random.randint(classes - 1, size=num_image)) % classes
    targets_val = np.eye(classes)[targ]

    attacks = []

    # Adversarial attack
    print('Adversarial Attack')
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

    for penalty in penalties:
        # Int attack with penalty*eps_adv perturbation size
        print('Interprebility Attack, eps penalty ' + str(penalty))
        # Fix eps, search over gamma
        eps = penalty * eps_adv
        gamma_val = 0.00005 * np.ones(num_image)
        gamma_min = np.zeros(num_image)
        gamma_max = np.ones(num_image)
        for i in range(10):
            x_int = x_test[idx].copy()
            x_nat = x_test[idx].copy()
            for j in range(10):
                feed_dict_attack = {inputs: x_nat, inputs_adv: x_int, targets: targets_val, labels: y_test[idx, :],
                                    gamma: gamma_val}
                grad_val = sess.run(grad, feed_dict=feed_dict_attack)
                delta = 0.2 * np.clip(1000 * grad_val, -1, 1) * eps[:, np.newaxis, np.newaxis, np.newaxis]
                x_int = x_int - delta
                x_int = np.clip(x_int, x_nat - eps[:, np.newaxis, np.newaxis, np.newaxis],
                                x_nat + eps[:, np.newaxis, np.newaxis, np.newaxis])
                x_int = np.clip(x_int, 0, 1)
            feed_dict_attack = {inputs: x_nat, inputs_adv: x_int, targets: targets_val, labels: y_test[idx, :],
                                gamma: gamma_val}

            margin_val = sess.run(margin, feed_dict=feed_dict_attack)
            for k in range(num_image):
                if margin_val[k] < 0:  # Success: Increase gamma
                    gamma_min[k] = gamma_val[k]
                    gamma_val[k] = min(2 * gamma_val[k], (gamma_min[k] + gamma_max[k]) / 2)
                else:  # Failure: Decrease gamma
                    gamma_max[k] = gamma_val[k]
                    gamma_val[k] = (gamma_min[k] + gamma_max[k]) / 2
        gamma_val = gamma_min
        x_int = x_test[idx].copy()
        x_nat = x_test[idx].copy()
        for j in range(10):
            feed_dict_attack = {inputs: x_nat, inputs_adv: x_int, targets: targets_val, labels: y_test[idx, :],
                                gamma: gamma_val}
            grad_val = sess.run(grad, feed_dict=feed_dict_attack)
            delta = 0.2 * np.clip(1000 * grad_val, -1, 1) * eps[:, np.newaxis, np.newaxis, np.newaxis]
            x_int = x_int - delta
            x_int = np.clip(x_int, x_nat - eps[:, np.newaxis, np.newaxis, np.newaxis],
                            x_nat + eps[:, np.newaxis, np.newaxis, np.newaxis])
            x_int = np.clip(x_int, 0, 1)
        attacks.append(x_int)

    # Process attacks: find discrepancy and eps
    epss = []
    ds = []
    for x_atk in attacks:
        feed_dict_attack = {inputs: x_nat, inputs_adv: x_atk, targets: targets_val, labels: y_test[idx, :]}
        rep_diff_val, clean_out_val, atk_out_val = sess.run([rep_diff, out, out_adv], feed_dict=feed_dict_attack)

        # eps
        eps = np.max(np.abs(x_nat[:, :, :, :] - x_atk[:, :, :, :]), axis=(1, 2, 3))
        epss.append(eps)

        # discrepancy
        d = []
        for i in range(num_image):
            tr = true[i]
            trg = targ[i]
            if atk_out_val[i, trg] < atk_out_val[i, tr]:  # Attack failed
                d.append(-1)
            elif clean_out_val[i, trg] > clean_out_val[i, tr]:  # Clean preferred target over true
                d.append(-1)
            else:
                d.append(rep_diff_val[i])
        d = np.asarray(d)
        ds.append(d)

    # Plot!!!
    if dataset == "cifar":
        i = 3

    xs = []
    ys = []
    for eps, d in zip(epss, ds):
        if d[i] >= 0:
            xs.append(d[i])
            ys.append(eps[i])

    # If cam_d becomes larger with larger epsilon, can just use a previous attack
    if len(xs) >= 1:
        new_xs = []
        last = xs[0]
        for x in xs:
            last = min(x, last)
            new_xs.append(last)
        xs = new_xs

    plt.plot(ys, xs, marker='o')

    if norm == "l1":
        plt.ylabel(r'$\ell_1$ Repr Discrepancy')
    elif norm == "l2":
        plt.ylabel(r'$\ell_2$ Repr Discrepancy')
    plt.xlabel(r'$\epsilon$')

    if norm == "l1":
        name = "l1_"
    elif norm == "l2":
        name = "l2_"

    plt.savefig('gen_images/tradeoff_oneptrepr_' + name + str(file_name[:-4]) + '.png', bbox_inches='tight')
    plt.close()


# isa Statistics, IG (attack version)
def isa_ig(file_name, sess, network_type, penalties, dataset, num_class, norm, inp_method, rep_l, kappa, num_image,
            output_file):
    np.random.seed(99)
    tf.set_random_seed(99)
    random.seed(99)

    data, inputs, inputs_adv, x_test, y_test, classes, targets, labels = data_utils(dataset)

    filters, kernels, strides, paddings = network_utils(network_type)

    model = load_model(file_name, sess, filters, kernels, strides, paddings)

    out, rep, cam = model.predict(inputs)
    ig_true, ig_targ = model.ig(inputs, labels), model.ig(inputs, targets)
    out_adv, rep_adv, cam_adv = model.predict(inputs_adv)
    ig_true_adv, ig_targ_adv = model.ig(inputs_adv, labels), model.ig(inputs_adv, targets)
    f_target = tf.reduce_sum(out_adv * targets, axis=1, keepdims=True)
    clean_f_target = tf.reduce_sum(out * targets, axis=1, keepdims=True)
    f_range = tf.stop_gradient(tf.reduce_max(out_adv) - tf.reduce_min(out_adv))
    clean_f_range = tf.reduce_max(out) - tf.reduce_min(out)
    all_margin = out_adv - f_range * targets - f_target
    margin = tf.maximum(tf.reduce_max(out_adv - f_range * targets - f_target, axis=1), -kappa)
    clean_margin = tf.reduce_max(out - clean_f_range * targets - clean_f_target, axis=1)

    if num_class == "two_class":

        if norm == "l1":
            ig_diff = 0.5 * tf.reduce_sum(
                tf.abs(tf.layers.flatten(ig_targ_adv - ig_targ)) + tf.abs(tf.layers.flatten(ig_true_adv - ig_true)),
                axis=1)
        elif norm == "l2":
            ig_diff = 0.5 * tf.reduce_sum(
                tf.layers.flatten((ig_targ_adv - ig_targ) ** 2) + tf.layers.flatten((ig_true_adv - ig_true) ** 2),
                axis=1)
    elif num_class == "one_class":

        if norm == "l1":
            ig_diff = tf.reduce_sum(tf.abs(tf.layers.flatten(ig_true_adv - ig_true)), axis=1)
        elif norm == "l2":
            ig_diff = tf.reduce_sum(tf.layers.flatten((ig_true_adv - ig_true) ** 2), axis=1)

    elif num_class == "all_class":
        ig_true = []
        ig_true_adv = []
        for i in range(classes):
            label = np.eye(classes)[i]
            ig_true.append(model.ig(inputs, label))
            ig_true_adv.append(model.ig(inputs_adv, label))
        ig_diff = 0
        if norm == "l1":
            for ig, ig_adv in zip(ig_true, ig_true_adv):
                ig_diff += tf.reduce_sum(tf.abs(tf.layers.flatten(ig_adv - ig)), axis=1)

            ig_diff = ig_diff / classes

        if norm == "l2":
            for ig, ig_adv in zip(ig_true, ig_true_adv):
                ig_diff += tf.reduce_sum(tf.layers.flatten((ig_adv - ig) ** 2), axis=1)

            ig_diff = ig_diff / classes

    margin_loss = tf.reduce_mean(margin)
    gamma = tf.placeholder('float', shape=(None,))
    loss = margin_loss + tf.reduce_mean(gamma * ig_diff)

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

    for penalty in penalties:
        # Int attack with penalty*eps_adv perturbation size
        # Fix eps, search over gamma
        eps = penalty * eps_adv
        gamma_val = 0.00005 * np.ones(num_image)
        gamma_min = np.zeros(num_image)
        gamma_max = np.ones(num_image)
        for i in range(10):
            x_int = x_test[idx].copy()
            x_nat = x_test[idx].copy()
            for j in range(10):
                feed_dict_attack = {inputs: x_nat, inputs_adv: x_int, targets: targets_val, labels: y_test[idx, :],
                                    gamma: gamma_val}
                grad_val = sess.run(grad, feed_dict=feed_dict_attack)
                delta = 0.2 * np.clip(1000 * grad_val, -1, 1) * eps[:, np.newaxis, np.newaxis, np.newaxis]
                x_int = x_int - delta
                x_int = np.clip(x_int, x_nat - eps[:, np.newaxis, np.newaxis, np.newaxis],
                                x_nat + eps[:, np.newaxis, np.newaxis, np.newaxis])
                x_int = np.clip(x_int, 0, 1)
            feed_dict_attack = {inputs: x_nat, inputs_adv: x_int, targets: targets_val, labels: y_test[idx, :],
                                gamma: gamma_val}

            margin_val = sess.run(margin, feed_dict=feed_dict_attack)
            for k in range(num_image):
                if margin_val[k] < 0:  # Success: Increase gamma
                    gamma_min[k] = gamma_val[k]
                    gamma_val[k] = min(2 * gamma_val[k], (gamma_min[k] + gamma_max[k]) / 2)
                else:  # Failure: Decrease gamma
                    gamma_max[k] = gamma_val[k]
                    gamma_val[k] = (gamma_min[k] + gamma_max[k]) / 2
        gamma_val = gamma_min
        x_int = x_test[idx].copy()
        x_nat = x_test[idx].copy()
        for j in range(10):
            feed_dict_attack = {inputs: x_nat, inputs_adv: x_int, targets: targets_val, labels: y_test[idx, :],
                                gamma: gamma_val}
            grad_val = sess.run(grad, feed_dict=feed_dict_attack)
            delta = 0.2 * np.clip(1000 * grad_val, -1, 1) * eps[:, np.newaxis, np.newaxis, np.newaxis]
            x_int = x_int - delta
            x_int = np.clip(x_int, x_nat - eps[:, np.newaxis, np.newaxis, np.newaxis],
                            x_nat + eps[:, np.newaxis, np.newaxis, np.newaxis])
            x_int = np.clip(x_int, 0, 1)
        attacks.append(x_int)

    # Process attacks: find discrepancy and eps
    epss = []
    ds = []
    corrs = []
    p_corrs = []
    for x_atk in attacks:
        feed_dict_attack = {inputs: x_nat, inputs_adv: x_atk, targets: targets_val, labels: y_test[idx, :]}
        atk_out_val, clean_out_val, ig_diff_val, ig_true_val, ig_targ_val, ig_true_adv_val, ig_targ_adv_val = \
            sess.run([out_adv, out, ig_diff, ig_true, ig_targ, ig_true_adv, ig_targ_adv], feed_dict=feed_dict_attack)

        # eps
        eps = np.max(np.abs(x_nat[:, :, :, :] - x_atk[:, :, :, :]), axis=(1, 2, 3))
        epss.append(eps)

        # discrepancy
        d = []
        corr = []
        p_corr = []
        for i in range(num_image):
            tr = true[i]
            trg = targ[i]
            if atk_out_val[i, trg] < atk_out_val[i, tr]:  # Attack failed
                d.append(-1)
                corr.append(-2)
                p_corr.append(-2)
            elif clean_out_val[i, trg] > clean_out_val[i, tr]:  # Clean preferred target over true
                d.append(-1)
                corr.append(-2)
                p_corr.append(-2)
            else:
                if num_class == "two_class":
                    if norm == "l1":
                        d.append(0.5 * np.sum(np.abs(ig_true_val[i, :, :] - ig_true_adv_val[i, :, :])) / (
                                ig_true_val[i, :, :].max() - ig_true_val[i, :, :].min()) + \
                                 0.5 * np.sum(np.abs(ig_targ_val[i, :, :] - ig_targ_adv_val[i, :, :])) / (
                                         ig_targ_val[i, :, :].max() - ig_targ_val[i, :, :].min()))
                    elif norm == "l2":
                        d.append(0.5 * np.sqrt(np.sum((ig_true_val[i, :, :] - ig_true_adv_val[i, :, :]) ** 2)) / (
                                ig_true_val[i, :, :].max() - ig_true_val[i, :, :].min()) + \
                                 0.5 * np.sqrt(np.sum((ig_targ_val[i, :, :] - ig_targ_adv_val[i, :, :]) ** 2)) / (
                                         ig_targ_val[i, :, :].max() - ig_targ_val[i, :, :].min()))

                elif num_class == "one_class":

                    if norm == "l1":
                        d.append(np.sum(np.abs(ig_true_val[i, :, :] - ig_true_adv_val[i, :, :])) / (
                                ig_true_val[i, :, :].max() - ig_true_val[i, :, :].min()))
                    elif norm == "l2":
                        d.append(np.sqrt(np.sum((ig_true_val[i, :, :] - ig_true_adv_val[i, :, :]) ** 2)) / (
                                ig_true_val[i, :, :].max() - ig_true_val[i, :, :].min()))
                        
                elif num_class == "all_class":
                    
                    if norm == "l1":
                        total_d = 0

                        for ig, ig_adv in zip(ig_true_val, ig_true_adv_val):
                            total_d += np.sum(np.abs(ig[i, :, :] - ig_adv[i, :, :])) / (
                                    ig[i, :, :].max() - ig[i, :, :].min())
                        d.append(total_d / classes)
                    elif norm == "l2":

                        total_d = 0

                        for ig, ig_adv in zip(ig_true_val, ig_true_adv_val):
                            total_d += np.sqrt(np.sum((ig[i, :, :] - ig_adv[i, :, :]) ** 2)) / (
                                    ig[i, :, :].max() - ig[i, :, :].min())
                        d.append(total_d / classes)

                corr.append(
                    0.5 * tau_corr(ig_true_val[i, :, :][:, :, np.newaxis], ig_true_adv_val[i, :, :][:, :, np.newaxis])
                    + 0.5 * tau_corr(ig_targ_val[i, :, :][:, :, np.newaxis],
                                     ig_targ_adv_val[i, :, :][:, :, np.newaxis]))
                p_corr.append(0.5 * pearson_corr(ig_true_val[i, :, :][:, :, np.newaxis],
                                                 ig_true_adv_val[i, :, :][:, :, np.newaxis])
                              + 0.5 * pearson_corr(ig_targ_val[i, :, :][:, :, np.newaxis],
                                                   ig_targ_adv_val[i, :, :][:, :, np.newaxis]))

        ds.append(np.asarray(d))
        corrs.append(np.asarray(corr))
        p_corrs.append(np.asarray(p_corr))

    total_discrepancy = 0
    total_corr = 0
    total_pcorr = 0
    total_nslope = 0
    num_pts = 0
    for i in range(num_image):
        cs = []
        ps = []
        xs = []
        ys = []
        for eps, d, corr, p_corr in zip(epss, ds, corrs, p_corrs):
            if d[i] >= 0:
                xs.append(d[i])
                ys.append(eps[i])
                cs.append(corr[i])
                ps.append(p_corr[i])

        # If cam_d becomes larger with larger epsilon, can just use a previous attack
        if len(xs) >= 1:
            new_xs = []
            new_cs = []
            new_is = [0]

            for i in range(1, len(xs)):
                if xs[new_is[-1]] < xs[i]:
                    new_is.append(new_is[-1])
                else:
                    new_is.append(i)

            xs = [xs[i] for i in new_is]
            cs = [cs[i] for i in new_is]
            if xs[0] != xs[-1]:
                total_discrepancy += sum(xs)
                total_corr += sum(cs)
                total_pcorr += sum(ps)
                total_nslope += ((xs[0] - xs[-1]) * ys[0]) / ((ys[-1] - ys[0]) * xs[0])
                num_pts += len(xs)

    with tf.gfile.GFile(output_file, "w+") as writer:
        num_written_lines = 0
        tf.logging.info("***** isa results *****")

        output_line = f"total discrepancy: {total_discrepancy / num_pts}"
        writer.write(output_line)
        output_line = f"total nslope: {total_nslope / num_pts}"
        writer.write(output_line)

    return total_discrepancy / num_pts, total_nslope / num_pts


# isa Statistics, CAM (attack version)
def isa_cam(file_name, sess, network_type, penalties, dataset, num_class, norm, inp_method, rep_l, kappa, num_image,
             output_file):
    seed = 123
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)

    data, inputs, inputs_adv, x_test, y_test, classes, targets, labels = data_utils(dataset)

    filters, kernels, strides, paddings = network_utils(network_type)

    model = load_model(file_name, sess, filters, kernels, strides, paddings)

    out, rep, cam = model.predict(inputs)
    if inp_method == 'cam++':
        cam = model.campp(out, rep)
    out_adv, rep_adv, cam_adv = model.predict(inputs_adv)
    if inp_method == 'cam++':
        cam_adv = model.campp(out_adv, rep_adv)
    f_target = tf.reduce_sum(out_adv * targets, axis=1, keepdims=True)
    clean_f_target = tf.reduce_sum(out * targets, axis=1, keepdims=True)
    f_range = tf.stop_gradient(tf.reduce_max(out_adv) - tf.reduce_min(out_adv))
    clean_f_range = tf.reduce_max(out) - tf.reduce_min(out)
    all_margin = out_adv - f_range * targets - f_target
    margin = tf.maximum(tf.reduce_max(out_adv - f_range * targets - f_target, axis=1), -kappa)
    clean_margin = tf.reduce_max(out - clean_f_range * targets - clean_f_target, axis=1)

    if norm == 'l1':
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

    if num_class != "all_class":
        if num_class == "one_class":
            if norm == "l1":
                cam_diff = tf.reduce_sum(tf.abs(tf.layers.flatten(cam_true_adv - cam_true)), axis=1)
            else:
                cam_diff = tf.reduce_sum(tf.layers.flatten((cam_true_adv - cam_true) ** 2), axis=1)
        else:
            if norm == "l1":
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

    np.random.seed(seed)
    idx = np.random.permutation(x_test.shape[0])[:num_image]

    start_time = timer.time()
    true = np.argmax(y_test[idx], axis=1)
    targ = (true + 1 + np.random.randint(classes - 1, size=num_image)) % classes
    targets_val = np.eye(classes)[targ]

    attacks = []

    # Adversarial attack
    # print('Adversarial Attack')
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

    for penalty in penalties:
        # Int attack with penalty*eps_adv perturbation size
        # Fix eps, search over gamma
        eps = penalty * eps_adv
        gamma_val = 0.00005 * np.ones(num_image)
        gamma_min = np.zeros(num_image)
        gamma_max = np.ones(num_image)
        for i in range(10):
            x_int = x_test[idx].copy()
            x_nat = x_test[idx].copy()
            for j in range(10):
                feed_dict_attack = {inputs: x_nat, inputs_adv: x_int, targets: targets_val, labels: y_test[idx, :],
                                    gamma: gamma_val}
                grad_val = sess.run(grad, feed_dict=feed_dict_attack)
                delta = 0.2 * np.clip(1000 * grad_val, -1, 1) * eps[:, np.newaxis, np.newaxis, np.newaxis]
                x_int = x_int - delta
                x_int = np.clip(x_int, x_nat - eps[:, np.newaxis, np.newaxis, np.newaxis],
                                x_nat + eps[:, np.newaxis, np.newaxis, np.newaxis])
                x_int = np.clip(x_int, 0, 1)
            feed_dict_attack = {inputs: x_nat, inputs_adv: x_int, targets: targets_val, labels: y_test[idx, :],
                                gamma: gamma_val}

            margin_val = sess.run(margin, feed_dict=feed_dict_attack)
            for k in range(num_image):
                if margin_val[k] < 0:  # Success: Increase gamma
                    gamma_min[k] = gamma_val[k]
                    gamma_val[k] = min(2 * gamma_val[k], (gamma_min[k] + gamma_max[k]) / 2)
                else:  # Failure: Decrease gamma
                    gamma_max[k] = gamma_val[k]
                    gamma_val[k] = (gamma_min[k] + gamma_max[k]) / 2
        gamma_val = gamma_min
        x_int = x_test[idx].copy()
        x_nat = x_test[idx].copy()
        for j in range(10):
            feed_dict_attack = {inputs: x_nat, inputs_adv: x_int, targets: targets_val, labels: y_test[idx, :],
                                gamma: gamma_val}
            grad_val = sess.run(grad, feed_dict=feed_dict_attack)
            delta = 0.2 * np.clip(1000 * grad_val, -1, 1) * eps[:, np.newaxis, np.newaxis, np.newaxis]
            x_int = x_int - delta
            x_int = np.clip(x_int, x_nat - eps[:, np.newaxis, np.newaxis, np.newaxis],
                            x_nat + eps[:, np.newaxis, np.newaxis, np.newaxis])
            x_int = np.clip(x_int, 0, 1)
        attacks.append(x_int)

    # Process attacks: find discrepancy and eps
    epss = []
    ds = []
    corrs = []
    p_corrs = []
    for x_atk in attacks:
        feed_dict_attack = {inputs: x_nat, inputs_adv: x_atk, targets: targets_val, labels: y_test[idx, :]}
        cam_diff_val, rep_diff_val, clean_out_val, atk_out_val, cam_val, cam_adv_val, rep_val, rep_adv_val = \
            sess.run([cam_diff, rep_diff, out, out_adv, cam, cam_adv, rep, rep_adv], feed_dict=feed_dict_attack)

        # eps
        eps = np.max(np.abs(x_nat[:, :, :, :] - x_atk[:, :, :, :]), axis=(1, 2, 3))
        epss.append(eps)

        # discrepancy
        d = []
        corr = []
        p_corr = []
        for i in range(num_image):
            tr = true[i]
            trg = targ[i]
            if atk_out_val[i, trg] < atk_out_val[i, tr]:  # Attack failed
                d.append(-1)
                corr.append(-2)
                p_corr.append(-2)
            elif clean_out_val[i, trg] > clean_out_val[i, tr]:  # Clean preferred target over true
                d.append(-1)
                corr.append(-2)
                p_corr.append(-2)
            else:
                if not rep_l:
                    if num_class == "one_class":
                        if norm == "l1":
                            d.append(np.sum(np.abs(cam_val[i, :, :, tr] - cam_adv_val[i, :, :, tr])) / (
                                    cam_val[i, :, :, tr].max() - cam_val[i, :, :, tr].min()))
                        else:
                            d.append(np.sqrt(np.sum((cam_val[i, :, :, tr] - cam_adv_val[i, :, :, tr]) ** 2)) / (
                                    cam_val[i, :, :, tr].max() - cam_val[i, :, :, tr].min()))
                        corr.append(tau_corr(cam_val[i, :, :, tr][:, :, np.newaxis],
                                             cam_adv_val[i, :, :, tr][:, :, np.newaxis]))
                        p_corr.append(pearson_corr(cam_val[i, :, :, tr][:, :, np.newaxis],
                                                   cam_adv_val[i, :, :, tr][:, :, np.newaxis]))
                    elif num_class == "two_class":
                        if norm == "l1":
                            d.append(0.5 * np.sum(np.abs(cam_val[i, :, :, tr] - cam_adv_val[i, :, :, tr])) / (
                                    cam_val[i, :, :, tr].max() - cam_val[i, :, :, tr].min()) + \
                                     0.5 * np.sum(np.abs(cam_val[i, :, :, trg] - cam_adv_val[i, :, :, trg])) / (
                                             cam_val[i, :, :, trg].max() - cam_val[i, :, :, trg].min()))
                        else:
                            d.append(0.5 * np.sqrt(np.sum((cam_val[i, :, :, tr] - cam_adv_val[i, :, :, tr]) ** 2)) / (
                                    cam_val[i, :, :, tr].max() - cam_val[i, :, :, tr].min()) + \
                                     0.5 * np.sqrt(np.sum((cam_val[i, :, :, trg] - cam_adv_val[i, :, :, trg]) ** 2)) / (
                                             cam_val[i, :, :, trg].max() - cam_val[i, :, :, trg].min()))
                        corr.append(0.5 * tau_corr(cam_val[i, :, :, tr][:, :, np.newaxis],
                                                   cam_adv_val[i, :, :, tr][:, :, np.newaxis])
                                    + 0.5 * tau_corr(cam_val[i, :, :, trg][:, :, np.newaxis],
                                                     cam_adv_val[i, :, :, trg][:, :, np.newaxis]))
                        p_corr.append(0.5 * pearson_corr(cam_val[i, :, :, tr][:, :, np.newaxis],
                                                         cam_adv_val[i, :, :, tr][:, :, np.newaxis])
                                      + 0.5 * pearson_corr(cam_val[i, :, :, trg][:, :, np.newaxis],
                                                           cam_adv_val[i, :, :, trg][:, :, np.newaxis]))
                    else:
                        if norm == "l1":
                            d.append(np.sum(np.sum(np.abs(
                                np.reshape(cam_val[i, :, :, :], (-1, classes)) - np.reshape(cam_adv_val[i, :, :, :],
                                                                                            (-1, classes))), axis=0) / \
                                            (np.amax(np.reshape(cam_val[i, :, :, :], (-1, classes)), axis=0) - np.amin(
                                                np.reshape(cam_val[i, :, :, :], (-1, classes)), axis=0)),
                                            axis=0) / classes)
                        else:
                            d.append(np.sum(np.sqrt(np.sum((np.reshape(cam_val[i, :, :, :], (-1, classes)) - np.reshape(
                                cam_adv_val[i, :, :, :], (-1, classes))) ** 2, axis=0)) / \
                                            (np.amax(np.reshape(cam_val[i, :, :, :], (-1, classes)), axis=0) - np.amin(
                                                np.reshape(cam_val[i, :, :, :], (-1, classes)), axis=0)),
                                            axis=0) / classes)
                        corr.append(tau_corr(cam_val[i, :, :, :], cam_adv_val[i, :, :, :]))
                        p_corr.append(pearson_corr(cam_val[i, :, :, :], cam_adv_val[i, :, :, :]))
                else:
                    channels = rep_val.shape[-1]
                    if norm == "l1":
                        d.append(np.sum(np.sum(np.abs(
                            np.reshape(rep_val[i, :, :, :], (-1, channels)) - np.reshape(rep_adv_val[i, :, :, :],
                                                                                         (-1, channels))), axis=0) / \
                                        (np.amax(np.reshape(rep_val[i, :, :, :], (-1, channels)), axis=0) - np.amin(
                                            np.reshape(rep_val[i, :, :, :], (-1, channels)), axis=0)),
                                        axis=0) / channels)
                    else:
                        d.append(np.sum(np.sqrt(np.sum((np.reshape(rep_val[i, :, :, :], (-1, channels)) - np.reshape(
                            rep_adv_val[i, :, :, :], (-1, channels))) ** 2, axis=0)) / \
                                        (np.amax(np.reshape(rep_val[i, :, :, :], (-1, channels)), axis=0) - np.amin(
                                            np.reshape(rep_val[i, :, :, :], (-1, channels)), axis=0)),
                                        axis=0) / channels)
                    corr.append(tau_corr(rep_val[i, :, :, :], rep_adv_val[i, :, :, :]))
                    p_corr.append(pearson_corr(rep_val[i, :, :, :], rep_adv_val[i, :, :, :]))
        ds.append(np.asarray(d))
        corrs.append(np.asarray(corr))
        p_corrs.append(np.asarray(p_corr))

    total_discrepancy = 0
    total_corr = 0
    total_pcorr = 0
    total_nslope = np.zeros(3)
    num_pts = 0
    for i in range(num_image):
        cs = []
        ps = []
        xs = []
        ys = []
        for eps, d, corr, p_corr in zip(epss, ds, corrs, p_corrs):
            if d[i] >= 0:
                xs.append(d[i])
                ys.append(eps[i])
                cs.append(corr[i])
                ps.append(p_corr[i])

        # If cam_d becomes larger with larger epsilon, can just use a previous attack
        if len(xs) >= 1:
            new_xs = []
            new_cs = []
            new_is = [0]

            for i in range(1, len(xs)):
                if xs[new_is[-1]] < xs[i]:
                    new_is.append(new_is[-1])
                else:
                    new_is.append(i)

            xs = [xs[i] for i in new_is]
            cs = [cs[i] for i in new_is]
            if xs[0] != xs[-1]:
                total_discrepancy += sum(xs)
                total_nslope[0] += ((xs[0] - xs[-1]) * ys[0]) / ((ys[-1] - ys[0]) * xs[0])
                total_corr += sum(cs)
                total_pcorr += sum(ps)
                num_pts += len(xs)

            for index in range(1, len(xs)):
                total_nslope[1] += (xs[index - 1] - xs[index]) / ((ys[index] - ys[index - 1]))

            total_nslope[1] /= (len(xs) - 1)
            total_nslope[1] *= ys[0] * xs[0]

            reg = LinearRegression().fit(np.log(np.array(xs))[:, np.newaxis], np.array(ys))
            total_nslope[2] = reg.coef_[0]

    with tf.gfile.GFile(output_file, "w+") as writer:
        num_written_lines = 0
        tf.logging.info("***** isa results *****")

        output_line = f"total discrepancy: {total_discrepancy / num_pts}"
        writer.write(output_line)
        output_line = f"total nslope: {total_nslope / num_pts}"
        writer.write(output_line)

    return total_discrepancy / num_pts, total_nslope / num_pts


def attack_and_display(file_name, sess, network_type, penalties, dataset, num_class, norm, inp_method, rep_l, kappa,
                       num_image, output_file):
    seed = 99
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)

    data, inputs, inputs_adv, x_test, y_test, classes, targets, labels = data_utils(dataset)

    filters, kernels, strides, paddings = network_utils(network_type)

    model = load_model(file_name, sess, filters, kernels, strides, paddings)

    out, rep, cam = model.predict(inputs)
    if inp_method == 'cam++':
        cam = model.campp(out, rep)
    out_adv, rep_adv, cam_adv = model.predict(inputs_adv)
    if inp_method == 'cam++':
        cam_adv = model.campp(out_adv, rep_adv)
    f_target = tf.reduce_sum(out_adv * targets, axis=1, keepdims=True)
    clean_f_target = tf.reduce_sum(out * targets, axis=1, keepdims=True)
    f_range = tf.stop_gradient(tf.reduce_max(out_adv) - tf.reduce_min(out_adv))
    clean_f_range = tf.reduce_max(out) - tf.reduce_min(out)
    all_margin = out_adv - f_range * targets - f_target
    margin = tf.maximum(tf.reduce_max(out_adv - f_range * targets - f_target, axis=1), -kappa)
    clean_margin = tf.reduce_max(out - clean_f_range * targets - clean_f_target, axis=1)

    if norm == 'l1':
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

    if num_class != "all_class":
        if num_class == "one_class":
            if norm == "l1":
                cam_diff = tf.reduce_sum(tf.abs(tf.layers.flatten(cam_true_adv - cam_true)), axis=1)
            else:
                cam_diff = tf.reduce_sum(tf.layers.flatten((cam_true_adv - cam_true) ** 2), axis=1)
        else:
            if norm == "l1":
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

    np.random.seed(seed)
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

    for penalty in penalties:
        # Int attack with penalty*eps_adv perturbation size
        # Fix eps, search over gamma
        eps = penalty * eps_adv
        gamma_val = 0.00005 * np.ones(num_image)
        gamma_min = np.zeros(num_image)
        gamma_max = np.ones(num_image)
        for i in range(10):
            x_int = x_test[idx].copy()
            x_nat = x_test[idx].copy()
            for j in range(10):
                feed_dict_attack = {inputs: x_nat, inputs_adv: x_int, targets: targets_val, labels: y_test[idx, :],
                                    gamma: gamma_val}
                grad_val = sess.run(grad, feed_dict=feed_dict_attack)
                delta = 0.2 * np.clip(1000 * grad_val, -1, 1) * eps[:, np.newaxis, np.newaxis, np.newaxis]
                x_int = x_int - delta
                x_int = np.clip(x_int, x_nat - eps[:, np.newaxis, np.newaxis, np.newaxis],
                                x_nat + eps[:, np.newaxis, np.newaxis, np.newaxis])
                x_int = np.clip(x_int, 0, 1)
            feed_dict_attack = {inputs: x_nat, inputs_adv: x_int, targets: targets_val, labels: y_test[idx, :],
                                gamma: gamma_val}

            margin_val = sess.run(margin, feed_dict=feed_dict_attack)
            for k in range(num_image):
                if margin_val[k] < 0:  # Success: Increase gamma
                    gamma_min[k] = gamma_val[k]
                    gamma_val[k] = min(2 * gamma_val[k], (gamma_min[k] + gamma_max[k]) / 2)
                else:  # Failure: Decrease gamma
                    gamma_max[k] = gamma_val[k]
                    gamma_val[k] = (gamma_min[k] + gamma_max[k]) / 2
        gamma_val = gamma_min
        x_int = x_test[idx].copy()
        x_nat = x_test[idx].copy()
        for j in range(10):
            feed_dict_attack = {inputs: x_nat, inputs_adv: x_int, targets: targets_val, labels: y_test[idx, :],
                                gamma: gamma_val}
            grad_val = sess.run(grad, feed_dict=feed_dict_attack)
            delta = 0.2 * np.clip(1000 * grad_val, -1, 1) * eps[:, np.newaxis, np.newaxis, np.newaxis]
            x_int = x_int - delta
            x_int = np.clip(x_int, x_nat - eps[:, np.newaxis, np.newaxis, np.newaxis],
                            x_nat + eps[:, np.newaxis, np.newaxis, np.newaxis])
            x_int = np.clip(x_int, 0, 1)
        attacks.append(x_int)

    # Process attacks: find discrepancy and eps
    epss = []
    ds = []
    corrs = []
    p_corrs = []

    cams = []
    for i, x_atk in enumerate(attacks):
        feed_dict_attack = {inputs: x_nat, inputs_adv: x_atk, targets: targets_val, labels: y_test[idx, :]}
        cm_true, cm_true_adv, cm_targ, cm_targ_adv, o, o_adv = \
            sess.run([cam_true, cam_true_adv, cam_targ, cam_targ_adv, out, out_adv], feed_dict=feed_dict_attack)

        cams.append(cm_true)
        cams.append(cm_true_adv)
        cams.append(cm_targ)
        cams.append(cm_targ_adv)

        break

    save_cam(cams, x_nat, "aaa")


# isa Statistics, IG (attack version)
def attack_and_display_ig(file_name, sess, network_type, penalties, dataset, num_class, norm, inp_method, rep_l, kappa,
                          num_image, output_file):
    seed = 52
    np.random.seed(seed)
    tf.set_random_seed(seed)
    random.seed(seed)

    data, inputs, inputs_adv, x_test, y_test, classes, targets, labels = data_utils(dataset)

    filters, kernels, strides, paddings = network_utils(network_type)

    model = load_model(file_name, sess, filters, kernels, strides, paddings)

    out, rep, cam = model.predict(inputs)
    ig_true, ig_targ = model.ig(inputs, labels), model.ig(inputs, targets)
    out_adv, rep_adv, cam_adv = model.predict(inputs_adv)
    ig_true_adv, ig_targ_adv = model.ig(inputs_adv, labels), model.ig(inputs_adv, targets)
    f_target = tf.reduce_sum(out_adv * targets, axis=1, keepdims=True)
    clean_f_target = tf.reduce_sum(out * targets, axis=1, keepdims=True)
    f_range = tf.stop_gradient(tf.reduce_max(out_adv) - tf.reduce_min(out_adv))
    clean_f_range = tf.reduce_max(out) - tf.reduce_min(out)
    all_margin = out_adv - f_range * targets - f_target
    margin = tf.maximum(tf.reduce_max(out_adv - f_range * targets - f_target, axis=1), -kappa)
    clean_margin = tf.reduce_max(out - clean_f_range * targets - clean_f_target, axis=1)

    if norm == "l1":
        ig_diff = 0.5 * tf.reduce_sum(
            tf.abs(tf.layers.flatten(ig_targ_adv - ig_targ)) + tf.abs(tf.layers.flatten(ig_true_adv - ig_true)), axis=1)
    else:
        ig_diff = 0.5 * tf.reduce_sum(
            tf.layers.flatten((ig_targ_adv - ig_targ) ** 2) + tf.layers.flatten((ig_true_adv - ig_true) ** 2), axis=1)

    margin_loss = tf.reduce_mean(margin)
    gamma = tf.placeholder('float', shape=(None,))
    loss = margin_loss + tf.reduce_mean(gamma * ig_diff)

    grad = tf.gradients(loss, inputs_adv)[0]

    np.random.seed(seed)
    idx = np.random.permutation(x_test.shape[0])[87:97]

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
    eps = eps_max * 2
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

    # Process attacks: find discrepancy and eps
    epss = []
    ds = []
    corrs = []
    p_corrs = []
    for x_atk in attacks:
        feed_dict_attack = {inputs: x_nat, inputs_adv: x_atk, targets: targets_val, labels: y_test[idx, :]}
        atk_out_val, clean_out_val, ig_diff_val, ig_true_val, ig_targ_val, ig_true_adv_val, ig_targ_adv_val = \
            sess.run([out_adv, out, ig_diff, ig_true, ig_targ, ig_true_adv, ig_targ_adv], feed_dict=feed_dict_attack)

        saliency = np.sum(np.abs(ig_true_val[idx]), -1)
        original_saliency1 = 224 * 224 * saliency / np.sum(saliency)
        original_saliency1 = cv2.resize(original_saliency1, (224, 224))

        saliency = np.sum(np.abs(ig_true_adv_val[idx]), -1)
        original_saliency2 = 224 * 224 * saliency / np.sum(saliency)
        original_saliency2 = cv2.resize(original_saliency2, (224, 224))

        print(tau_corr(original_saliency1[:, :, np.newaxis], original_saliency2[:, :, np.newaxis]))

        saliency = np.sum(np.abs(ig_targ_val[idx]), -1)
        original_saliency3 = 224 * 224 * saliency / np.sum(saliency)
        original_saliency3 = cv2.resize(original_saliency3, (224, 224))

        saliency = np.sum(np.abs(ig_targ_adv_val[idx]), -1)
        original_saliency4 = 224 * 224 * saliency / np.sum(saliency)
        original_saliency4 = cv2.resize(original_saliency4, (224, 224))

        print(tau_corr(original_saliency3[:, :, np.newaxis], original_saliency4[:, :, np.newaxis]))
        # plt.imshow(original_saliency, cmap="hot")
        # plt.show()
        max_value = max(max(np.max(original_saliency1), np.max(original_saliency2)),
                        max(np.max(original_saliency3), np.max(original_saliency4)))
        min_value = min(min(np.min(original_saliency1), np.min(original_saliency2)),
                        min(np.min(original_saliency3), np.min(original_saliency4)))

        def filt(a, min_value, max_value, th):
            m = np.mean(a)
            v = np.std(a)
            upper_limit = m + 2 * v
            lower_limit = m - 2 * v

            a[a > upper_limit] = upper_limit
            a[a < lower_limit] = lower_limit
            return a

        th = 0
        plt.imsave('ig_true_val.jpg', filt(original_saliency1, min_value, max_value, th), cmap="gray", vmin=min_value,
                   vmax=max_value)
        plt.imsave('ig_true_adv_val.jpg', filt(original_saliency2, min_value, max_value, th), cmap="gray",
                   vmin=min_value, vmax=max_value)
        plt.imsave('ig_targ_val.jpg', filt(original_saliency3, min_value, max_value, th), cmap="gray", vmin=min_value,
                   vmax=max_value)
        plt.imsave('ig_targ_adv_val.jpg', filt(original_saliency4, min_value, max_value, th), cmap="gray",
                   vmin=min_value, vmax=max_value)
        break
