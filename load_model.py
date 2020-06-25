"""
load_model.py
Model loading code
Copyright (C) 2020, Akhilan Boopathy <akhilan@mit.edu>
                    Sijia Liu <Sijia.Liu@ibm.com>
                    Gaoyuan Zhang <Gaoyuan.Zhang@ibm.com>
                    Cynthia Liu <cynliu98@mit.edu>
                    Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
                    Shiyu Chang <Shiyu.Chang@ibm.com>
                    Luca Daniel <luca@mit.edu>
"""

import pickle
import tensorflow as tf


class Model:
    pass


def load_model(network, sess, filters, kernels, strides, paddings, act=tf.nn.relu):
    with open('networks/' + network + '.pkl', 'rb') as file:
        param_vals = pickle.load(file)

    model = Model()

    def predict(inputs, act_fn=act):
        x = inputs
        layers = [x]
        # Define network
        for i, (l, k, s, p) in enumerate(zip(filters, kernels, strides, paddings)):
            if type(s) is str:  # Residual
                s = int(s[1:])
                W, b = param_vals[i]
                x = tf.nn.conv2d(act_fn(x), W, [1, s, s, 1], p) + b

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
                W, b = param_vals[i]
                if i == 0:
                    x = tf.nn.conv2d(x, W, [1, s, s, 1], p) + b
                else:
                    x = tf.nn.conv2d(act_fn(x), W, [1, s, s, 1], p) + b
                layers.append(x)
        pooled = tf.nn.avg_pool(x, [1, int(x.shape[1]), int(x.shape[2]), 1], [1, 1, 1, 1], 'VALID')

        W = param_vals[-1][0]
        logits = tf.nn.conv2d(pooled, W, [1, 1, 1, 1], 'SAME')
        logits = tf.layers.flatten(logits)

        # CAM
        cam = tf.nn.conv2d(x, W, [1, 1, 1, 1], 'SAME')

        return logits, x, cam

    def campp(out, rep):
        W = param_vals[-1][0]
        # GradCAM++
        campp = tf.nn.conv2d(rep, tf.nn.relu(W), [1, 1, 1, 1],
                             'SAME')
        return campp

    def ig(inputs, labels):
        grad_sum = tf.zeros_like(inputs)
        for k in range(100):
            z = inputs * (k + 1) / 100
            f = tf.reduce_sum(predict(z, tf.nn.softplus)[0] * labels, axis=1)
            grad_sum += tf.gradients(f, z)[0]
        return inputs * grad_sum / 100

    model.predict = predict
    model.ig = ig
    model.campp = campp
    return model
