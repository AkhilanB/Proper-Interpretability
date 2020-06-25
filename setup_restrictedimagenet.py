"""
setup_restrictedimagenet.py
Restricted imagenet data and model loading code
Copyright (C) 2020, Akhilan Boopathy <akhilan@mit.edu>
                    Sijia Liu <Sijia.Liu@ibm.com>
                    Gaoyuan Zhang <Gaoyuan.Zhang@ibm.com>
                    Cynthia Liu <cynliu98@mit.edu>
                    Pin-Yu Chen <Pin-Yu.Chen@ibm.com>
                    Shiyu Chang <Shiyu.Chang@ibm.com>
                    Luca Daniel <luca@mit.edu>
"""

import os
import numpy as np
import cv2
import matplotlib.pyplot as plt


class RestrictedImagenet:
    def __init__(self):
        path = "data/imagenet"  # Place restricted imagenet data in this folder
        train_images, train_labels, eval_images, eval_labels = (np.load(os.path.join(path, 'train_images.npy')),
                                                                np.load(os.path.join(path, 'train_labels.npy')),
                                                                np.load(os.path.join(path, 'eval_images.npy')),
                                                                np.load(os.path.join(path, 'eval_labels.npy')))

        self.train_data = train_images
        self.train_labels = np.eye(9)[train_labels]

        self.validation_data = eval_images
        self.validation_labels = np.eye(9)[eval_labels]

    @staticmethod
    def load_restricted_imagenet(rootpath, resize=True, dtype=np.uint8):

        # First load wnids
        wnids_file = os.path.join(rootpath, 'labels' + '.txt')
        with open(wnids_file, 'r') as f:
            wnids = [x.strip() for x in f]

        # Map wnids to integer labels
        wnid_to_label = {wnid: i for i, wnid in enumerate(wnids)}

        # Next load training data.
        X_train = []
        y_train = []
        for i, wnid in enumerate(wnids):
            print('loading training data for synset %s' % (wnid))
            filenames = os.listdir(os.path.join(rootpath, 'train', wnid))
            num_images = len(filenames)

            if resize:
                X_train_block = np.zeros((num_images, 224, 224, 3), dtype=dtype)

            y_train_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int32)
            for j, img_file in enumerate(filenames):
                img_file = os.path.join(rootpath, 'train', wnid, img_file)
                img = plt.imread(img_file)

                if resize:
                    img = cv2.resize(img, (224, 224))
                if img.ndim == 2:
                    ## grayscale file
                    if resize:
                        img = np.expand_dims(img, axis=-1)
                if img.shape[2] == 4:
                    img = img[:, :, :3]
                X_train_block[j] = img
            X_train.append(X_train_block)
            y_train.append(y_train_block)

        # We need to concatenate all training data
        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        # Next load training data.
        X_val = []
        y_val = []
        for i, wnid in enumerate(wnids):
            print('loading val data for synset %s' % (wnid))
            filenames = os.listdir(os.path.join(rootpath, 'val', wnid))
            num_images = len(filenames)

            if resize:
                X_val_block = np.zeros((num_images, 224, 224, 3), dtype=dtype)

            y_val_block = wnid_to_label[wnid] * np.ones(num_images, dtype=np.int32)
            for j, img_file in enumerate(filenames):
                img_file = os.path.join(rootpath, 'val', wnid, img_file)
                img = plt.imread(img_file)

                if resize:
                    img = cv2.resize(img, (224, 224))
                if img.ndim == 2:
                    ## grayscale file
                    if resize:
                        img.shape = (224, 224, 1)
                X_val_block[j] = img
            X_val.append(X_val_block)
            y_val.append(y_val_block)

        # We need to concatenate all training data
        X_val = np.concatenate(X_val, axis=0)
        y_val = np.concatenate(y_val, axis=0)
        return X_train, y_train, X_val, y_val


if __name__ == '__main__':
    ri = RestrictedImagenet()
    pass
