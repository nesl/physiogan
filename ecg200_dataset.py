"""
Author: Moustafa Alzantot (malzantot@ucla.edu)
"""


import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_preprocs import subsample


class ECG200Dataset:
    classes = ['Normal', 'Abnormal']
    num_labels = len(classes)
    num_feats = 1
    max_len = 48

    def __init__(self, ds_root, is_train=True, mini=False):
        self.ds_root = ds_root
        self.is_train = is_train
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.idx2class = {i: c for c, i in self.class2idx.items()}
        data = []
        labels = []
        if self.is_train:
            file_name = 'ECG200_TRAIN.txt'
        else:
            file_name = 'ECG200_TEST.txt'

        file_path = os.path.join(self.ds_root, file_name)
        all_data = np.loadtxt(file_path)

        data = all_data[:, 1:, np.newaxis].astype(np.float32)

        data = [subsample(x, 2).astype(
                np.float32) for x in data]
        self.data = np.array(data)
        self.labels = (all_data[:, 0] == -1).astype(np.int32)
        print(np.unique(self.labels, return_counts=True))

    def to_dataset(self):
        """ returns a tensorflow dataset object """
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.data, self.labels))
        return dataset


if __name__ == '__main__':
    ecg200_dataset = ECG200Dataset(
        'dataset/ecg200', is_train=True, mini=False)
    print(ecg200_dataset.class2idx)
    print(ecg200_dataset.data.shape)
    print(ecg200_dataset.labels.shape)
    print(ecg200_dataset.to_dataset())
