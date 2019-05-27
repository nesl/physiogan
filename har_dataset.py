import numpy as np
import os
import tensorflow as tf


class HARDataset:
    """ Class for loading the HARDAtaset """
    classes = ["WALKING",
               "WALKING_UPSTAIRS",
               "WALKING_DOWNSTAIRS",
               "SITTING", "STANDING", "LAYING"]

    num_labels = 6
    num_feats = 6

    def __init__(self, path, is_train=True, is_syn=False):
        self.path = path
        self.is_syn = is_syn
        if is_syn:
            self.data = np.load(os.path.join(path, 'samples_x.npy'))
            self.labels = np.load(os.path.join(path, 'samples_y.npy'))
        else:
            self.is_train = is_train
            name_suffix = 'train' if is_train else 'test'
            data_path = os.path.join(
                self.path, 'UCI HAR Dataset', name_suffix)
            files_list = ['body_acc_x',
                          'body_acc_y',
                          'body_acc_z',
                          'body_gyro_x',
                          'body_gyro_y',
                          'body_gyro_z'
                          ]

            data_list = [np.expand_dims(np.loadtxt(
                os.path.join(
                    data_path, 'Inertial Signals',
                    '{}_{}.txt'.format(name, name_suffix))).astype(np.float32), axis=2) for name in files_list]
            self.data = np.concatenate(data_list, axis=2)
            self.labels = np.loadtxt(os.path.join(
                data_path, 'y_{}.txt'.format(name_suffix))).astype(np.int32) - 1

        # TODO(malzantot): add loading of subject

    def to_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.data, self.labels))
        return dataset
