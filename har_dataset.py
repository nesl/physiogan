import numpy as np
import os
import tensorflow as tf

from data_preprocs import subsample


class HARDataset:
    """ Class for loading the HARDAtaset """
    classes = ["WALKING",
               "WALKING_UPSTAIRS",
               "WALKING_DOWNSTAIRS",
               "SITTING", "STANDING", "LAYING"]

    num_labels = 6
    num_feats = 6
    max_len = 64

    def __init__(self, path, is_train=True, mini=None):
        self.path = path
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
        data = np.concatenate(data_list, axis=2)
        data = np.array([subsample(x, 2).astype(
                np.float32) for x in data])
        self.data = data
        self.labels = np.loadtxt(os.path.join(
            data_path, 'y_{}.txt'.format(name_suffix))).astype(np.int32) - 1
        print(np.unique(self.labels, return_counts=True))
        print(self.data.shape)
        # TODO(malzantot): add loading of subject

    def to_dataset(self):
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.data, self.labels))
        return dataset


if __name__ == '__main__':
    har_train = HARDataset('dataset/har/', True)
    har_test = HARDataset('dataset/har/', False)
