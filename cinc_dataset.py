import numpy as np
import os
from scipy.io import loadmat
import tensorflow as tf

import matplotlib.pyplot as plt


class CINCDataset:
    classes = ['N', 'O', 'A', '~']
    num_feats = 1
    num_labels = len(classes)
    max_len = 272

    def __init__(self, dset_root, is_train=True):
        self.dset_root = dset_root
        self.is_train = is_train

        if self.is_train:
            train_data_path = os.path.join(self.dset_root, 'training')
            gt_file = os.path.join(train_data_path, 'reference.csv')
            data_files_path = os.path.join(train_data_path, 'training2017')
            data_files = [fname for fname in os.listdir(
                data_files_path) if fname.endswith('.mat')]
            data_files_path = [os.path.join(data_files_path, x)
                               for x in data_files]
        else:
            valid_data_path = os.path.join(
                self.dset_root, 'validation', 'sample2017')

            data_files_path = os.path.join(valid_data_path, 'validation')
            gt_file = os.path.join(data_files_path, 'REFERENCE.csv')
            data_files = [fname for fname in os.listdir(
                data_files_path) if fname.endswith('.mat')]
            data_files_path = [os.path.join(
                data_files_path, x) for x in data_files]
        data_x = [loadmat(fname)['val'] for fname in data_files_path]
        # 2714 is the minimu length in train and valid sets
        data_x = [x[0, :2714] for x in data_x]

        # parse ground truth data
        reference_lines = [x.split(',')
                           for x in open(gt_file, 'r').readlines()]
        reference_lines = {x[0]: self.classes.index(
            x[1][:-1]) for x in reference_lines}
        # print(reference_lines)
        self.labels = [reference_lines[f[:-4]] for f in data_files]
        data_x = [x-np.mean(x) for x in data_x]
        self.data = np.expand_dims(np.array(data_x), axis=2).astype(np.float32)
        self.data = self.data/100        # for numerical stability of training
        self.data = self.data[:, ::10, :]  # Subsample
        print(np.unique(self.labels, return_counts=True))

    def to_dataset(self):
        """ returns a tensorflow dataset object """
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.data, self.labels))
        return dataset


if __name__ == '__main__':
    tf.enable_eager_execution()
    train_set = CINCDataset('dataset/cinc')
    valid_set = CINCDataset('dataset/cinc', False)

    train_set = train_set.to_dataset().shuffle(1000).batch(5)
    batch_x, batch_y = next(iter(train_set))
    fig, axes = plt.subplots(1, 5)
    for i in range(5):
        axes[i].plot(batch_x[i].numpy())
        axes[i].set_title('Label = {}'.format(batch_y[i].numpy()))
    #plt.show()
