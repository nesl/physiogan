import numpy as np
import os

import tensorflow as tf


class SynDataset:

    def __init__(self, dset_root, is_train=None):
        """
        is_train: dummy placeholder fo consistency with other datasets
        """
        self.dset_root = dset_root
        self.data = np.load(os.path.join(self.dset_root, 'samples_x.npy'))
        self.labels = np.load(os.path.join(self.dset_root, 'samples_y.npy'))
        self.num_feats = self.data.shape[-1]
        self.num_labels = np.unique(self.labels).shape[0]

    def to_dataset(self):
        """ returns a tensorflow dataset object """
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.data, self.labels))
        return dataset


if __name__ == '__main__':
    tf.enable_eager_execution()
    syn_dataset = SynDataset('samples/har_crnn/05_27_15_03', True)
    print(syn_dataset.num_feats, ' ', syn_dataset.num_labels)
    syn_dataset = syn_dataset.to_dataset()
