"""
Author: Moustafa Alzantot (malzantot@ucla.edu)
"""


import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_preprocs import subsample


class ADLDataset:
    classes = ['Brush_teeth', 'Climb_stairs', 'Comb_hair',
               'Descend_stairs', 'Getup_bed', 'Pour_water', 'Sitdown_chair', 'Standup_chair', 'Walk']
    num_labels = len(classes)
    num_feats = 3
    max_len = 125

    subsample_every = 4

    def __init__(self, ds_root, is_train=True, mini=False):
        if mini:
            self.classes = ['Climb_stairs', 'Comb_hair',
                            'Descend_stairs', 'Sitdown_chair']
            self.num_labels = len(self.classes)
        else:
            self.subsample_every = 2
        self.max_len = self.max_len // self.subsample_every
        self.ds_root = ds_root
        self.is_train = is_train
        self.class2idx = {c: i for i, c in enumerate(self.classes)}
        self.idx2class = {i: c for c, i in self.class2idx.items()}
        data = []
        labels = []
        for c_name, c_id in self.class2idx.items():
            c_path = os.path.join(self.ds_root, c_name)
            c_files = [os.path.join(c_path, fname) for fname in os.listdir(
                c_path) if fname.endswith('.txt')]
            c_data = [np.loadtxt(c_file).astype(np.float32)
                      for c_file in c_files]
            data.extend(c_data)
            labels.extend([c_id]*len(c_data))

        # min_len = min([x.shape[0] for x in data])
        # clip
        if mini:
            #self.max_len = 125
            # TODO(malzantot): try smoothing istead of subsampling.
            data = [subsample(x, self.subsample_every).astype(
                np.float32) for x in data]
        else:
            data = [subsample(x, 2).astype(
                np.float32) for x in data]
        data = [x[:self.max_len, :] for x in data]
        all_data = (np.array(data)/100)-0.5
        all_labels = np.array(labels).astype(np.int32)

        train_data, test_data, train_labels, test_labels = train_test_split(
            all_data, all_labels, test_size=0.25, random_state=1000)
        if self.is_train:
            self.data, self.labels = train_data, train_labels
        else:
            self.data, self.labels = test_data, test_labels
        print(np.unique(self.labels, return_counts=True))

    def to_dataset(self):
        """ returns a tensorflow dataset object """
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.data, self.labels))
        return dataset


if __name__ == '__main__':
    adl_dataset = ADLDataset('dataset/adl', is_train=True, mini=False)
    print(adl_dataset.class2idx)
    print(adl_dataset.data.shape)
    print(adl_dataset.labels.shape)
    print(adl_dataset.to_dataset())
