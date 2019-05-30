"""
Author: Moustafa Alzantot (malzantot@ucla.edu)a
"""


import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split


class ADLDataset:
    classes = ['Brush_teeth', 'Climb_stairs', 'Comb_hair',
               'Descend_stairs', 'Getup_bed', 'Pour_water', 'Sitdown_chair', 'Standup_chair', 'Walk']
    num_labels = len(classes)
    num_feats = 3
    max_len = 125

    def __init__(self, ds_root, is_train=True):
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

        min_len = min([x.shape[0] for x in data])
        # clip
        data = [x[:min_len, :] for x in data]
        all_data = (np.array(data)/100)-0.5
        all_labels = np.array(labels).astype(np.int32)
        if self.is_train:
            self.data, _, self.labels, _ = train_test_split(
                all_data, all_labels, test_size=0.25)
        else:
            _, self.data, _, self.labels = train_test_split(
                all_data, all_labels, test_size=0.25)

    def to_dataset(self):
        """ returns a tensorflow dataset object """
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.data, self.labels))
        return dataset


if __name__ == '__main__':
    adl_dataset = ADLDataset('dataset/adl', is_train=False)
    print(adl_dataset.class2idx)
    print(adl_dataset.data.shape)
    print(adl_dataset.labels.shape)
    print(adl_dataset.to_dataset())
