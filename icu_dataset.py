import matplotlib.pyplot as plt
import os
import random
from scipy.io import loadmat
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_preprocs import subsample
import numpy as np
import matplotlib
import pandas as pd
matplotlib.use('agg')

tf.enable_eager_execution()


class ICUDataset:
    classes = ['Died', 'Survived']
    num_labels = len(classes)
    num_feats = 1
    max_len = 40

    def __init__(self, ds_root, is_train=True, mini=False):
        self.root = ds_root
        self.is_train = is_train
        self.mini = mini

        if self.is_train:
            data_path = os.path.join(self.root, 'set-a')
            labels_path = os.path.join(self.root, 'outcome_a.txt')
        else:
            data_path = os.path.join(self.root, 'set-b')
            labels_path = os.path.join(self.root, 'outcome_b.txt')

        records = [f for f in os.listdir(data_path) if f.endswith('.txt')]
        labels_data = pd.read_csv(labels_path, index_col='RecordID')
        labels_dict = {idx: int(row['Survival'] == -1)
                       for i, row in df.to_dict('index').items()}
        import pdb
        pdb.set_trace()

        print(np.unique(self.labels, return_counts=True))

    def to_dataset(self):
        """ returns a tensorflow dataset object """
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.data, self.labels))
        return dataset


if __name__ == '__main__':
    train_dataset = ICUDataset('dataset/icu', is_train=True, mini=True)
    test_dataset = ICUDataset('dataset/icu', is_train=False, mini=True)
