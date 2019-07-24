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


def parse_record(path, record, var_list, min_len=16):
    df = pd.read_csv(os.path.join(path, record))
    vals_list = []
    for var_name in var_list:
        var_vals = df[df.Parameter == var_name].Value.values.astype(np.float32)
        if var_vals.shape[0] < min_len:
            return None
        vals_list.append(var_vals[-min_len:])

    return np.stack(vals_list).transpose(1, 0)


class ICUDataset:
    classes = ['Died', 'Survived']
    num_labels = len(classes)
    num_feats = 1
    max_len = 40

    var_list = ['HR', 'Temp', 'MAP']

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

        self.records = [f for f in os.listdir(data_path) if f.endswith('.txt')]
        labels_data = pd.read_csv(labels_path, index_col='RecordID')
        self.labels_dict = {idx: int(row['Survival'] == -1)
                            for idx, row in labels_data.to_dict('index').items()}

        record_data = [(parse_record(data_path, record, self.var_list),
                        self.labels_dict[int(record[:-4])]) for record in self.records]
        record_data = [(x, y) for (x, y) in record_data if x is not None]
        self.labels = np.array([y for x, y in record_data]).astype(np.int32)
        self.data = np.array([x for x, yx in record_data]).astype(np.float32)
        print(np.unique(self.labels, return_counts=True))

    def to_dataset(self):
        """ returns a tensorflow dataset object """
        dataset = tf.data.Dataset.from_tensor_slices(
            (self.data, self.labels))
        return dataset


if __name__ == '__main__':
    train_dataset = ICUDataset('dataset/icu', is_train=True, mini=True)
    test_dataset = ICUDataset('dataset/icu', is_train=False, mini=True)
