import matplotlib.pyplot as plt
import os
import random
from scipy.io import loadmat
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_preprocs import subsample
import numpy as np
import matplotlib
matplotlib.use('agg')

tf.enable_eager_execution()


class ECGDataset:
    classes = ['1 NSR', '4 AFIB', '7 PVC',  '14 LBBBB',  '15 RBBBB']
    num_labels = len(classes)
    num_feats = 1
    max_len = 40

    def __init__(self, ds_root, is_train=True, mini=True):
        self.root = ds_root
        self.is_train = is_train
        self.mini = mini

        if self.mini:
            self.classes = ['1 NSR', '4 AFIB']
            self.num_labels = len(self.classes)
            self.max_len = 40
            seq_end = 500
        else:
            self.classes = ['1 NSR', '4 AFIB']
            self.num_labels = len(self.classes)
            self.max_len = 120
            seq_end = 1300
        classes_found = [x for x in os.listdir(self.root)
                         if os.path.isdir(os.path.join(self.root, x))]
        classes_found = [x for x in self.classes if x in classes_found]

        #self.classes = classes_found[:num_classes]
        print(self.classes)
        print(classes_found)
        assert len(self.classes) == len(
            classes_found), 'Some classes were missing'
        self.class2idx = {x: i for i, x in enumerate(self.classes)}
        self.idx2class = {i: x for x, i in self.class2idx.items()}
        #self.num_classes = len(self.classes)
        self.num_features = 1
        # read files
        all_data = []
        all_labels = []
        for arrhy_class in self.classes:
            class_files = [f for f in os.listdir(os.path.join(self.root, arrhy_class))
                           if f.endswith('.mat')]
            for fname in class_files:
                all_labels.append(self.class2idx[arrhy_class])
                data = loadmat(
                    os.path.join(self.root, arrhy_class, fname))['val'].astype(np.float32).ravel()
                data = data[:, np.newaxis]
                data = (data - np.mean(data)) / (np.std(data)*10)
                all_data.append(data)
        all_data = [subsample(x[100:seq_end], 10) for x in all_data]
        all_data = np.stack(all_data).astype(np.float32)
        all_labels = np.stack(all_labels).astype(np.int32)
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
    train_dataset = ECGDataset('dataset/ecg_data', is_train=True, mini=False)
    test_dataset = ECGDataset('dataset/ecg_data', is_train=False, mini=False)
    print(train_dataset.to_dataset().batch(10))

    train_set = train_dataset.to_dataset().shuffle(1000).batch(10)
    batch_x, batch_y = next(iter(train_set))
    fig, axes = plt.subplots(1, 10)
    print(batch_x.shape, batch_y.shape)
    for i in range(10):
        axes[i].plot(batch_x[i].numpy())
        axes[i].set_title('Label = {}'.format(batch_y[i].numpy()))
    plt.show()
