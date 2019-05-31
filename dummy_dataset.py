import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split

import tensorflow as tf


class DummyDataset:

    classes = ['sin-1', 'tri', 'sin-2']

    num_feats = 1
    num_labels = len(classes)
    max_len = 128

    @classmethod
    def generate_sin1(cls, length):
        freq = 50
        x_start = np.random.uniform(0, 100)
        x_range = np.linspace(x_start, x_start+length, num=length)
        signal = np.sin(2*np.pi*x_range/freq)
        return signal

    @classmethod
    def generate_sin2(cls, length):
        freq = 20
        x_start = np.random.uniform(0, 100)
        x_range = np.linspace(x_start, x_start+length, num=length)
        signal = np.sin(2*np.pi*x_range/freq)
        return signal

    @classmethod
    def generate_tri(cls, length):
        freq = 50
        x_start = np.random.uniform(0, 100)
        x_range = np.linspace(x_start, x_start+length, num=length)
        signal = (x_range % freq) / freq
        return signal

    @classmethod
    def generate_example(cls, c_name, length=128):
        if c_name == 'sin-1':
            signal = cls.generate_sin1(length)
        elif c_name == 'sin-2':
            signal = cls.generate_sin2(length)
        elif c_name == 'tri':
            signal = cls.generate_tri(length)
        else:
            raise Exception("Unsupported waveform")
        signal = signal + np.random.normal(scale=0.10, size=signal.shape)
        signal = signal.reshape((-1, 1))
        return signal.astype(np.float32)

    def __init__(self, dset_root=None, is_train=True):
        """
        dset_root :  classesummy parameter to maintain consistency with other datasets.

        is_train: booclassesean whether to return training or validation data.
        """

        self.is_train = is_train

        all_data = []
        all_labels = []
        for c_id, c_name in enumerate(self.classes):
            for _ in range(300):
                all_data.append(self.generate_example(c_name))
                all_labels.append(c_id)

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
    tf.enable_eager_execution()
    dummy_dataset = DummyDataset(None, True)
    dset = dummy_dataset.to_dataset().batch(5)
    batch_x, batch_y = next(iter(dset))
    print(batch_x.shape)
    fig, axes = plt.subplots(1, 5, figsize=(20, 4))
    for i in range(5):
        axes[i].plot(batch_x[i].numpy())
        axes[i].set_title('label = {}'.format(batch_y[i]))
    plt.show()
