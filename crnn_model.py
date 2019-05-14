"""
Conditional genration using RNN
"""

import io
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import datetime

from har_dataset import HARDataset
import tb_utils

tf.enable_eager_execution()


flags = tf.app.flags
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_string('model_name', 'har_crnn', 'Model name')
flags.DEFINE_string('restore', None, 'checkpoint directory')
flags.DEFINE_boolean('sample', False, 'Generate Samples')


def gen_plot(samples):
    fig = plt.figure(figsize=(18, 4))
    num_samples = samples.shape[0]
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.plot(samples[i])
    image = tb_utils.fig_to_image_tensor(fig)
    return image


class CRNNModel(tf.keras.Model):

    def __init__(self, rnn_units=256):
        super(CRNNModel, self).__init__()
        self.rnn_units = rnn_units
        self.fc = layers.Dense(32)
        self.emb = layers.Embedding(6, 32)
        self.gru = layers.CuDNNGRU(
            self.rnn_units, return_state=True, return_sequences=True)
        self.gru2 = layers.CuDNNGRU(
            self.rnn_units, return_state=True, return_sequences=True)
        self.gru3 = layers.CuDNNGRU(
            self.rnn_units, return_state=True, return_sequences=True)

        self.fc_last = layers.Dense(6)

    def __call__(self, x, y, hidden=None):
        batch_size, time_len, feat_size = x.shape
        if hidden is None:
            hidden = [tf.zeros((batch_size, self.rnn_units)) for _ in range(3)]
        batch_size, time_len, feats = x.shape
        h1 = self.fc(x)
        h2 = self.emb(y)
        h2 = tf.tile(tf.expand_dims(h2, 1), [1, time_len, 1])
        h = tf.concat([h1, h2], axis=-1)
        outputs, last_state1 = self.gru(h, hidden[0])
        outputs, last_state2 = self.gru2(outputs, hidden[1])
        outputs, last_state3 = self.gru3(outputs, hidden[2])
        preds = self.fc_last(outputs)
        return preds, [last_state1, last_state2, last_state3]

    def sample(self, labels, max_len=128):
        """ generates samples conditioned on the given label """
        num_examples = labels.shape[0]
        step_pred = tf.convert_to_tensor(
            [[[0. for _ in range(6)]] for _ in range(num_examples)])
        preds = []
        last_state = None
        for _ in range(max_len):
            step_pred, last_state = self(step_pred, labels, last_state)
            preds.append(step_pred)
        output = tf.concat(preds, axis=1)
        return output.numpy()


if __name__ == '__main__':
    FLAGS = flags.FLAGS

    train_data = HARDataset(
        './dataset/har', is_train=True).to_dataset().batch(FLAGS.batch_size)
    test_data = HARDataset(
        './dataset/har', is_train=False).to_dataset().batch(FLAGS.batch_size)

    model = CRNNModel()
    optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    model_name = '{}/{}'.format(
        FLAGS.model_name, datetime.datetime.now().strftime('%m_%d_%H_%M'))
    log_dir = './logs/{}'.format(model_name)
    save_dir = './save/{}'.format(model_name)
    file_writer = tf.contrib.summary.create_file_writer(log_dir)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_prefix = '{}/ckpt'.format(save_dir)

    test_samples = model.sample(tf.range(6))
    tf.contrib.summary.image('sample', gen_plot(test_samples), step=0)
    checkpoint = tf.train.Checkpoint(model=model)
    if FLAGS.restore is not None:
        status = checkpoint.restore(
            tf.train.latest_checkpoint('./save/'+FLAGS.restore))
        status.assert_consumed()
        print('Model restored from {}'.format(FLAGS.restore))

    if FLAGS.sample:
        assert FLAGS.restore is not None, 'Must provide checkpoint'
        uniform_logits = tf.log([[10.0 for _ in range(6)]])
        cond_labels = tf.cast(tf.random.categorical(
            uniform_logits, 10000), tf.int32)
        cond_labels = tf.squeeze(cond_labels)
        samples = model.sample(cond_labels)
        samples_out_dir = 'samples/{}'.format(FLAGS.restore)
        if not os.path.exists(samples_out_dir):
            os.makedirs(samples_out_dir)
        np.save('{}/samples_x.npy'.format(samples_out_dir), samples)
        np.save('{}/samples_y.npy'.format(samples_out_dir), cond_labels.numpy())
        print('Saved {} samples to {}'.format(
            samples.shape[0], samples_out_dir))
        sys.exit(0)

    with file_writer.as_default(), tf.contrib.summary.always_record_summaries():
        for epoch in range(1, FLAGS.num_epochs+1):
            epoch_loss = 0
            for batch_x, batch_y in train_data:
                train_x = batch_x[:, :-1, :]
                train_y = batch_x[:, 1:, :]
                with tf.GradientTape() as gt:
                    batch_preds, _ = model(train_x, batch_y)
                    batch_loss = tf.reduce_sum(
                        tf.square(batch_preds - train_y))
                grads = gt.gradient(batch_loss, model.trainable_variables)
                optim.apply_gradients(zip(grads, model.trainable_variables))
                epoch_loss += batch_loss.numpy()
            print('{} - {}'.format(epoch, epoch_loss))
            test_samples = model.sample(tf.range(6))
            tf.contrib.summary.image(
                'sample', gen_plot(test_samples), step=epoch)
            tf.contrib.summary.scalar('training loss', epoch_loss, step=epoch)
            file_writer.flush()
            checkpoint.save(file_prefix=save_prefix)
