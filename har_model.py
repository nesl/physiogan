import os
import sys
import datetime
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers
from har_dataset import HARDataset

from tb_utils import plot_confusion_matrix, fig_to_image_tensor

from sklearn.metrics import confusion_matrix
tf.enable_eager_execution()

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('num_epochs', 50, 'Number of epochs')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_string('model_name', 'har_lstm', 'Model name')
flags.DEFINE_string('restore', None, 'checkpoint directory')
flags.DEFINE_boolean('evaluate', False, 'Run evaluation only')
flags.DEFINE_string('train_syn', None, 'Synthetic samples to train on')
flags.DEFINE_string('evaluate_syn', None, 'Synthetic dataset to evaluate')


class HARMLPModel(tf.keras.Model):
    def __init__(self):
        super(HARMLPModel, self).__init__()
        self.fc1 = layers.Dense(256, activation='relu')
        self.fc2 = layers.Dense(256, activation='relu')
        self.fc3 = layers.Dense(6, activation=None)

    def __call__(self, x):
        out = tf.reshape(x, [-1, 128*6])
        out = self.fc1(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out


class HARLSTMModel(tf.keras.Model):
    def __init__(self):
        super(HARLSTMModel, self).__init__()
        self.lstm = layers.CuDNNGRU(
            256, return_sequences=True, return_state=True)
        self.lstm2 = layers.CuDNNGRU(
            256, return_sequences=True, return_state=True)
        self.lstm3 = layers.CuDNNGRU(
            256, return_sequences=True, return_state=True)
        self.fc = layers.Dense(6, activation=None)

    def __call__(self, x):
        out, last_h = self.lstm(x)
        out, last_h = self.lstm2(out)
        out, last_h = self.lstm3(out)
        out = self.fc(out[:, -1, :])
        return out


def train_epoch(model, dataset):
    epoch_loss = 0.0
    num_examples = 0
    num_correct = 0
    for batch_idx, (batch_x, batch_y) in enumerate(train_data):
        with tf.GradientTape() as gt:
            batch_size = int(batch_x.shape[0])
            model_logits = model(batch_x)
            model_pred = tf.cast(tf.arg_max(model_logits, 1), tf.int32)
            batch_loss = tf.losses.sparse_softmax_cross_entropy(
                batch_y, model_logits, reduction=tf.losses.Reduction.SUM_BY_NONZERO_WEIGHTS)
            num_correct += tf.reduce_sum(tf.cast(tf.equal(model_pred,
                                                          batch_y), tf.float32)).numpy()
            num_examples += batch_size

        grads = gt.gradient(batch_loss, model.trainable_variables)
        optim.apply_gradients(zip(grads, model.trainable_variables))
        epoch_loss += batch_loss
    epoch_loss /= batch_idx
    epoch_accuracy = (num_correct / num_examples)

    return epoch_loss, epoch_accuracy


def evaluate(model, dataset):
    num_examples = 0
    num_correct = 0

    all_true = []
    all_preds = []
    for batch_x, batch_y in dataset:
        model_logits = model(batch_x)
        model_pred = tf.cast(tf.arg_max(model_logits, 1), tf.int32)
        num_correct += tf.reduce_sum(
            tf.cast(tf.equal(model_pred, batch_y), tf.float32))
        num_examples += int(batch_x.shape[0])
        all_preds.append(model_pred.numpy())
        all_true.append(batch_y.numpy())

    all_true = np.concatenate(all_true)
    all_preds = np.concatenate(all_preds)

    conf_mat = confusion_matrix(all_true, all_preds)
    return num_correct / num_examples, conf_mat


if __name__ == '__main__':

    FLAGS = flags.FLAGS

    if FLAGS.train_syn is None:
        # train on real data
        train_data = HARDataset(
            './dataset/har', is_train=True).to_dataset().batch(FLAGS.batch_size)
    else:
        # train on synthetic data
        train_data = HARDataset(FLAGS.train_syn,
                                is_syn=True).to_dataset().batch(FLAGS.batch_size)
    test_data = HARDataset(
        './dataset/har', is_train=False).to_dataset().batch(FLAGS.batch_size)

    model = HARLSTMModel()
    optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    model_name = '{}/{}'.format(
        FLAGS.model_name, datetime.datetime.now().strftime('%m_%d_%H_%M'))
    log_dir = './logs/{}'.format(model_name)
    save_dir = './save/{}'.format(model_name)
    save_prefix = '{}/ckpt'.format(save_dir)

    if FLAGS.restore is not None:
        checkpoint = tf.train.Checkpoint(model=model)
        status = checkpoint.restore(
            tf.train.latest_checkpoint('./save/'+FLAGS.restore))
        # status.assert_consumed()
        print('Model restored from {}'.format(FLAGS.restore))

    if FLAGS.evaluate or FLAGS.evaluate_syn:
        assert FLAGS.restore is not None, "must provide checkpoint"
        if FLAGS.evaluate_syn:
            test_data = HARDataset(FLAGS.evaluate_syn,
                                   is_syn=True).to_dataset().batch(FLAGS.batch_size)

        test_accuracy, conf_mat = evaluate(model, test_data)
        print('Test accuracy = {:.2f}'.format(test_accuracy))
        print('Confusion matrix = \n {}'.format(conf_mat))
        cm_fig = plot_confusion_matrix(
            conf_mat, HARDataset.classes, normalize=True)
        plt.show()
        sys.exit(0)

    file_writer = tf.contrib.summary.create_file_writer(log_dir)

    with file_writer.as_default(), tf.contrib.summary.always_record_summaries():
        for epoch in range(1, FLAGS.num_epochs+1):
            epoch_loss, epoch_accuracy = train_epoch(model, train_data)
            test_accuracy, conf_mat = evaluate(model, test_data)
            print('{} - {} - {} - {}'.format(epoch,
                                             epoch_loss, epoch_accuracy, test_accuracy))
            tf.contrib.summary.scalar('training loss', epoch_loss, step=epoch)
            tf.contrib.summary.scalar(
                'train accuracy', epoch_accuracy, step=epoch)
            tf.contrib.summary.scalar(
                'test accuracy', test_accuracy, step=epoch)
            cm_fig = plot_confusion_matrix(
                conf_mat, HARDataset.classes, normalize=True)
            cm_tensor = fig_to_image_tensor(cm_fig)
            tf.contrib.summary.image('confusion matrix', cm_tensor, step=epoch)
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.save(file_prefix=save_prefix)
