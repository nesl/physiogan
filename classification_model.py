from models import HARLSTMModel
import os
import sys
import datetime
import numpy as np

import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras import layers

from data_utils import DataFactory
from syn_dataset import SynDataset

from tb_utils import plot_confusion_matrix, fig_to_image_tensor

from sklearn.metrics import confusion_matrix


tf.enable_eager_execution()

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('num_epochs', 50, 'Number of epochs')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_string('model_name', 'lstm', 'Model name')
flags.DEFINE_string('dataset', 'dummy', 'Dataset')
flags.DEFINE_string('restore', None, 'checkpoint directory')
flags.DEFINE_boolean('evaluate', False, 'Run evaluation only')
flags.DEFINE_string('train_syn', None, 'Synthetic samples to train on')
flags.DEFINE_string('evaluate_syn', None, 'Synthetic dataset to evaluate')
flags.DEFINE_boolean('augment', False, 'Augment two datasets')


def train_epoch(model, dataset):
    epoch_loss = 0.0
    num_examples = 0
    accuracy_metric = tf.keras.metrics.Accuracy()
    loss_metric = tf.keras.metrics.Mean()
    for batch_idx, (batch_x, batch_y) in enumerate(train_data):
        with tf.GradientTape() as gt:
            batch_size = int(batch_x.shape[0])
            model_logits = model(batch_x)
            model_pred = tf.cast(tf.arg_max(model_logits, 1), tf.int32)
            batch_loss = tf.losses.sparse_softmax_cross_entropy(
                batch_y, model_logits, reduction=tf.losses.Reduction.SUM_OVER_BATCH_SIZE)
            accuracy_metric.update_state(batch_y, model_pred)
            loss_metric.update_state(batch_loss)
            num_examples += batch_size

        grads = gt.gradient(batch_loss, model.trainable_variables)
        optim.apply_gradients(zip(grads, model.trainable_variables))
    epoch_loss = loss_metric.result()
    epoch_accuracy = accuracy_metric.result()

    return epoch_loss, epoch_accuracy


def evaluate(model, dataset):
    num_examples = 0

    all_true = []
    all_preds = []
    accuracy_metric = tf.keras.metrics.Accuracy()
    for batch_x, batch_y in dataset:
        model_logits = model(batch_x)
        model_pred = tf.cast(tf.arg_max(model_logits, 1), tf.int32)
        accuracy_metric.update_state(batch_y, model_pred)
        num_examples += int(batch_x.shape[0])
        all_preds.append(model_pred.numpy())
        all_true.append(batch_y.numpy())

    all_true = np.concatenate(all_true)
    all_preds = np.concatenate(all_preds)

    conf_mat = confusion_matrix(all_true, all_preds)
    return accuracy_metric.result(), conf_mat


if __name__ == '__main__':

    FLAGS = flags.FLAGS
    model_tag = '{}_{}'.format(FLAGS.dataset, FLAGS.model_name)
    train_data, test_data, metadata = DataFactory.create_dataset(
        FLAGS.dataset)
    if FLAGS.train_syn is not None:
        # train on synthetic data
        syn_train_dataset = SynDataset(
            FLAGS.train_syn)
        assert syn_train_dataset.num_feats == metadata.num_feats and syn_train_dataset.num_labels == metadata.num_labels, 'Datasets mismatch'
        if FLAGS.augment:
            syn_data = syn_train_dataset.to_dataset()
            #syn_data = syn_data.shuffle(10000).take(1000)
            # TODO(malzantot): make sure that size of datasets are reasonably equal
            train_data = train_data.concatenate(syn_data)

            model_tag = '{}_{}'.format('aug', model_tag)
            print('**** Will train on Augmented data !! ')
        else:
            train_data = syn_train_dataset.to_dataset()
            #train_data = train_data.shuffle(10000).take(10000)
            model_tag = '{}_{}'.format('syn', model_tag)
            print('**** Will train on Synthetic data !! ')
    train_data = train_data.batch(FLAGS.batch_size)
    test_data = test_data.batch(FLAGS.batch_size)

    model = HARLSTMModel(metadata.num_feats, metadata.num_labels)
    optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    model_name = '{}/{}'.format(
        model_tag, datetime.datetime.now().strftime('%m_%d_%H_%M'))
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
            test_data = SynDataset(
                FLAGS.evaluate_syn)
            assert metadata.num_feats == test_data.num_feats and metadata.num_labels == test_data.num_labels, 'Datasets mismatch'
            test_data = test_data.to_dataset().batch(FLAGS.batch_size)
        test_accuracy, conf_mat = evaluate(model, test_data)
        print('Test accuracy = {:.2f}'.format(test_accuracy))
        print('Confusion matrix = \n {}'.format(conf_mat))
        cm_fig = plot_confusion_matrix(
            conf_mat, metadata.classes, normalize=True)
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
                conf_mat, metadata.classes, normalize=True)
            cm_tensor = fig_to_image_tensor(cm_fig)
            tf.contrib.summary.image('confusion matrix', cm_tensor, step=epoch)
            checkpoint = tf.train.Checkpoint(model=model)
            checkpoint.save(file_prefix=save_prefix)
