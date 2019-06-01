"""
Conditional genration using recurrent VAE
"""

import io
import os
import sys
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
import datetime


import tb_utils
from data_utils import DataFactory
from syn_dataset import SynDataset

from models import HARLSTMModel, RVAEModel
tf.enable_eager_execution()

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_integer('z_dim', 64, 'Latent space vector size')
flags.DEFINE_string('model_name', 'rvae', 'Model name')
flags.DEFINE_string('dataset', 'dummy', 'dataset')
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


def train_rvae(model, train_data, optim):
    recon_metric = tf.keras.metrics.Mean()
    kl_metric = tf.keras.metrics.Mean()
    for batch_x, batch_y in train_data:
        batch_size = int(batch_x.shape[0])
        with tf.GradientTape() as gt:
            batch_preds, mu, log_var = model(batch_x, batch_y)
            recon_loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(
                tf.square(batch_preds - batch_x), axis=2), axis=1), axis=0)
            kl_loss = -0.5 * tf.reduce_mean(tf.reduce_mean(1 + log_var - mu**2 -
                                                           tf.exp(log_var), axis=1), axis=0)
            recon_metric.update_state(recon_loss)
            kl_metric.update_state(kl_loss)
            total_loss = recon_loss + kl_loss
        grads = gt.gradient(total_loss, model.trainable_variables)
        clipped_grads, _ = tf.clip_by_global_norm(grads, 1.0)
        optim.apply_gradients(zip(grads, model.trainable_variables))

    epoch_recon_metric = recon_metric.result()
    epoch_kl_metric = kl_metric.result()
    epoch_loss = epoch_recon_metric+epoch_kl_metric

    return epoch_recon_metric, epoch_kl_metric, epoch_loss


if __name__ == '__main__':
    FLAGS = flags.FLAGS

    train_data, test_data, metadata = DataFactory.create_dataset(
        FLAGS.dataset)
    train_data = train_data.batch(FLAGS.batch_size)
    test_data = test_data.batch(FLAGS.batch_size)

    model = RVAEModel(num_feats=metadata.num_feats,
                      num_labels=metadata.num_labels)

    # disc_model = HARLSTMModel(metadata.num_feats, metadata.num_labels)
    # if FLAGS.disc_restore:

    optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    model_tag = '{}_{}'.format(FLAGS.model_name, FLAGS.dataset)
    model_name = '{}/{}'.format(
        model_tag, datetime.datetime.now().strftime('%m_%d_%H_%M'))
    log_dir = './logs/{}'.format(model_name)
    save_dir = './save/{}'.format(model_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_prefix = '{}/ckpt'.format(save_dir)

    checkpoint = tf.train.Checkpoint(model=model)
    if FLAGS.restore is not None:
        status = checkpoint.restore(
            tf.train.latest_checkpoint('./save/'+FLAGS.restore))
        # status.assert_consumed()
        print('Model restored from {}'.format(FLAGS.restore))

    if FLAGS.sample:
        assert FLAGS.restore is not None, 'Must provide checkpoint'
        uniform_logits = tf.log([[10.0 for _ in range(metadata.num_labels)]])
        cond_labels = tf.cast(tf.random.categorical(
            uniform_logits, 10000), tf.int32)
        cond_labels = tf.squeeze(cond_labels)
        samples = model.sample(cond_labels, max_len=metadata.max_len)
        samples_out_dir = 'samples/{}'.format(FLAGS.restore)
        if not os.path.exists(samples_out_dir):
            os.makedirs(samples_out_dir)
        np.save('{}/samples_x.npy'.format(samples_out_dir), samples)
        np.save('{}/samples_y.npy'.format(samples_out_dir), cond_labels.numpy())
        print('Saved {} samples to {}'.format(
            samples.shape[0], samples_out_dir))
        sys.exit(0)
    file_writer = tf.contrib.summary.create_file_writer(log_dir)
    test_samples = model.sample(
        tf.range(metadata.num_labels),  max_len=metadata.max_len)
    tf.contrib.summary.image('sample', gen_plot(test_samples.numpy()), step=0)
    with file_writer.as_default(), tf.contrib.summary.always_record_summaries():

        for epoch in range(1, FLAGS.num_epochs+1):
            epoch_recon_loss, epoch_kl_loss, epoch_total_loss = train_rvae(
                model, train_data, optim)
            print('{} - ({:.3f}, {:.3f})'.format(epoch,
                                                 epoch_recon_loss, epoch_kl_loss))
            test_samples = model.sample(
                tf.range(metadata.num_labels), max_len=metadata.max_len)
            tf.contrib.summary.image(
                'sample', gen_plot(test_samples.numpy()), step=epoch)
            tf.contrib.summary.scalar(
                'recon loss', epoch_recon_loss, step=epoch)
            tf.contrib.summary.scalar('kl loss', epoch_kl_loss, step=epoch)
            tf.contrib.summary.scalar(
                'training loss', epoch_total_loss, step=epoch)
            file_writer.flush()
            checkpoint.save(file_prefix=save_prefix)
