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

from tensorflow.losses import sparse_softmax_cross_entropy
from data_utils import DataFactory
import tb_utils

from models import CRNNModel, HARLSTMModel, RNNDiscriminator, ConvDiscriminator


tf.enable_eager_execution()


flags = tf.app.flags
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('num_epochs', 50, 'Number of epochs')
flags.DEFINE_integer(
    "mle_epochs", 10, "Number of epochs to train using MLE only")
flags.DEFINE_integer('disc_pre_train_epochs', 5,
                     'Number of epochs to pre-train the discriminator')
flags.DEFINE_integer('num_units', 64, 'Number of RNN units')
flags.DEFINE_string('dataset', 'dummy', "dataset")
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_string('model_name', 'crnn', 'Model name')
flags.DEFINE_string('restore', None, 'checkpoint directory')
flags.DEFINE_string('aux_restore', None,
                    'checkpoint directory for discriminator')
flags.DEFINE_boolean('sample', False, 'Generate Samples')


def gen_plot(samples):
    fig = plt.figure(figsize=(18, 4))
    num_samples = samples.shape[0]
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.plot(samples[i])
    image = tb_utils.fig_to_image_tensor(fig)
    return image


def train_epoch(model, train_data, optim):
    loss_metric = tf.keras.metrics.Mean()
    for batch_x, batch_y in train_data:
        batch_size = int(batch_x.shape[0])
        train_x = batch_x[:, :-1, :]
        train_y = batch_x[:, 1:, :]
        init_hidden = [tf.random_normal((batch_size, model.num_units))
                       for _ in range(model.num_layers)]
        with tf.GradientTape() as gt:
            batch_preds, _ = model(train_x, batch_y, init_hidden)
            batch_loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.square(batch_preds - train_y), axis=2),
                                                       axis=1), axis=0)
            loss_metric.update_state(batch_loss)
        grads = gt.gradient(batch_loss, model.trainable_variables)
        optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss_metric.result()


def pretrain_disc_epoch(model, disc_model, train_data, d_optim):
    loss_metric = tf.keras.metrics.Mean()
    d_accuracy_metric = tf.keras.metrics.Accuracy()
    for batch_x, batch_y in train_data:
        batch_size = int(batch_x.shape[0])
        init_hidden = [tf.random_normal((batch_size, model.num_units))
                       for _ in range(model.num_layers)]
        with tf.GradientTape() as d_tape:
            #batch_preds, _ = model(train_x, batch_y, init_hidden)
            cond_labels = tf.random.uniform(
                minval=0, maxval=metadata.num_labels, shape=(batch_size,), dtype=tf.int32)
            sampling_z = [tf.random_normal((batch_size, model.num_units))
                          for _ in range(model.num_layers)]
            samples = model.sample(cond_labels, sampling_z,
                                   max_len=metadata.max_len)
            d_out_real = disc_model(batch_x[:, ::, :], batch_y)
            d_out_fake = disc_model(samples[:, ::, :], cond_labels)
            d_out = tf.concat([d_out_real, d_out_fake], axis=0)
            d_target = tf.concat([tf.ones(shape=(batch_size,), dtype=tf.int32),
                                  tf.zeros(shape=(batch_size,), dtype=tf.int32)], axis=0)
            # d_loss
            d_loss = sparse_softmax_cross_entropy(d_target, d_out) / batch_size
            d_pred = tf.argmax(d_out, axis=1)

        print('\t', d_loss.numpy())

        loss_metric.update_state(d_loss)
        d_accuracy_metric.update_state(d_target, d_pred)
        d_grads = d_tape.gradient(d_loss, disc_model.trainable_variables)
        d_optim.apply_gradients(zip(d_grads, disc_model.trainable_variables))

    return loss_metric.result(), d_accuracy_metric.result()


def adv_train_epoch(model, disc_model, train_data, d_optim, g_optim):
    loss_metric = tf.keras.metrics.Mean()
    d_accuracy_metric = tf.keras.metrics.Accuracy()
    for batch_x, batch_y in train_data:
        batch_size = int(batch_x.shape[0])
        batch_size = int(batch_x.shape[0])
        train_x = batch_x[:, :-1, :]
        train_y = batch_x[:, 1:, :]
        init_hidden = [tf.random_normal((batch_size, model.num_units))
                       for _ in range(model.num_layers)]
        with tf.GradientTape() as d_tape, tf.GradientTape() as g_tape:
            # Reconsturction error
            batch_preds, _ = model(train_x, batch_y, init_hidden)
            recon_loss = tf.reduce_mean(tf.reduce_mean(tf.reduce_mean(tf.square(batch_preds - train_y), axis=2),
                                                       axis=1), axis=0)
            cond_labels = tf.random.uniform(
                minval=0, maxval=metadata.num_labels, shape=(batch_size,), dtype=tf.int32)
            sampling_z = [tf.random_normal((batch_size, model.num_units))
                          for _ in range(model.num_layers)]
            samples = model.sample(cond_labels, sampling_z,
                                   max_len=metadata.max_len)
            d_out_real = disc_model(batch_x[:, ::, :], batch_y)
            d_out_fake = disc_model(samples[:, ::, :], cond_labels)

            # d_loss
            d_out = tf.concat([d_out_real, d_out_fake], axis=0)
            d_target = tf.concat([tf.ones(shape=(batch_size,), dtype=tf.int32),
                                  tf.zeros(shape=(batch_size,), dtype=tf.int32)], axis=0)
            d_pred = tf.argmax(d_out, axis=1)
            d_loss = sparse_softmax_cross_entropy(d_target, d_out) / batch_size
            d_accuracy_metric.update_state(d_target, d_pred)
            cond_labels = tf.random.uniform(
                minval=0, maxval=metadata.num_labels, shape=(batch_size,), dtype=tf.int32)
            sampling_z = [tf.random_normal((batch_size, model.num_units))
                          for _ in range(model.num_layers)]
            samples = model.sample(cond_labels, sampling_z,
                                   max_len=metadata.max_len)
            d_out_fake = disc_model(samples[:, ::, :], cond_labels)
            # g_loss
            g_adv_loss = sparse_softmax_cross_entropy(
                tf.ones(shape=(batch_size,), dtype=tf.int32), d_out_fake) / batch_size
            g_recon_loss = recon_loss
            g_loss = g_adv_loss + 10 * g_recon_loss

        print('\t', d_loss.numpy(), ' ; ', g_loss.numpy())
        loss_metric.update_state(g_loss)
        d_grads = d_tape.gradient(d_loss, disc_model.trainable_variables)
        d_optim.apply_gradients(zip(d_grads, disc_model.trainable_variables))

        g_grads = g_tape.gradient(g_loss, model.trainable_variables)
        g_optim.apply_gradients(zip(g_grads, model.trainable_variables))

    return loss_metric.result(), d_accuracy_metric.result()


def evaluate_samples(model, disc_model, metadata):
    accuracy_metric = tf.keras.metrics.Accuracy()
    for _ in range(10):
        sampling_size = 64
        cond_labels = tf.random.uniform(
            minval=0, maxval=metadata.num_labels, shape=(sampling_size,), dtype=tf.int32)
        # cond_labels = tf.squeeze(cond_labels)
        sampling_z = [tf.random_normal((sampling_size, model.num_units))
                      for _ in range(model.num_layers)]
        samples = model.sample(cond_labels, sampling_z,
                               max_len=metadata.max_len)
        model_preds = tf.argmax(disc_model(samples), axis=1)
        accuracy_metric.update_state(cond_labels, model_preds)
    return accuracy_metric.result()


if __name__ == '__main__':
    FLAGS = flags.FLAGS

    train_data, test_data, metadata = DataFactory.create_dataset(
        FLAGS.dataset)
    train_data = train_data.batch(FLAGS.batch_size)
    test_data = test_data.batch(FLAGS.batch_size)

    model = CRNNModel(num_feats=metadata.num_feats,
                      num_labels=metadata.num_labels,
                      num_units=FLAGS.num_units)

    aux_model = HARLSTMModel(
        num_feats=metadata.num_feats, num_labels=metadata.num_labels)

    aux_checkpoint = tf.train.Checkpoint(model=aux_model)
    if FLAGS.aux_restore is not None:
        status = aux_checkpoint.restore(
            tf.train.latest_checkpoint('./save/'+FLAGS.aux_restore))
        print('Aux. classifier Model restored from {}'.format(FLAGS.aux_restore))
        # status.assert_consumed()

    d_model = ConvDiscriminator(
        num_feats=metadata.num_feats, num_labels=metadata.num_labels)
    optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    d_optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    model_tag = '{}_{}'.format(FLAGS.dataset, FLAGS.model_name)
    model_name = '{}/{}'.format(
        model_tag, datetime.datetime.now().strftime('%m_%d_%H_%M'))
    log_dir = './logs/{}'.format(model_name)
    save_dir = './save/{}'.format(model_name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    save_prefix = '{}/ckpt'.format(save_dir)

    sampling_size = metadata.num_labels
    fixed_z = [tf.random_normal(
        (sampling_size, model.num_units)) for _ in range(model.num_layers)]
    fixed_labels = tf.range(metadata.num_labels)
    test_samples = model.sample(
        labels=fixed_labels, z=fixed_z, max_len=metadata.max_len)
    tf.contrib.summary.image('sample', gen_plot(test_samples.numpy()), step=0)
    checkpoint = tf.train.Checkpoint(model=model)
    if FLAGS.restore is not None:
        ckpt_name = tf.train.latest_checkpoint('./save/'+FLAGS.restore)
        print(ckpt_name)
        status = checkpoint.restore(ckpt_name)
        status.assert_consumed()
        print('Model restored from {}'.format(ckpt_name))

    if FLAGS.sample:
        assert FLAGS.restore is not None, 'Must provide checkpoint'
        uniform_logits = tf.log([[10.0 for _ in range(metadata.num_labels)]])
        sampling_size = 10000
        cond_labels = tf.cast(tf.random.categorical(
            uniform_logits, sampling_size), tf.int32)
        cond_labels = tf.squeeze(cond_labels)
        sampling_z = [tf.random_normal((sampling_size, model.num_units))
                      for _ in range(model.num_layers)]
        samples = model.sample(cond_labels, sampling_z,
                               max_len=metadata.max_len)
        samples_out_dir = 'samples/{}'.format(FLAGS.restore)
        if not os.path.exists(samples_out_dir):
            os.makedirs(samples_out_dir)
        np.save('{}/samples_x.npy'.format(samples_out_dir), samples)
        np.save('{}/samples_y.npy'.format(samples_out_dir), cond_labels.numpy())
        print('Saved {} samples to {}'.format(
            samples.shape[0], samples_out_dir))
        sys.exit(0)
    file_writer = tf.contrib.summary.create_file_writer(log_dir)

    with file_writer.as_default(), tf.contrib.summary.always_record_summaries():

        for epoch in range(1, FLAGS.num_epochs+1):
            if epoch <= FLAGS.mle_epochs:
                epoch_loss = train_epoch(model, train_data, optim)
                print('{} - {}'.format(epoch, epoch_loss))
            else:
                if epoch == (FLAGS.mle_epochs+1):
                    print('*** Pre-Training Discriminator ***')
                    for ii in range(FLAGS.disc_pre_train_epochs):
                        _, disc_acc = pretrain_disc_epoch(
                            model, d_model, train_data, d_optim)
                        print('pre: {} - {}'.format(ii, disc_acc))
                epoch_loss, epoch_acc = adv_train_epoch(
                    model, d_model, train_data, d_optim, optim)

                print('{} - {}'.format(epoch, epoch_acc))

            sampling_acc = evaluate_samples(model, aux_model, metadata)
            print('** {} ** '.format(sampling_acc))
            test_samples = model.sample(
                fixed_labels, fixed_z, max_len=metadata.max_len)
            tf.contrib.summary.scalar('sampling acc', sampling_acc, step=epoch)
            tf.contrib.summary.image(
                'sample', gen_plot(test_samples.numpy()), step=epoch)
            tf.contrib.summary.scalar('training loss', epoch_loss, step=epoch)
            file_writer.flush()
            checkpoint.save(file_prefix=save_prefix)
