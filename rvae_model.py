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
from tensorflow.losses import sparse_softmax_cross_entropy

from models import RVAEModel, HARLSTMModel, RNNDiscriminator, ConvDiscriminator, RVAEModel
tf.enable_eager_execution()

flags = tf.app.flags
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs')
flags.DEFINE_integer(
    "mle_epochs", 10, "Number of epochs to train using MLE only")
flags.DEFINE_integer('disc_pre_train_epochs', 5,
                     'Number of epochs to pre-train the discriminator')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_integer('z_dim', 64, 'Latent space vector size')
flags.DEFINE_string('model_name', 'rvae', 'Model name')
flags.DEFINE_string('dataset', 'dummy', 'dataset')
flags.DEFINE_string('restore', None, 'checkpoint directory')
flags.DEFINE_boolean('sample', False, 'Generate Samples')
flags.DEFINE_string('aux_restore', None,
                    'checkpoint directory for discriminator')


def gen_plot(samples):
    fig = plt.figure(figsize=(18, 4))
    num_samples = samples.shape[0]
    for i in range(num_samples):
        plt.subplot(1, num_samples, i+1)
        plt.plot(samples[i])
    image = tb_utils.fig_to_image_tensor(fig)
    return image


def pretrain_disc_epoch(model, disc_model, train_data, d_optim):
    loss_metric = tf.keras.metrics.Mean()
    d_accuracy_metric = tf.keras.metrics.Accuracy()
    for batch_x, batch_y in train_data:
        batch_size = int(batch_x.shape[0])

        with tf.GradientTape() as d_tape:
            #batch_preds, _ = model(train_x, batch_y, init_hidden)

            sampling_size = max(1, batch_size // (disc_model.num_labels))
            cond_labels = tf.random.uniform(
                minval=0, maxval=metadata.num_labels, shape=(sampling_size,), dtype=tf.int32)
            sampling_z = model.init_hidden(sampling_size)
            samples = model.sample(cond_labels, sampling_z,
                                   max_len=metadata.max_len)
            d_out_real = disc_model(batch_x[:, ::, :])
            d_out_fake = disc_model(samples[:, ::, :])
            d_out = tf.concat([d_out_real, d_out_fake], axis=0)
            d_target = tf.concat([batch_y,
                                  disc_model.num_labels*tf.ones(shape=(sampling_size,), dtype=tf.int32)], axis=0)
            # d_loss
            d_loss = sparse_softmax_cross_entropy(
                d_target, d_out) / int(d_target.shape[0])
            d_pred = tf.argmax(d_out, axis=1)

        print('\t', d_loss.numpy())

        loss_metric.update_state(d_loss)
        d_accuracy_metric.update_state(d_target, d_pred)
        d_grads = d_tape.gradient(d_loss, disc_model.trainable_variables)
        d_optim.apply_gradients(zip(d_grads, disc_model.trainable_variables))

    return loss_metric.result(), d_accuracy_metric.result()


def evaluate_samples(model, eval_model, metadata):
    accuracy_metric = tf.keras.metrics.Accuracy()
    for _ in range(10):
        sampling_size = 64
        cond_labels = tf.random.uniform(
            minval=0, maxval=metadata.num_labels, shape=(sampling_size,), dtype=tf.int32)
        # cond_labels = tf.squeeze(cond_labels)
        sampling_z = model.init_hidden(sampling_size)
        samples = model.sample(cond_labels,
                               max_len=metadata.max_len)
        model_preds = tf.argmax(eval_model(samples), axis=1)
        accuracy_metric.update_state(cond_labels, model_preds)
    return accuracy_metric.result()


if __name__ == '__main__':
    FLAGS = flags.FLAGS

    train_data, test_data, metadata = DataFactory.create_dataset(
        FLAGS.dataset)
    train_data = train_data.batch(FLAGS.batch_size)
    test_data = test_data.batch(FLAGS.batch_size)

    aux_model = HARLSTMModel(
        num_feats=metadata.num_feats, num_labels=metadata.num_labels)

    aux_checkpoint = tf.train.Checkpoint(model=aux_model)
    if FLAGS.aux_restore is not None:
        status = aux_checkpoint.restore(
            tf.train.latest_checkpoint('./save/'+FLAGS.aux_restore))
        print('Aux. classifier Model restored from {}'.format(FLAGS.aux_restore))

    model = RVAEModel(num_feats=metadata.num_feats,
                      num_labels=metadata.num_labels)

    # disc_model = HARLSTMModel(metadata.num_feats, metadata.num_labels)
    # if FLAGS.disc_restore:
    d_model = ConvDiscriminator(
        num_feats=metadata.num_feats, num_labels=metadata.num_labels)
    d_optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
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
            if epoch <= FLAGS.mle_epochs:
                epoch_recon_loss, epoch_kl_loss, epoch_total_loss =
                print('{} - {}'.format(epoch, epoch_total_loss))
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
            print('{} - ({:.3f}, {:.3f})'.format(epoch,
                                                 epoch_recon_loss, epoch_kl_loss))
            if epoch % 10 == 0:
                sampling_acc = evaluate_samples(model, aux_model, metadata)
                print('** {} ** '.format(sampling_acc))
                tf.contrib.summary.scalar(
                    'sampling acc', sampling_acc, step=epoch)
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
