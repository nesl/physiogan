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
from data_utils import DataFactory
import tb_utils
from models import CRGANModel, ClassModel, ConvDiscriminator, RVAEModel
from train_utils import train_mse_epoch, train_adv_epoch, mse_train_g_epoch, adv_train_d_epoch,  evaluate_samples, gen_plot
import matplotlib as mpl
mpl.use('agg')


tf.enable_eager_execution()


flags = tf.app.flags
flags.DEFINE_integer('batch_size', 256, 'batch size')
flags.DEFINE_integer('num_epochs', 5000, 'Number of epochs')
flags.DEFINE_integer('z_dim', 16, 'Size of latent space noise vector')
flags.DEFINE_boolean('bidir_encoder', True, 'Use a bidirectional encoder')
flags.DEFINE_boolean(
    'z_context', True, 'Use z as a context input to the decoder timesteps')

flags.DEFINE_integer('disc_pre_train_epochs', 5,
                     'Number of epochs to pre-train the discriminator')
flags.DEFINE_integer('num_units', 128, 'Number of RNN units')
flags.DEFINE_string('dataset', 'dummy', "dataset")
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_string('restore', None, 'checkpoint directory')
flags.DEFINE_string('aux_restore', None,
                    'checkpoint directory for discriminator')
flags.DEFINE_boolean('sample', False, 'Generate Samples')
flags.DEFINE_float('sampling_scale', 1.0,
                   'Scale of sampling size relative to the training size')
flags.DEFINE_string('model_type', 'rvae', 'MOdel name')
flags.DEFINE_boolean('filter_samples', False,
                     'Use discriminator to filter samples')

if __name__ == '__main__':
    FLAGS = flags.FLAGS

    train_data, test_data, metadata = DataFactory.create_dataset(
        FLAGS.dataset)
    train_data = train_data.batch(FLAGS.batch_size)
    test_data = test_data.batch(FLAGS.batch_size)

    if FLAGS.model_type == 'crnn':
        g_model = CRGANModel(num_feats=metadata.num_feats,
                             num_labels=metadata.num_labels,
                             z_dim=FLAGS.z_dim,
                             num_units=FLAGS.num_units)
        use_mle = True
    elif FLAGS.model_type == 'rganar':
        # RGAN with autoregressive decoder
        g_model = CRGANModel(num_feats=metadata.num_feats,
                             num_labels=metadata.num_labels,
                             z_dim=FLAGS.z_dim,
                             num_units=FLAGS.num_units)
        use_mle = False
    elif FLAGS.model_type == 'rgan':
        g_model = CRGANModel(num_feats=metadata.num_feats,
                             num_labels=metadata.num_labels,
                             z_dim=FLAGS.z_dim,
                             num_units=FLAGS.num_units,
                             autoregressive=False)
        use_mle = False
    elif FLAGS.model_type == 'crvae':
        # conditional recurrent variational autoencoder
        g_model = RVAEModel(num_feats=metadata.num_feats,
                            z_dim=FLAGS.z_dim,
                            num_labels=metadata.num_labels,
                            enc_rnn_units=FLAGS.num_units,
                            dec_rnn_units=FLAGS.num_units,
                            bidir_encoder=FLAGS.bidir_encoder)
        use_mle = True
    elif FLAGS.model_type == 'physgan':
        g_model = RVAEModel(num_feats=metadata.num_feats,
                            z_dim=FLAGS.z_dim,
                            num_labels=metadata.num_labels,
                            enc_rnn_units=FLAGS.num_units,
                            dec_rnn_units=FLAGS.num_units,
                            bidir_encoder=FLAGS.bidir_encoder)
        use_mle = False
    else:
        raise NotImplementedError("Unsupported model type")
    g_optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

    aux_model = ClassModel(
        num_feats=metadata.num_feats, num_labels=metadata.num_labels)

    aux_checkpoint = tf.train.Checkpoint(model=aux_model)
    if FLAGS.aux_restore is not None:
        status = aux_checkpoint.restore(
            tf.train.latest_checkpoint('./save/'+FLAGS.aux_restore))
        print('Aux. classifier Model restored from {}'.format(FLAGS.aux_restore))
        # status.assert_consumed()

    d_model = ConvDiscriminator(
        num_feats=metadata.num_feats, num_labels=metadata.num_labels)
    d_optim = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)
    model_tag = '{}_{}'.format(
        FLAGS.dataset,  FLAGS.model_type)
    model_name = '{}/{}'.format(
        model_tag, datetime.datetime.now().strftime('%m_%d_%H_%M'))
    log_dir = './logs/{}'.format(model_name)
    save_dir = './save/{}'.format(model_name)
    d_save_dir = os.path.join(save_dir, 'disc')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
        os.mkdir(d_save_dir)

    save_prefix = '{}/ckpt'.format(save_dir)
    d_save_prefix = '{}/ckpt'.format(d_save_dir)

    fixed_sampling_batch = 3
    fixed_sampling_size = fixed_sampling_batch*metadata.num_labels
    fixed_sampling_z = tf.random_normal(
        shape=(fixed_sampling_size, g_model.z_dim))
    fixed_sampling_labels = tf.tile(tf.range(metadata.num_labels), [
        fixed_sampling_batch])
    test_samples = g_model.sample(
        labels=fixed_sampling_labels, z=fixed_sampling_z, max_len=metadata.max_len)
    tf.contrib.summary.image('sample', gen_plot(
        test_samples.numpy(), metadata.num_labels), step=0)
    checkpoint = tf.train.Checkpoint(model=g_model)
    disc_checkpoint = tf.train.Checkpoint(model=d_model)
    if FLAGS.restore is not None:
        ckpt_name = tf.train.latest_checkpoint('./save/'+FLAGS.restore)
        print(ckpt_name)
        status = checkpoint.restore(ckpt_name)
        # status.assert_consumed()
        print('G Model restored from {}'.format(ckpt_name))

        d_ckpt_name = tf.train.latest_checkpoint(
            './save/'+FLAGS.restore+'/disc')
        print(d_ckpt_name)
        status = disc_checkpoint.restore(d_ckpt_name)
        # status.assert_consumed()
        print('D Model restored from {}'.format(d_ckpt_name))

    if FLAGS.sample:
        assert FLAGS.restore is not None, 'Must provide checkpoint'
        uniform_logits = tf.log([[10.0 for _ in range(metadata.num_labels)]])
        sampling_size = int(FLAGS.sampling_scale * metadata.num_examples)
        cond_labels = tf.cast(tf.random.categorical(
            uniform_logits, sampling_size), tf.int32)
        cond_labels = tf.squeeze(cond_labels)
        sampling_z = tf.random_normal(shape=(sampling_size, g_model.z_dim))
        samples = g_model.sample(cond_labels, sampling_z,
                                 max_len=metadata.max_len)

        samples = samples.numpy()
        cond_labels = cond_labels.numpy()

        if FLAGS.filter_samples:
            d_out = tf.argmax(d_model(samples), axis=1)
            d_out = d_out.numpy()
            selected_samples = samples[np.where(d_out == cond_labels)]
            selected_labels = cond_labels[np.where(d_out == cond_labels)]
        else:
            selected_labels = cond_labels
            selected_samples = samples
        samples_out_dir = 'samples/{}'.format(FLAGS.restore)

        if FLAGS.dataset == 'ecg_long':
            samples_out_dir = '{}_{}'.format(
                samples_out_dir, 'long')

        if FLAGS.sampling_scale != 1:
            samples_out_dir = '{}_{}'.format(
                samples_out_dir, FLAGS.sampling_scale)

        if not os.path.exists(samples_out_dir):
            os.makedirs(samples_out_dir)
        np.save('{}/samples_x.npy'.format(samples_out_dir), selected_samples)
        np.save('{}/samples_y.npy'.format(samples_out_dir), selected_labels)
        print('Saved {} samples to {}'.format(
            selected_samples.shape[0], samples_out_dir))
        sys.exit(0)
    file_writer = tf.contrib.summary.create_file_writer(log_dir)

    with file_writer.as_default(), tf.contrib.summary.always_record_summaries():

        for epoch in range(1, FLAGS.num_epochs+1):
            if use_mle:
                epoch_loss, kl_loss = train_mse_epoch(
                    g_model, train_data, g_optim)
                print('{} - {}'.format(epoch, epoch_loss))
            else:
                if epoch == 1:
                    print('*** Pre-Training Discriminator ***')
                    for ii in range(FLAGS.disc_pre_train_epochs):
                        _, d_acc = adv_train_d_epoch(
                            g_model, d_model, train_data, d_optim, metadata.max_len)
                        print('pre: {} - {}'.format(ii, d_acc))
                epoch_loss, epoch_acc, kl_loss = train_adv_epoch(
                    g_model, d_model, train_data, g_optim, d_optim, epoch, metadata.max_len)

                print('{} - {}'.format(epoch, epoch_acc))
            tf.contrib.summary.scalar('training loss', epoch_loss, step=epoch)
            tf.contrib.summary.scalar('KLD', kl_loss, step=epoch)
            if epoch % 10 == 0:
                sampling_acc = evaluate_samples(
                    g_model, aux_model, metadata.max_len)
                print('** {} ** '.format(sampling_acc))
                tf.contrib.summary.scalar(
                    'sampling acc', sampling_acc, step=epoch)
                test_samples = g_model.sample(
                    fixed_sampling_labels, fixed_sampling_z, max_len=metadata.max_len)
                tf.contrib.summary.image(
                    'sample', gen_plot(test_samples.numpy(), g_model.num_labels), step=epoch)
                file_writer.flush()
                checkpoint.save(file_prefix=save_prefix)
                disc_checkpoint.save(file_prefix=d_save_prefix)
