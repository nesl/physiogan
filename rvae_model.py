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

tf.enable_eager_execution()


flags = tf.app.flags
flags.DEFINE_integer('batch_size', 32, 'batch size')
flags.DEFINE_integer('num_epochs', 10, 'Number of epochs')
flags.DEFINE_float('learning_rate', 0.001, 'learning rate')
flags.DEFINE_integer('z_dim', 64, 'Latent space vector size')
flags.DEFINE_string('model_name', 'rvae', 'Model name')
flags.DEFINE_string('dataset', 'har', 'dataset')
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


class RNNEncoder(tf.keras.Model):
    def __init__(self, rnn_units=64, z_dim=64):
        super(RNNEncoder, self).__init__()
        self.rnn_units = rnn_units
        self.z_dim = z_dim
        self.gru = layers.CuDNNGRU(
            self.rnn_units, return_state=True, return_sequences=True)
        self.fc_mu = layers.Dense(self.z_dim)
        self.fc_logvar = layers.Dense(self.z_dim)

    def reparmeterize(self, mu, sigma):
        eps = tf.random_normal(tf.shape(sigma))
        return tf.multiply(eps, sigma) + mu

    def __call__(self, x, hidden=None):
        batch_size, time_len, feat_size = x.shape
        if hidden is None:
            hidden = tf.zeros([batch_size, self.rnn_units])
        outputs, last_hidden = self.gru(x, hidden)
        mu = self.fc_mu(last_hidden)
        log_var = self.fc_logvar(last_hidden)
        sigma = tf.exp(log_var/2)
        z = self.reparmeterize(mu, sigma)
        return z, mu, log_var


class RNNDecoder(tf.keras.Model):
    def __init__(self, rnn_units=64, num_feats=6, num_labels=6):
        super(RNNDecoder, self).__init__()
        self.rnn_units = rnn_units
        self.num_feats = num_feats
        self.num_labels = num_labels
        self.gru1 = layers.CuDNNGRU(
            self.rnn_units, return_state=True, return_sequences=True)
        self.gru2 = layers.CuDNNGRU(
            self.rnn_units, return_state=True, return_sequences=True)
        self.gru3 = layers.CuDNNGRU(
            self.rnn_units, return_state=True, return_sequences=True)
        self.fc_out = layers.Dense(self.num_feats)
        self.label_emb = layers.Embedding(self.num_labels, 6)

    def __call__(self, x, hidden, y):
        batch_size, time_len, feat_dim = x.shape
        y_emb = self.label_emb(y)
        y_emb = tf.tile(tf.expand_dims(y_emb, axis=1), [1, time_len, 1])

        rnn_input = tf.concat([x, y_emb], axis=-1)
        output1, last_state1 = self.gru1(rnn_input, hidden[0])
        output2, last_state2 = self.gru2(output1, hidden[1])
        output3, last_state3 = self.gru3(output2, hidden[2])

        output = self.fc_out(output3)
        new_hidden = [last_state1, last_state2, last_state3]
        return output, new_hidden


class RVAEModel(tf.keras.Model):

    def __init__(self, enc_rnn_units=64, z_dim=64, dec_rnn_units=64, num_feats=6, num_labels=6, z_input=True):
        super(RVAEModel, self).__init__()
        self.enc_rnn_units = enc_rnn_units
        self.z_dim = z_dim
        self.dec_rnn_units = dec_rnn_units
        self.num_feats = num_feats
        self.num_labels = num_labels
        self.z_input = z_input
        self.encoder = RNNEncoder(enc_rnn_units)
        self.decoder = RNNDecoder(dec_rnn_units, num_feats, num_labels)
        self.fc_z = layers.Dense(6)
        self.fc_hidden = layers.Dense(3*self.dec_rnn_units)

    def __call__(self, x, y):
        batch_size, time_len, feat_dim = x.shape
        z, mu, log_var = self.encoder(x)

        decoder_outputs = []
        # initialize the decoder state with the z vector
        dec_init_state = tf.reshape(self.fc_hidden(z), [3, batch_size, -1])
        init_step = tf.zeros_like(x[:, 0:1, :])
        dec_input = tf.concat([init_step, x[:, :-1]], axis=1)
        if self.z_input:
            z_emb = self.fc_z(z)
            z_with_time = tf.tile(tf.expand_dims(
                z_emb, 1), [1, time_len, 1])
            dec_input = tf.concat([z_with_time, dec_input], axis=2)
        recon_output, _ = self.decoder(dec_input, dec_init_state, y)
        return recon_output, mu, log_var

    def sample(self, labels, max_len=125):
        """ generates samples conditioned on the given label """
        num_examples = int(labels.shape[0])
        z = tf.random_normal(shape=(num_examples, self.decoder.rnn_units))
        last_pred = tf.zeros(shape=(num_examples, 1, self.num_feats))
        preds = []
        last_state = tf.reshape(self.fc_hidden(z), [3, num_examples, -1])
        z_emb = self.fc_z(z)
        z_with_time = tf.expand_dims(z_emb, axis=1)
        for _ in range(max_len):
            if self.z_input:
                step_input = tf.concat([z_with_time, last_pred], axis=2)
            else:
                step_input = last_pred
            last_pred, last_state = self.decoder(
                step_input, last_state, labels)
            preds.append(last_pred)
        output = tf.concat(preds, axis=1)
        return output.numpy()


if __name__ == '__main__':
    FLAGS = flags.FLAGS

    train_data, test_data, metadata = DataFactory.create_dataset(
        FLAGS.dataset)
    train_data = train_data.batch(FLAGS.batch_size)
    test_data = test_data.batch(FLAGS.batch_size)

    model = RVAEModel(num_feats=metadata.num_feats,
                      num_labels=metadata.num_labels)
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
        samples = model.sample(cond_labels)
        samples_out_dir = 'samples/{}'.format(FLAGS.restore)
        if not os.path.exists(samples_out_dir):
            os.makedirs(samples_out_dir)
        np.save('{}/samples_x.npy'.format(samples_out_dir), samples)
        np.save('{}/samples_y.npy'.format(samples_out_dir), cond_labels.numpy())
        print('Saved {} samples to {}'.format(
            samples.shape[0], samples_out_dir))
        sys.exit(0)
    file_writer = tf.contrib.summary.create_file_writer(log_dir)
    test_samples = model.sample(tf.range(metadata.num_labels))
    tf.contrib.summary.image('sample', gen_plot(test_samples), step=0)
    with file_writer.as_default(), tf.contrib.summary.always_record_summaries():
        for epoch in range(1, FLAGS.num_epochs+1):
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            for batch_x, batch_y in train_data:
                with tf.GradientTape() as gt:
                    batch_preds, mu, log_var = model(batch_x, batch_y)
                    recon_loss = tf.reduce_sum(tf.reduce_mean(tf.reduce_mean(
                        tf.square(batch_preds - batch_x), axis=2), axis=1), axis=0)
                    kl_loss = -0.5 * tf.reduce_sum(tf.reduce_mean(1 + log_var - mu**2 -
                                                                  tf.exp(log_var), axis=1), axis=0)
                    total_loss = recon_loss + kl_loss
                grads = gt.gradient(total_loss, model.trainable_variables)
                clipped_grads, _ = tf.clip_by_global_norm(grads, 1.0)
                optim.apply_gradients(zip(grads, model.trainable_variables))
                epoch_recon_loss += recon_loss.numpy()
                epoch_kl_loss += kl_loss.numpy()
            print('{} - ({:.3f}, {:.3f})'.format(epoch,
                                                 epoch_recon_loss, epoch_kl_loss))
            test_samples = model.sample(tf.range(metadata.num_labels))
            tf.contrib.summary.image(
                'sample', gen_plot(test_samples), step=epoch)
            tf.contrib.summary.scalar(
                'recon loss', epoch_recon_loss, step=epoch)
            tf.contrib.summary.scalar('kl loss', epoch_kl_loss, step=epoch)
            tf.contrib.summary.scalar(
                'training loss', epoch_kl_loss+epoch_recon_loss, step=epoch)
            file_writer.flush()
            checkpoint.save(file_prefix=save_prefix)
