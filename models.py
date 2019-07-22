import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


class ConvDiscriminator(tf.keras.Model):
    def __init__(self, num_feats, num_labels):
        super(ConvDiscriminator, self).__init__()
        self.num_feats = num_feats
        self.num_labels = num_labels
        self.conv1 = layers.Conv1D(32, kernel_size=(
            3), strides=(3), padding='same', activation='relu')
        self.conv2 = layers.Conv1D(32, kernel_size=(
            3), strides=(3), padding='same', activation='relu')
        self.conv3 = layers.Conv1D(32, kernel_size=(
            3), strides=(3), padding='same', activation='relu')
        self.fc_out = layers.Dense(self.num_labels+1)

    def __call__(self, x):
        batch_size, time_len, _ = x.shape
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        last_feats = tf.reshape(out, [batch_size, -1])
        out = self.fc_out(last_feats)
        return out, last_feats


class RNNDiscriminator(tf.keras.Model):
    def __init__(self, num_feats, num_labels, num_units=32):
        super(RNNDiscriminator, self).__init__()
        self.num_feats = num_feats
        self.num_labels = num_labels
        self.num_units = num_units

        self.emb_layer = layers.Embedding(self.num_labels, 32)
        self.lstm = layers.CuDNNGRU(
            self.num_units, return_sequences=True, return_state=True)
        self.fc_out = layers.Dense(self.num_labels+1)

    def __call__(self, x):
        out, last_h = self.lstm(x)
        last_out = out[:, -1, :]
        out = self.fc_out(last_out)
        return out, last_out


class ClassModel(tf.keras.Model):
    """
    Classificatin model to classify input samples.
    """

    def __init__(self, num_feats, num_labels, num_units=64):
        super(ClassModel, self).__init__()
        self.num_feats = num_feats
        self.num_labels = num_labels
        self.num_units = num_units
        self.lstm = layers.CuDNNGRU(
            self.num_units, return_sequences=True, return_state=True)
        self.lstm2 = layers.CuDNNGRU(
            self.num_units, return_sequences=True, return_state=True)
        self.lstm3 = layers.CuDNNGRU(
            self.num_units, return_sequences=True, return_state=True)
        self.fc = layers.Dense(self.num_labels, activation=None)

    def __call__(self, x):
        out, last_h = self.lstm(x)
        out, last_h = self.lstm2(out)
        out, last_h = self.lstm3(out)
        out = self.fc(out[:, -1, :])
        return out


class ConvClassModel(tf.keras.Model):
    def __init__(self, num_feats, num_labels):
        super(ConvClassModel, self).__init__()
        self.num_feats = num_feats
        self.num_labels = num_labels
        self.conv1 = layers.Conv1D(32, kernel_size=(
            3), strides=(3), padding='same', activation='relu')
        self.conv2 = layers.Conv1D(32, kernel_size=(
            3), strides=(3), padding='same', activation='relu')
        self.conv3 = layers.Conv1D(32, kernel_size=(
            3), strides=(3), padding='same', activation='relu')
        self.fc_out = layers.Dense(self.num_labels)

    def __call__(self, x):
        batch_size, time_len, _ = x.shape
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        last_feats = tf.reshape(out, [batch_size, -1])
        out = self.fc_out(last_feats)
        return out


class RNNEncoder(tf.keras.Model):
    def __init__(self, rnn_units=64, z_dim=64, bidirectional=True):
        super(RNNEncoder, self).__init__()
        self.rnn_units = rnn_units
        self.z_dim = z_dim
        self.bidirectional = bidirectional
        self.gru = layers.CuDNNGRU(
            self.rnn_units, return_state=True, return_sequences=True)
        if self.bidirectional:
            self.gru = layers.Bidirectional(
                layer=self.gru, merge_mode='concat')
        self.fc_mu = layers.Dense(self.z_dim)
        self.fc_logvar = layers.Dense(self.z_dim)

    def reparmeterize(self, mu, sigma):
        eps = tf.random_normal(tf.shape(sigma))
        return tf.multiply(eps, sigma) + mu

    def __call__(self, x, hidden=None):
        batch_size, time_len, feat_size = x.shape
        if hidden is None:
            if self.bidirectional:
                hidden = [tf.zeros([batch_size, self.rnn_units])
                          for _ in range(2)]
            else:
                hidden = tf.zeros([batch_size, self.rnn_units])

        if self.bidirectional:
            # TODO(malzantot): why it doesn't accept hidden state here ?!
            outputs, forward_last_h, backward_last_h = self.gru(x)
            last_hidden = tf.concat([forward_last_h, backward_last_h], axis=1)
        else:
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

    def __init__(self, enc_rnn_units=64, z_dim=64, dec_rnn_units=64, num_feats=6, num_labels=6, bidir_encoder=True, z_context=True):
        super(RVAEModel, self).__init__()
        self.enc_rnn_units = enc_rnn_units
        self.z_dim = z_dim
        self.dec_rnn_units = dec_rnn_units
        self.num_feats = num_feats
        self.num_labels = num_labels
        self.z_context = z_context
        # TODO(malzantot): add y as an output of the encoder.
        self.encoder = RNNEncoder(
            enc_rnn_units, z_dim=z_dim, bidirectional=bidir_encoder)
        self.decoder = RNNDecoder(dec_rnn_units, num_feats, num_labels)
        self.fc_z = layers.Dense(6)
        self.fc_hidden = layers.Dense(3*self.dec_rnn_units)

    def __call__(self, x, y):
        batch_size, time_len, feat_dim = x.shape
        z, mu, log_var = self.encoder(x)

        decoder_outputs = []
        # initialize the decoder state with the z vector
        dec_init_state = tf.reshape(self.noise2hidden(z), [3, batch_size, -1])

        init_step = tf.zeros_like(x[:, 0:1, :])
        dec_input = tf.concat([init_step, x[:, :-1]], axis=1)
        if self.z_context:
            z_emb = self.fc_z(z)
            z_with_time = tf.tile(tf.expand_dims(
                z_emb, 1), [1, time_len, 1])
            dec_input = tf.concat([z_with_time, dec_input], axis=2)
        recon_output, _ = self.decoder(dec_input, dec_init_state, y)
        return recon_output, mu, log_var

    def impute(self, x, y, mask):
        batch_size, time_len, feat_dim = x.shape
        z, mu, log_var = self.encoder(x)
        decoder_outputs = []
        last_pred = tf.zeros(shape=(batch_size, 1, feat_dim))
        z_emb = self.fc_z(z)
        last_state = tf.reshape(self.noise2hidden(z), [3, batch_size, -1])
        z_with_time = tf.expand_dims(z_emb, axis=1)
        preds = []
        for step in range(time_len):
            if step == 0 or mask[0, step, 0] == 0:
                step_input = last_pred
            else:
                step_input = x[:, step-1:step, :]
            if self.z_context:
                step_input = tf.concat([z_with_time, step_input], axis=2)

            last_pred, last_state = self.decoder(
                step_input, last_state, y)
            if mask[0, step, 0] == 0:
                preds.append(last_pred)
            else:
                preds.append(x[:, step:step+1, :])

        output = tf.concat(preds, axis=1)
        return output

    def noise2hidden(self, z):
        out = self.fc_hidden(z)
        out = tf.math.tanh(out)
        return out

    def sample(self, labels, z=None, max_len=125):
        """ generates samples conditioned on the given label """
        num_examples = int(labels.shape[0])
        if z is None:
            z = tf.random_normal(shape=(num_examples, self.decoder.rnn_units))
        last_pred = tf.zeros(shape=(num_examples, 1, self.num_feats))
        preds = []
        last_state = tf.reshape(self.noise2hidden(z), [3, num_examples, -1])
        z_emb = self.fc_z(z)
        z_with_time = tf.expand_dims(z_emb, axis=1)
        for _ in range(max_len):
            if self.z_context:
                step_input = tf.concat([z_with_time, last_pred], axis=2)
            else:
                step_input = last_pred
            last_pred, last_state = self.decoder(
                step_input, last_state, labels)
            preds.append(last_pred)
        output = tf.concat(preds, axis=1)
        return output


class CRGANModel(tf.keras.Model):
    def __init__(self, num_feats, num_labels, z_dim, num_units, z_context=True):
        super(CRGANModel, self).__init__()
        self.num_feats = num_feats
        self.num_labels = num_labels
        self.z_dim = z_dim
        self.num_units = num_units
        self.z_context = z_context
        self.num_layers = 3
        self.fc_hidden = layers.Dense(3*self.num_units)
        self.fc_z = layers.Dense(6)
        self.decoder = RNNDecoder(
            self.num_units, self.num_feats, self.num_labels)
        self.start_token = tf.zeros(shape=(1, 1, self.num_feats))

    def __call__(self, x, y, z=None):
        """ uses teacher forcing to predict the next token in sequence """
        batch_size, time_len, feat_dim = x.shape
        # initialize the decoder state with the z vector
        dec_init_state = tf.reshape(self.noise2hidden(z), [3, batch_size, -1])

        dec_input = x
        if self.z_context:
            z_emb = self.fc_z(z)
            z_with_time = tf.tile(tf.expand_dims(
                z_emb, 1), [1, time_len, 1])
            dec_input = tf.concat([z_with_time, dec_input], axis=2)
        recon_output, _ = self.decoder(dec_input, dec_init_state, y)
        return recon_output

    def sample(self, labels, z=None, max_len=125):
        """ generates samples conditioned on the given label """
        num_examples = int(labels.shape[0])
        if z is None:
            z = tf.random_normal(shape=(num_examples, self.z_dims))
        last_pred = tf.zeros(shape=(num_examples, 1, self.num_feats))
        preds = []
        last_state = tf.reshape(self.noise2hidden(z), [3, num_examples, -1])
        z_emb = self.fc_z(z)
        z_with_time = tf.expand_dims(z_emb, axis=1)
        for _ in range(max_len):
            if self.z_context:
                step_input = tf.concat([z_with_time, last_pred], axis=2)
            else:
                step_input = last_pred
            last_pred, last_state = self.decoder(
                step_input, last_state, labels)
            preds.append(last_pred)
        output = tf.concat(preds, axis=1)
        return output

    def noise2hidden(self, z):
        out = self.fc_hidden(z)
        out = tf.math.tanh(out)
        return out
