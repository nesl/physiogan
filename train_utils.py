"""
Auxiliary methods for model training
"""

import tensorflow as tf
from tensorflow.losses import sparse_softmax_cross_entropy
import matplotlib.pyplot as plt
import tb_utils
# Losses


def mse_loss(y, y_hat):
    loss = tf.reduce_mean(
        tf.reduce_mean(tf.reduce_mean(tf.square(y - y_hat), axis=2),
                       axis=1), axis=0)
    return loss


def mse_train_g_epoch(model, train_data, optim):
    """ Train the generator model for one epoch using mse loss"""
    loss_metric = tf.keras.metrics.Mean()
    for batch_x, batch_y in train_data:
        batch_size = int(batch_x.shape[0])
        train_x = tf.concat(
            (tf.tile(model.start_token, [batch_size, 1, 1]), batch_x[:, :-1, :]), axis=1)
        train_y = batch_x
        noise_z = tf.random.normal(shape=(batch_size, model.z_dim))
        init_hidden = model.noise2hidden(noise_z)
        with tf.GradientTape() as gt:
            batch_preds, _ = model(train_x, batch_y, init_hidden)
            batch_loss = mse_loss(train_y, batch_preds)
            loss_metric.update_state(batch_loss)
        grads = gt.gradient(batch_loss, model.trainable_variables)
        optim.apply_gradients(zip(grads, model.trainable_variables))

    return loss_metric.result()


def train_d_model_batch(g_model, d_model, batch_x, batch_y, d_optim, max_len):
    batch_size = int(batch_x.shape[0])
    with tf.GradientTape() as d_tape:
        sampling_size = max(1, batch_size // (d_model.num_labels))
        cond_labels = tf.random.uniform(
            minval=0, maxval=g_model.num_labels, shape=(sampling_size,), dtype=tf.int32)
        sampling_z = tf.random_normal(shape=(sampling_size, g_model.z_dim))
        samples = g_model.sample(cond_labels, sampling_z,
                                 max_len=max_len)
        d_out_real = d_model(batch_x[:, ::, :])
        d_out_fake = d_model(samples[:, ::, :])
        d_out = tf.concat([d_out_real, d_out_fake], axis=0)
        d_targets = tf.concat([batch_y,
                               d_model.num_labels*tf.ones(shape=(sampling_size,), dtype=tf.int32)], axis=0)
        # d_loss
        d_loss = sparse_softmax_cross_entropy(
            d_targets, d_out) / int(d_targets.shape[0])
        d_preds = tf.argmax(d_out, axis=1)

    d_grads = d_tape.gradient(d_loss, d_model.trainable_variables)
    d_optim.apply_gradients(zip(d_grads, d_model.trainable_variables))
    return d_loss, d_preds, d_targets


def adv_train_d_epoch(g_model, d_model, train_data, d_optim, max_len):
    loss_metric = tf.keras.metrics.Mean()
    d_accuracy_metric = tf.keras.metrics.Accuracy()
    for batch_x, batch_y in train_data:
        batch_size = int(batch_x.shape[0])

        d_loss, d_preds, d_targets = train_d_model_batch(
            g_model, d_model, batch_x, batch_y, d_optim, max_len)

        loss_metric.update_state(d_loss)
        d_accuracy_metric.update_state(d_targets, d_preds)

    return loss_metric.result(), d_accuracy_metric.result()


def adv_train_epoch(g_model, d_model, train_data, g_optim, d_optim, max_len=128):
    g_loss_metric = tf.keras.metrics.Mean()
    d_accuracy_metric = tf.keras.metrics.Accuracy()
    for batch_x, batch_y in train_data:
        batch_size = int(batch_x.shape[0])
        batch_size = int(batch_x.shape[0])
        train_x = tf.concat(
            (tf.tile(g_model.start_token, [batch_size, 1, 1]), batch_x[:, :-1, :]), axis=1)
        train_y = batch_x
        # sampling size of "fake" class, at least 1 or poroportional to the size of each other classe samples.
        sampling_size = max(1, batch_size // (d_model.num_labels))

        d_batch_loss, d_batch_preds, d_batch_targets = train_d_model_batch(
            g_model, d_model, batch_x, batch_y, d_optim, max_len)
        d_accuracy_metric.update_state(d_batch_targets, d_batch_preds)
        with tf.GradientTape() as g_tape:

            # Train generator
            # Reconsturction error
            batch_noise = tf.random_normal(shape=(batch_size, g_model.z_dim))
            batch_hidden = g_model.noise2hidden(batch_noise)
            batch_preds, _ = g_model(train_x, batch_y, batch_hidden)
            recon_loss = mse_loss(train_y, batch_preds)

            cond_labels = tf.random.uniform(
                minval=0, maxval=g_model.num_labels, shape=(sampling_size,), dtype=tf.int32)
            sampling_z = tf.random_normal(shape=(sampling_size, g_model.z_dim))
            samples = g_model.sample(cond_labels, sampling_z,
                                     max_len=max_len)
            d_out_fake = d_model(samples[:, ::, :])
            # g_loss
            g_adv_loss = sparse_softmax_cross_entropy(
                cond_labels, d_out_fake) / batch_size
            g_recon_loss = recon_loss
            g_batch_loss = g_adv_loss + 100 * g_recon_loss

        g_loss_metric.update_state(g_batch_loss)
        g_grads = g_tape.gradient(g_batch_loss, g_model.trainable_variables)
        g_clipped_grads, _ = tf.clip_by_global_norm(g_grads, 1.0)
        g_optim.apply_gradients(
            zip(g_clipped_grads, g_model.trainable_variables))

    return g_loss_metric.result(), d_accuracy_metric.result()


# utility function to evalute conditional label of samples using auxiliary classifier
def evaluate_samples(g_model, aux_model, max_len):
    accuracy_metric = tf.keras.metrics.Accuracy()
    num_eval_batches = 1
    batch_size = 1024
    for _ in range(num_eval_batches):
        cond_labels = tf.random.uniform(
            minval=0, maxval=g_model.num_labels, shape=(batch_size,), dtype=tf.int32)
        # cond_labels = tf.squeeze(cond_labels)
        sampling_z = tf.random_normal(shape=(batch_size, g_model.z_dim))
        samples = g_model.sample(cond_labels, sampling_z,
                                 max_len=max_len)
        model_preds = tf.argmax(aux_model(samples), axis=1)
        accuracy_metric.update_state(cond_labels, model_preds)
    return accuracy_metric.result()


def gen_plot(samples, num_labels):
    sampling_size = samples.shape[0]
    num_cols = num_labels
    num_rows = sampling_size // num_cols
    fig = plt.figure(figsize=(4*num_cols, num_rows*4))
    num_samples = sampling_size
    for i in range(num_samples):
        plt.subplot(num_rows, num_cols, i+1)
        plt.plot(samples[i])
    image = tb_utils.fig_to_image_tensor(fig)
    return image