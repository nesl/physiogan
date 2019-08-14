import numpy as np
import pickle
import tensorflow as tf
import matplotlib.pyplot as plt
tf.enable_eager_execution()


def get_testset(test_dataset, test_size=200, mask_type='mar', mask_prob = 0.5):
    test_x, test_y = next(iter(test_dataset.shuffle(1000).batch(test_size)))
    test_masks = []
    test_originals = []
    test_inputs = []
    test_labels = []
    for i in range(test_x.shape[0]):
        rand_test_idx = i# np.random.choice(test_x.shape[0])
        test_sample = test_x[rand_test_idx]
        test_sample_np = test_sample.numpy()
        seq_len = test_sample_np.shape[0]
        if mask_type=='mar':
            mask = np.ones_like(test_sample)
            #drop_range = np.where(np.random.uniform(size=(test_sample_np.shape))<mask_prob)
            drop_range = np.random.choice(np.arange(5,seq_len),size=int(seq_len*mask_prob),replace=False)
            mask[drop_range,:] = np.nan
        
        elif mask_type == 'segment':
            mask_len = int(mask_prob * test_sample_np.shape[0])
            rand_pos = int(np.random.uniform(low=0, high=test_sample_np.shape[0]-mask_len))
            mask = np.ones_like(test_sample_np)
            drop_range = np.arange(rand_pos, rand_pos+mask_len)
            mask[drop_range,:]= np.nan
        else:
            raise ValueError('Unsupported mask type')
        test_masks.append(np.expand_dims(mask,0))
        test_sample_masked = np.multiply(mask, test_sample)
        test_inputs.append(test_sample_masked)
        test_originals.append(test_sample)
        test_labels .append(test_y[rand_test_idx:rand_test_idx+1])

    test_labels = np.array(test_labels).ravel()
    test_originals = np.array(test_originals)
    test_masks = np.array(test_masks)[:,0,:,:]
    test_inputs = np.array(test_inputs)
    return (test_originals, test_labels, test_masks, test_inputs)

def plot_results(x, x_complete, masks, results, mask_type, num_examples =3, draw_alpha=0.5):
    methods = results.keys()
    fig, axes = plt.subplots(len(methods)+2, num_examples, figsize=(24,6))
    test_idxs = np.random.choice(len(x_complete),size=3,replace=False)
    for idx in range(num_examples):
        test_idx = test_idxs[idx]
        test_mask = masks[test_idx]
        test_x_masked = x[test_idx]
        test_orig = x_complete[test_idx]
    
        drop_range = np.where(test_mask[:,0] != 1)
        axes[0][idx].plot(test_orig[:,0])
        axes[0][idx].set_title('Original signal', fontsize=22)
        axes[0][idx].set_ylim(-0.25, 0.35)
        if mask_type == 'segment':
            axes[0][idx].axvspan(np.min(drop_range), np.max(drop_range), color='green', alpha=draw_alpha)
            axes[1][idx].axvspan(np.min(drop_range), np.max(drop_range), color='black', alpha=draw_alpha)
            axes[2][idx].axvspan(np.min(drop_range), np.max(drop_range), color='red', alpha=draw_alpha)
            axes[3][idx].axvspan(np.min(drop_range), np.max(drop_range), color='red', alpha=draw_alpha)
        else:
            for j in range(test_mask.shape[0]):
                if np.isnan(test_mask[j,0]):
                    axes[0][idx].axvspan(j,j+1, color='green', alpha=draw_alpha)
                    axes[1][idx].axvspan(j,j+1, color='black', alpha=draw_alpha)
                    axes[2][idx].axvspan(j,j+1, color='red', alpha=draw_alpha)
                    axes[3][idx].axvspan(j,j+1, color='red', alpha=draw_alpha)

        axes[1][idx].set_title('Input Signal',fontsize=22)
        axes[1][idx].plot(test_x_masked[:,0])
        axes[1][idx].set_ylim(-0.25, 0.35)

        
        for i,m in enumerate(methods):
            axes[i+2][idx].plot(results[m][test_idx,:,0])
            axes[i+2][idx].set_title('{} Imputation'.format(m), fontsize=22)
            axes[i+2][idx].set_ylim(-0.25, 0.35)
    
            if mask_type == 'segment':
                axes[i+2][idx].axvspan(np.min(drop_range), np.max(drop_range), color='red', alpha=draw_alpha)
            else:
                for j in range(test_mask.shape[0]):
                    if np.isnan(test_mask[j,0]):
                        axes[i+2][idx].axvspan(j,j+1, color='red', alpha=draw_alpha)
    plt.tight_layout()

    return fig



def get_data_from_pickle(path):
    with open(path, 'rb') as fh:
        test_x,test_y, test_masks, test_masked = pickle.load(fh)
    return test_x,test_y, test_masks, test_masked



def evaluate_accuracy(aux_model, data_x, data_y):
    preds = tf.arg_max(aux_model(data_x),dimension=1)
    accuracy_metric = tf.keras.metrics.Accuracy()
    accuracy_metric.update_state(data_y, preds)
    return accuracy_metric.result().numpy()

def compute_mae(orig, imputed, mask):
    mae = np.sum(np.abs(orig-imputed)) / np.sum(np.nan_to_num(mask)==0)
    return mae

def evaluate_semantic(aux_model, data_orig, data_y, data_masked, data_completed, verbose=False):
    accuracy_orig = evaluate_accuracy(aux_model, data_orig, data_y)
    accuracy_masked = evaluate_accuracy(aux_model, data_masked, data_y)
    accuracy_completed = evaluate_accuracy(aux_model, data_completed, data_y)
    
    score = (accuracy_completed - accuracy_masked) /( accuracy_orig - accuracy_masked)
    if verbose:
        print('\t orig: {:.2f}, masked: {:.2f}, completed: {:.2f}'.format(accuracy_orig, accuracy_masked, accuracy_completed))
    return score

def evaluate_mae(data_orig, data_completed, masks):
    mae_list = []
    for x, x_hat, m in zip(data_orig, data_completed, masks):
        mae_list.append(compute_mae(x, x_hat, m))
    return np.mean(mae_list)