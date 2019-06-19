import argparse
import os
import numpy as np


def load_dset(dset_path):
    x_fpath = os.path.join(dset_path, 'samples_x.npy')
    y_fpath = os.path.join(dset_path, 'samples_y.npy')

    x_data = np.load(x_fpath)
    y_data = np.load(y_fpath)

    return x_data, y_data


def combine_dsets(x_data_list, y_data_list):
    x_data = np.concatenate(x_data_list, axis=0)
    y_data = np.concatenate(y_data_list, axis=0)

    rand_idces = np.random.permutation(np.arange(x_data.shape[0]))[
        :x_data_list[0].shape[0]]
    print(rand_idces.shape)
    x_data = x_data[rand_idces]
    y_data = y_data[rand_idces]
    return x_data, y_data


def save_dset(x_data, y_data, save_path):
    if not os.path.exists(save_path):
        os.mkdir(save_path)

    x_fpath = os.path.join(save_path, 'samples_x.npy')
    y_fpath = os.path.join(save_path, 'samples_y.npy')
    np.save(x_fpath, x_data)
    np.save(y_fpath, y_data)

    print('Data saved to {}'.format(save_path))


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Combine datasets')
    parser.add_argument('--inputs', type=str, action='store',
                        nargs='+', required=True)
    parser.add_argument('--out', type=str, required=True)

    args = parser.parse_args()
    print(args)
    all_data = [load_dset(x) for x in args.inputs]
    all_data_x, all_data_y = zip(*all_data)
    x_data_combined, y_data_combined = combine_dsets(all_data_x, all_data_y)
    print(x_data_combined.shape, y_data_combined.shape)
    save_dset(x_data_combined, y_data_combined, args.out)
