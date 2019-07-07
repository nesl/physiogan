import numpy as np


def subsample(signal, stride):
    win_size = 2*stride+1
    conv_filter = np.ones(shape=(2*stride+1,))/win_size
    if len(np.shape(signal)) == 2:
        output = np.stack([np.convolve(signal[:, i], conv_filter, mode='same')
                           for i in range(np.shape(signal)[1])], axis=1)
    else:
        output = np.convolve(signal, conv_filter, mode='same')
    return output[::stride]
