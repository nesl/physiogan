# feature extraction based on http://qurinet.ucdavis.edu/pubs/conf/li-mobisys15.pdf
import numpy as np
from numpy import mean, median, std, percentile, fft, abs, argmax
from scipy.stats import skew, kurtosis
from adl_dataset import ADLDataset


def Min(data):
    """Returns the minimum value of a time series"""
    return data.min(axis=1)


def Max(data):
    """Returns the maximum value of a time series"""
    return data.max(axis=1)


def Median(data):
    """Returns the median of a time series"""
    return median(data, axis=1)


def Std(data):
    """Returns the standard deviation a time series"""
    return data.std(axis=1)


def Range(data):
    """ Returns the range """
    return data.max(axis=1) - data.min(axis=1)


def CV(data):
    """" ratio of standard deviation and mean times 100. measure of signal dispersion """
    data_mean = np.mean(data, axis=1)
    data_std = np.std(data, axis=1)
    cv = (data_std / data_std) * 100
    return cv


def Skew(data):
    return skew(data, axis=1)


def Kurtosis(data):
    return kurtosis(data, axis=1)


def QR1(data):
    return percentile(data, 25, axis=1)


def QR3(data):
    return percentile(data, 75, axis=1)


def IQR(data):
    """Returns the interquartile range a time series"""
    return percentile(data, 75, axis=1) - percentile(data, 25, axis=1)


def MCR(data):
    """ mean crossing rate """
    data_mean = np.average(data, axis=1)
    mcr = np.mean(np.equal(np.expand_dims(data_mean, 1), data), axis=1)
    return mcr


def AbsArea(data):
    return np.mean(np.abs(data), axis=1)


def TotalArea(data):
    total_area = np.mean(np.sum(np.abs(data), axis=2), axis=1)[:, np.newaxis]
    return total_area


def RMS(data):
    """returns the RMS of a time series"""
    data = np.power(data, 2)
    data = np.mean(data, axis=1)
    data = np.sqrt(data)
    return data


def DominantFrequencyRatio(data):
    """Returns the ratio of highest magnitude FFT coefficient to sum of magnitue of all FFt coeffients """
    w = fft.fft(data, axis=1)
    w_abs = abs(w)
    return (np.max(w_abs, axis=1) / np.sum(w_abs, axis=1))


def FFTEntropy(data):
    """Returns the ratio of highest magnitude FFT coefficient to sum of magnitue of all FFt coeffients """
    w = fft.fft(data, axis=1)
    w_abs = abs(w)
    w_abs = w_abs / np.expand_dims(np.sum(w_abs, axis=1), axis=1)
    w_abs_log = np.log(w_abs)

    entropy = -1*np.sum(w_abs * w_abs_log, axis=1)
    return (np.max(w_abs, axis=1) / np.sum(w_abs, axis=1))


def get_all_feats(data):
    feat_funcs = [Min, Max, Median, Range,
                  CV, Skew, Kurtosis, QR1, QR3, IQR, MCR, AbsArea, TotalArea, RMS, DominantFrequencyRatio, FFTEntropy]
    feat_vals = [fun(data) for fun in feat_funcs]
    all_feat_vals = np.concatenate(feat_vals, axis=1)
    return all_feat_vals


if __name__ == '__main__':
    adl_dataset = ADLDataset('dataset/adl', is_train=True)
    data, labels = adl_dataset.data, adl_dataset.labels
    all_feats = get_all_feats(data)
    print(all_feats.shape, labels.shape)
