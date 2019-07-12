import numpy as np

from collections import namedtuple

from har_dataset import HARDataset
from adl_dataset import ADLDataset
from dummy_dataset import DummyDataset
from cinc_dataset import CINCDataset
from ecg2lead_dataset import ECG2LeadDataset

Metadata = namedtuple(
    'Point', ['num_feats', 'num_labels', 'classes', 'max_len', 'num_examples'], verbose=False)


class DataFactory:
    @classmethod
    def create_dataset(cls, dset_name):
        """ returns
            tuple of (train_data, test_data, metadata) """
        if dset_name == 'har':
            dset_root = 'dataset/har'
            dset_class = HARDataset
        elif dset_name == 'adl':
            dset_root = 'dataset/adl'
            dset_class = ADLDataset
        elif dset_name == 'cinc':
            dset_root = 'dataset/cinc'
            dset_class = CINCDataset
        elif dset_name == 'ecg2lead':
            dset_root = 'dataset/ecg2lead'
            dset_class = ECG2LeadDataset
        elif dset_name == 'dummy':
            dset_root = None
            dset_class = DummyDataset
        else:
            raise Exception("Invalid dataset requested")

        train_data = dset_class(dset_root, is_train=True, mini=True)
        test_data = dset_class(dset_root, is_train=False, mini=True)

        dset_meta = Metadata(train_data.num_feats,
                             train_data.num_labels, train_data.classes, train_data.max_len, train_data.labels.shape[0])
        return train_data.to_dataset(), test_data.to_dataset(), dset_meta
