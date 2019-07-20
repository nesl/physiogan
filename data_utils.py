import numpy as np

from collections import namedtuple

from har_dataset import HARDataset
from adl_dataset import ADLDataset
from dummy_dataset import DummyDataset
from cinc_dataset import CINCDataset
from ecg2lead_dataset import ECG2LeadDataset
from ecg200_dataset import ECG200Dataset
from ecg_dataset import ECGDataset
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
            mini = True
        elif dset_name == 'adl':
            dset_root = 'dataset/adl'
            dset_class = ADLDataset
            mini = True
        elif dset_name == 'adl_full':
            dset_root = 'dataset/adl'
            dset_class = ADLDataset
            mini = False 
        elif dset_name == 'cinc':
            dset_root = 'dataset/cinc'
            dset_class = CINCDataset
            mini = False
        elif dset_name == 'ecg2lead':
            dset_root = 'dataset/ecg2lead'
            dset_class = ECG2LeadDataset
            mini = False
        elif dset_name == 'ecg200':
            dset_root = 'dataset/ecg200'
            dset_class = ECG200Dataset
            mini = False
        elif dset_name == 'ecg':
            dset_root = 'dataset/ecg_data'
            dset_class = ECGDataset
            mini = True
        elif dset_name == 'ecg_full':
            dset_root = 'dataset/ecg_data'
            dset_class = ECGDataset
            mini = False          
        elif dset_name == 'dummy':
            dset_root = None
            dset_class = DummyDataset
            mini = False
        else:
            raise Exception("Invalid dataset requested", dset_name)

        train_data = dset_class(dset_root, is_train=True, mini=mini)
        test_data = dset_class(dset_root, is_train=False, mini=mini)

        dset_meta = Metadata(train_data.num_feats,
                             train_data.num_labels, train_data.classes, train_data.max_len, train_data.labels.shape[0])
        return train_data.to_dataset(), test_data.to_dataset(), dset_meta
