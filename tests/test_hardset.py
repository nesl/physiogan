""" unit tests for the HAR dataset """

import pytest
import numpy as np


import har_dataset


@pytest.fixture
def har_trainset():
    dataset = har_dataset.HARDataset('./dataset/har/', is_train=True)
    return dataset


@pytest.fixture
def har_testset():
    dataset = har_dataset.HARDataset('./dataset/har', is_train=False)
    return dataset


test_data = [
    (har_trainset, 7352),
    (har_testset, 2947)
]


def test_har_trainset(har_trainset):
    assert har_trainset.data.dtype == np.float32
    assert har_trainset.data.shape == (7352, 128, 6)
    assert har_trainset.labels.shape == (7352,)
    assert har_trainset.labels.dtype == np.int32
    assert np.min(har_trainset.labels) == 0
    assert np.max(har_trainset.labels) == 5


def test_har_testset(har_testset):
    assert har_testset.data.dtype == np.float32
    assert har_testset.data.shape == (2947, 128, 6)
    assert har_testset.labels.shape == (2947,)
    assert har_testset.labels.dtype == np.int32
    assert np.min(har_testset.labels) == 0
    assert np.max(har_testset.labels) == 5
