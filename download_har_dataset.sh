#!/bin/bash
test -d dataset || mkdir dataset
test -d dataset && cd dataset
mkdir har
cd har
wget -O HAR_dataset.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip HAR_dataset.zip && rm -f HAR_dataset.zip
cd ..
cd ..
