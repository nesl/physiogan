#!/bin/bash
test -d dataset || mkdir dataset
test -d dataset && cd dataset
test -d har || mkdir har
cd har
wget -O dataset_names.txt https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.names
wget -O uci_har.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00240/UCI%20HAR%20Dataset.zip
unzip uci_har.zip && rm -f uci_har.zip
cd ..
cd ..