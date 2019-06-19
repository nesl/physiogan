#!/bin/bash
test -d dataset || mkdir dataset
test -d dataset && cd dataset
test -d adl || mkdir adl
cd adl
wget -O ADL_Dataset.zip https://archive.ics.uci.edu/ml/machine-learning-databases/00283/ADL_Dataset.zip
unzip ADL_Dataset.zip && rm -f ADL_Dataset.zip
mv HMP_Dataset/* .
cd ..
cd ..
