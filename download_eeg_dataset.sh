#!/bin/bash
# Downloads and extracts the EEG database : http://archive.ics.uci.edu/ml/datasets/EEG+Database
test -d dataset || mkdir dataset
cd dataset
test -d eeg_database || mkdir eeg_database
cd eeg_database
test -f eeg_full.tar || wget -O eeg_full.tar http://archive.ics.uci.edu/ml/machine-learning-databases/eeg-mld/eeg_full.tar
test -f eeg_full.tar && tar -xvf eeg_full.tar
test -f eeg_full.tar && rm -f eeg_full.tar
tar_gz_files=`ls *.tar.gz`
echo $tar_gz_files
for file in $tar_gz_files; do
    tar -xvf $file && rm -f $file
done
cd ..
