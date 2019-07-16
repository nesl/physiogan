#!/bin/bash
test -d dataset | mkdir dataset
cd dataset
test -d ecg200 | mkdir ecg200
cd ecg200
wget http://www.timeseriesclassification.com/Downloads/ECG200.zip
unzip ECG200.zip
rm -f ECG200.zip
cd ..

