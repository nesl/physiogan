#!/bin/bash
test -d dataset | mkdir dataset
cd dataset
test -d ecg2lead | mkdir ecg2lead
cd ecg2lead
wget http://www.timeseriesclassification.com/Downloads/TwoLeadECG.zip
unzip TwoLeadECG.zip
rm -f TwoLeadECG.zip
cd ..

