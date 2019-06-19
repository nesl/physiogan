#!/bin/bash
test -d dataset || mkdir dataset
test -d dataset && cd dataset
test -d cinc || mkdir cinc
cd cinc
test -d training || mkdir training
cd training
wget -O reference.csv https://physionet.org/challenge/2017/REFERENCE-v3.csv
wget -O training2017.zip https://physionet.org/challenge/2017/training2017.zip
unzip training2017.zip && rm -f training2017.zip
cd ..
test -d validation || mkdir validation 
cd validation
wget -O sample2017.zip https://physionet.org/challenge/2017/sample2017.zip
unzip sample2017.zip && rm -f sample2017.zip
cd ..
cd ..