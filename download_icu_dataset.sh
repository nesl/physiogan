#!/bin/bash
test -d dataset || mkdir dataset
cd dataset
test -d icu || mkdir icu
cd icu
## Training set
wget https://physionet.org/challenge/2012/set-a.zip -O set_a.zip
wget https://physionet.org/challenge/2012/Outcomes-a.txt -O outcome_a.txt
unzip set_a.zip
rm set_a.zip


## Test set
wget https://physionet.org/challenge/2012/set-b.zip -O set_b.zip
wget https://physionet.org/challenge/2012/Outcomes-b.txt -O outcome_b.txt
unzip set_b.zip
rm set_b.zip