##Synthetic Bio-sensor data generation



Install requirements

```
pip install -r requirements.txt
```

Download datasets

```
bash download_adl_dataset.sh
bash download_har_dataset.sh
```

Train classifiers


```
python classification_model.py --dataset=xxx --num_epochs=200
```

where xxx is one of [har, adl]