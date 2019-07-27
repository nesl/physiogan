## Synthetic Sensor Data Generation



### Install requirements

```
pip install -r requirements.txt
```

### Download datasets

```
bash download_har_dataset.sh
bash download_ecg_dataset.sh
```

### Train classifiers


```
python classification_model.py --dataset=xxx --num_epochs=200
```

where xxx is one of [har, adl]


Note the dir of the classification model (from tensorboard) because it will be used later as an auxiliary model to evaluate the synthetic samples.



### Train the generative model:

Use the following command to train generative model:

```
 python crnn_model.py --model_type=rvae --num_epochs=15000 --dataset=adl \
  --aux_restore=AUXILIARY_MODEL_CHECKPOINT_DIR --mle_epochs=0 --batch_size=1024 \
   --num_units=128 --z_dim=16  --bidir_encoder=True --z_context=True
 ```

 ### Produce Synthetic samples

 Use same command as the training command above, abut add extra two flags `--sample --restore=xxx`

```
 python crnn_model.py --model_type=rvae --num_epochs=15000 --dataset=adl \
  --aux_restore=AUXILIARY_MODEL_CHECKPOINT_DIR --mle_epochs=0 --batch_size=1024 \
   --num_units=128 --z_dim=16  --bidir_encoder=True --z_context=True \
   --sample --restore=GENERATIVE_MODEL_CHECKPOINT_DIR
 ```

Make a note of the printed message that tells you where are the samples going to be saved.

### Train classification model on Synthetic samples

```
python classification_model.py --dataset=xxx --num_epochs=200 --train_syn=SAMPLES_DIR
```
