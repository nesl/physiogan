##Synthetic Bio-sensor data generation

###HAR experiment

####Classification model
Train the classification model
```
 python har_model.py
```

Evaluate classification model
```
 python har_model.py --evaluate --restore=xxx/xxx
```

To train on synthetic dataset

```
python har_model.py --train_syn=PATH_of_synthetic_dataset
```

To use the model to verify the matching rate of labels of synthetic damples

```
 python har_model.py --evaluate_syn=PATH_of_synthetic_dataset --restore=xxx/xxx

```


###Generative model (Conditional-RNN)

Training the model

```
 python crnn_model.py 
 ```

Generating samples from it
```
 python crnn_model.py --sample --restore=MODEL_CHECKPOINT_PATH
 ```

