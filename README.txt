### Synthetic Bio-sensor data generation

##### HAR experiment

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
python har_model.py --syn_train=PATH_of_synthetic_dataset
```

##### Train the generative model

