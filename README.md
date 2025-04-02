## Instructions for replication

1. Install dependencies
```
python3 -m venv venv
source venv/bin/activate/
pip3 install numpy pandas scipy tensorflow sklearn matplotlib keras_tuner seaborn
```

2. Download datasets
```
data/AudioSet/ # https://www.kaggle.com/datasets/zfturbo/audioset
data/UrbanSound8K/ # https://urbansounddataset.weebly.com/
```

3. Preprocess datasets
```
python3 preprocess_audioset.py
python3 preprocess_urbansound8k.py
```

4. Train models
```
python3 train_1dfno.py
python3 train_2dfno.py
python3 train_transformer_unsupervised.py
python3 train_transformer.py
python3 train_transformer_random_init.py
python3 train_transformer_gridsearch.py
python3 train_transformer_kfold.py
```

5. Plot results
```
mv *.pkl jobs/
cd jobs/
python3 plot_lr_schedule.py
python3 plot_results.py # reads jobs/results.csv
python3 plot_gridsearch.py
python3 plot_kfold.py
```
