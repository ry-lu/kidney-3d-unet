# 3D Unet Baseline for Segmenting Vasculature in 3D Scans of Human Kidney
A Pytorch baseline 3D Unet model to segment blood vessels in 3D scans of kidneys. Trained on data from the SenNet + HOA Kaggle competition.

Competition Link: [link](https://www.kaggle.com/competitions/blood-vessel-segmentation)

## Setup
Python 3.12.2

```console
> pip install -r requirements.txt
```
# Dataset
Dataset was obtained from here [link](https://www.kaggle.com/competitions/blood-vessel-segmentation).
For data access, please accept the organizer's TOS and download. 

Dataset was converted to npz format with 
```console
> python convert_to_npz.py -h
usage: convert_to_npz.py [-h] [--data_dir DATA_DIR]

Convert data to npz format.

options:
  -h, --help            show this help message and exit
  --data_dir DATA_DIR, -D DATA_DIR
                        where to load and save data
```

## Training
See notebook for training info. Notebook was run in a Kaggle Environment.

To run locally, configure data in right paths and run train.py.
```console
> python train.py -h
usage: train.py [-h] [--config CONFIG] [--data_dir DATA_DIR]

Train model on 3D kidney.

options:
  -h, --help            show this help message and exit
  --config CONFIG, -C CONFIG
                        yaml file path for model config
  --data_dir DATA_DIR, -D DATA_DIR
                        where to load and save data
```
