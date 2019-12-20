This repository contains some of the main EDM models for Knowledge Tracing.

For now it contains DAS3H model by Choffin et al., a basic FeedForwardModel, DKT and SAKT

## Setup



Create a new conda environment with python 3
```
conda create --name python3-env python=3.7
```
Activate conda env
```
conda activate python3-env
```

Install [PyTorch](https://pytorch.org) and the remaining requirements:

```
pip install -r requirements.txt
```

To use a dataset, download the data from one of the links above and:
- place the main file under `data/<dataset codename>/data.csv` for an ASSISTments dataset
- place the main file under `data/<dataset codename>/data.txt` for a KDDCup dataset

```
python prepare_data.py --dataset <dataset codename> --remove_nan_skills
```

## Training

#### Deep Knowledge Tracing

To train a DKT model:

```
python train_dkt.py --dataset <dataset codename> 
```

#### Self-Attentive Knowledge Tracing

To train a SAKT model:

```
python train_sakt.py --dataset <dataset codename>
```
