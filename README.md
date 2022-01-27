# DLNN-TrafficSign

Traffice Sign recognition

## Post-Graduation work in Deep Learning Neural Networks

# Intro

Based on the Kaggle dataset: <https://www.kaggle.com/valentynsichkar/traffic-signs-preprocessed>

# Setup

### Generate an api token in Kaggle

1. Login to Kaggle
2. Go to your profile page
3. select "account"
4. in the API section click on "Create New API Token"
5. Download the kaggle.json file and save it in the .kaggle folder on your home folder

### Install the Kaggle cli

```bash
pip install kaggle
```

### Download the dataset

```bash
kaggle datasets download -d valentynsichkar/traffic-signs-preprocessed
```

### Expand the dataset zip
```bash
unzip traffic-signs-preprocessed.zip -d ./dataset
```
# VGGNet

A CNN can be considered VGG-net like if:
- it makes use of only 3×3 filters, regardless of network depth.
- There are multiple CONV => RELU layers applied before a single POOL operation, sometimes with more CONV => RELU layers stacked on top of each other as the network increases in depth.

## Main Characteristics

- The use of these small kernels is arguably what helps VGGNet generalize to classification problems outside where the network was originally trained.

1. All CONV layers in the network using only 3×3 filters.
2. Stacking multiple CONV => RELU layer sets (where the number of consecutive CONV => RELU layers normally increases the deeper we go) before applying a POOL operation.
3. 

## Bactch Normalization



# Imbalanced Dataset Strategies

## Class Weights

Influencing the loss function by assigning relatively higher costs to examples from minority classes. We use the re-weighting method from scikit-learn library to estimate class weights for unbalanced dataset with ‘balanced’ as a parameter which the class weights will be given by n_samples / (n_classes * np.bincount(y)).

# Learning Rate

We started with values of 0.01 and 0.001. We then used the learning rate scheduler to adjust the learning rate according to the validation loss.

The learning rate scheduler makes the learning rate of optimizer adapt in a particular situation during the training phase. A learning rate scheduler relies on changes in loss function value to dictate whether the learning rate has decayed or not.

# Steps

# Validation Steps