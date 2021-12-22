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


