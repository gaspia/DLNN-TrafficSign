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

# Some definitions


## Max Pooling

Max Pooling is a pooling operation that calculates the maximum value for patches of a feature map, and uses it to create a downsampled (pooled) feature map. It is usually used after a convolutional layer.
## Drop-Out
As a neural network learns, neuron weights settle into their context within the network. Weights of neurons are tuned for specific features providing some specialization. Neighboring neurons become to rely on this specialization, which if taken too far can result in a fragile model too specialized to the training data. This reliant on context for a neuron during training is referred to complex co-adaptations.

You can imagine that if neurons are randomly dropped out of the network during training, that other neurons will have to step in and handle the representation required to make predictions for the missing neurons. This is believed to result in multiple independent internal representations being learned by the network.

Dropout is easily implemented by randomly selecting nodes to be dropped-out with a given probability (e.g. 20%) each weight update cycle. This is how Dropout is implemented in Keras. Dropout is only used during the training of a model and is not used when evaluating the skill of the model.

Dropout can be applied to input neurons called the visible layer.

Additionally, as recommended in the original paper on Dropout, a constraint is imposed on the weights for each hidden layer, ensuring that the maximum norm of the weights does not exceed a value of 3. This is done by setting the kernel_constraint argument on the Dense class when constructing the layers.

### Max Norm Constraint

Maxnorm(m) will, if the L2-Norm of your weights exceeds m, scale your whole weight matrix by a factor that reduces the norm to m. As you can find in the keras code in class MaxNorm(Constraint)

Aditionally, maxnorm has an axis argument, along which the norm is calculated. In your example you don't specify an axis, thus the norm is calculated over the whole weight matrix. If for example, you want to constrain the norm of every convolutional filter, assuming that you are using tf dimension ordering, the weight matrix will have the shape (rows, cols, input_depth, output_depth). Calculating the norm over axis = [0, 1, 2] will constrain each filter to the given norm.

Constraining the weight matrix directly is another kind of regularization. If you use a simple L2 regularization term you penalize high weights with your loss function. With this constraint, you regularize directly. As also linked in the keras code, this seems to work especially well in combination with a dropoutlayer.
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

The training deep neural networks with tens of layers is challenging as they can be sensitive to the initial random weights and configuration of the learning algorithm.

One possible reason for this difficulty is the distribution of the inputs to layers deep in the network may change after each mini-batch when the weights are updated. This can cause the learning algorithm to forever chase a moving target. This change in the distribution of inputs to layers in the network is referred to the technical name “internal covariate shift.”

Bactch Normalization standardizes the inputs to a layer for each mini-batch. This has the effect of stabilizing the learning process and dramatically reducing the number of training epochs required to train deep networks.

It does this scaling the output of the layer, specifically by standardizing the activations of each input variable per mini-batch, such as the activations of a node from the previous layer. Recall that standardization refers to rescaling data to have a mean of zero and a standard deviation of one, e.g. a standard Gaussian.

# Imbalanced Dataset Strategies

## Class Weights

Optional dictionary mapping class indices (integers) to a weight (float) value, used for weighting the loss function (during training only). This can be useful to tell the model to "pay more attention" to samples from an under-represented class.

Influencing the loss function by assigning relatively higher costs to examples from minority classes. We use the re-weighting method from scikit-learn library to estimate class weights for unbalanced dataset with ‘balanced’ as a parameter which the class weights will be given by n_samples / (n_classes * np.bincount(y)).

# Learning Rate

Should we use Learning Rate Scheduling?

It depends. ADAM updates any parameter with an individual learning rate. This means that every parameter in the network has a specific learning rate associated.
But the single learning rate for each parameter is computed using lambda (the initial learning rate) as an upper limit. This means that every single learning rate can vary from 0 (no update) to lambda (maximum update).

It's true, that the learning rates adapt themselves during training steps, but if you want to be sure that every update step doesn't exceed lambda you can than lower lambda using exponential decay or whatever. It can help to reduce loss during the latest step of training, when the computed loss with the previously associated lambda parameter has stopped to decrease.

We started with values of 0.01 and 0.001. 

We then should test with learning rate scheduler to adjust the learning rate according to the validation loss.

The learning rate scheduler makes the learning rate of optimizer adapt in a particular situation during the training phase. A learning rate scheduler relies on changes in loss function value to dictate whether the learning rate has decayed or not.

# Momentum

This algorithm is used to accelerate the gradient descent algorithm by taking into consideration the 'exponentially weighted average' of the gradients. Using averages makes the algorithm converge towards the minima in a faster pace.
# Steps

Number of batches used to train a model for each epoch. For fixed dataset sizes, this is equal to the number of samples in the dataset divided by the batch size.
For variable dataset sizes, this can vary, but a rule of thumb recommendation is to have have at least 2 times the number of samples divided by the batch size.

# Validation Steps