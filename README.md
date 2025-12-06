# modular-neural-network-from-scratch
## Project Description

This repository contains a fully modular neural network implemented from scratch in NumPy, trained on the Pen-Based Recognition of Handwritten Digits dataset (UCI ML Repository).
The goal of this project is to demonstrate a deep understanding of:
- Forward propagation
- Backpropagation and gradient derivation
- Multi-layer neural networks
- Softmax + Cross-Entropy for multi-class classification
- Numerical stability & weight initialization
- Training loops, accuracy computation, normalization
- Building a customizable and extensible neural network framework
No deep learning library (PyTorch, TensorFlow…) was used — only NumPy.

## Key Features
### Modular Architecture
You can freely define:
- Number of layers
- Number of neurons per layer
- Activation functions per layer
- Learning rate & training duration

### Forward & Backward Propagation Implemented Manually
Vectorized math only (NumPy)
Supports any activation function that defines:
- Forward_prop()
- Backward_prop()
Only ReLU, Sigmoid and Softmax have been implmented but more could be easly implemented.

### Multi-Class Classification
- Softmax output layer
- One-hot encoding
- Cross-entropy loss (numerically stable)

### Training Pipeline
- Dataset preprocessing
- Feature scaling
- Train/validation split
- Gradient descent optimizer
- Loss tracking
- Prediction & accuracy evaluation

## Model Architecture (Example)
- Input (16 features)
-       ↓
- Linear (16 → 32)
-       ↓
- ReLU
-       ↓
- Linear (64 → 16)
-       ↓
- ReLU
-       ↓
- Linear (16 → 10)
-       ↓
- Softmax
-       ↓
- Cross-Entropy Loss

## Loss Curve
![Loss Curve](figures/loss_NN.png)

## Required Package
- Numpy
- Matplotlib
- Pandas

## Usage Example
from src.neural_net import Neural_Net, ReLU, Sigmoid
from utils import scale_features, split_data

# Example: build a 3-layer network
layers_dim = [16, 48, 32, 10]
activations = [Sigmoid(), ReLU(), ReLU()]

model = Neural_Net(layers_dim, activations)

losses = model.train(X_train, y_train, lr=0.003, n_iters=1300)
preds = model.predict(X_test)

print("Test accuracy:", accuracy(y_test, preds))


## What I learned
- How to generalize gradient computation for any number of layers
- A much deeper understanding of backpropagation
- Debugging vanishing gradients, exploding loss, NaN instability
- Hyperparameter tuning (learning rate, architecture depth…)
- Writing clean, modular, object-oriented Python
- Handling matrix shapes and advanced NumPy operations
- Identifying and solving training issues (learning rate too high, dead ReLUs, unstable softmax…)

## Performance
- Validation accuracy 95.8%
  
## Possible Improvements
- Add dropout for regularization
- Implement mini-batch gradient descent
- Add more activation functions (LeakyReLU, Tanh, GELU…)
- Add He/Xavier initialization
- Add plotting utilities for gradients, weights, etc.




