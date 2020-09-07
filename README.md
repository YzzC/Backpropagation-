# Backpropagation
Implement the backpropagation algorithm without using any existing libraries.

## Data
There are four files in Data file. Data is split into two part, one for training and one for test.

## Getting started
Trained model saving: \savings

Model is saved with its accuracy test on data/test_data: train_savings{accuracy}.json

Validation sample (with model load from savings): python validate.py

One shot output sample (with model load from savings): python oneshot.py

Training data (training new model with data/train_data): python train.py

Hyper parameters: config.py
