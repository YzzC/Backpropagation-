python version:
3.7.7

data is split into two part
one for training and one for test

trained model saving:
\savings

model is saved with its accuracy test on data/test_data:
train_savings{accuracy}.json

validation sample (with model load from savings):
python validate.py

one shot output sample (with model load from savings):
python oneshot.py

training data (training new model with data/train_data):
python train.py

hyper parameters:
config.py
