import numpy as np

from multilayernetwork import MultiLayerNetWork
from util import *
from config import *


def train():
    train_data = list(load_csv(path=TRAIN_DATASET))
    train_label = list(load_csv(path=TRAIN_DATASET_LABELS))
    test_data = load_csv_np(path=TEST_DATASET)
    test_label = load_csv_np(path=TEST_DATASET_LABELS)

    model = MultiLayerNetWork(n_input=INPUT_SIZE,
                              n_hidden=HIDDEN_SIZE,
                              n_output=OUT_SIZE,
                              n_epoch=N_EPOCH,
                              batch_size=BATCH_SIZE,
                              learning_rate=LEARNING_RATE,
                              bias=BIAS)
    model.train_from_gen(train_data, train_label, test_data, test_label)


if __name__ == "__main__":
    train()
