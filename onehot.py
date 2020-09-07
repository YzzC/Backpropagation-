import numpy as np

from multilayernetwork import MultiLayerNetWork
from util import load_csv_np
from config import *


def one_shot_prediction(file_name) -> np.ndarray:
    """
    this load network and given a one shot predict to data
    :param file_name: read from file name
    :return one shot encoded in numpy array
    """
    np.set_printoptions(threshold=np.inf)
    data = load_csv_np(path=file_name)
    model = MultiLayerNetWork.load_net_work(91)
    predicts = model.one_hot_predict(data)
    return predicts


if __name__ == "__main__":
    print(one_shot_prediction(TEST_DATASET))
