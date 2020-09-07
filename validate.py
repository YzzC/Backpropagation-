from multilayernetwork import MultiLayerNetWork
from util import *
from config import *

if __name__ == "__main__":
    test_data = load_csv_np(path=TEST_DATASET)
    test_label = load_csv_np(path=TEST_DATASET_LABELS)
    net = MultiLayerNetWork.load_net_work(91)
    net.cross_validate(test_data, test_label)
