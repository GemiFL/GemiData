import os
import sys
import random
import pickle
import logging
import numpy as np
from zipfile import ZipFile

random.seed(1)
np.random.seed(1)
num_clients = 3
num_classes = 2

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

data_dir = os.path.join(parent_dir, "datasets/clintox/")

# sys.path.append(parent_dir)
from data_generate import *

zip_path = os.path.join(data_dir, "clintox.zip")

# set log obj and show level
logging.basicConfig(level=logging.INFO, stream=sys.stdout)
logging.info("The parent_dir is %s", data_dir)


def load_clintox(zip_path, parent_dir, num_clients, num_classes, niid, balance,
                 partition):
    # Setup directory for train/test data
    config_path = os.path.join(parent_dir, "config.json")
    train_path = os.path.join(parent_dir, "train/")
    test_path = os.path.join(parent_dir, "test/")

    with ZipFile(zip_path, "r") as zobj:
        zobj.extractall(path=parent_dir)

    with open(os.path.join(parent_dir, "adjacency_matrices.pkl"), "rb") as f:
        adj_matrix = np.array(pickle.load(f))

    with open(os.path.join(parent_dir, "feature_matrices.pkl"), "rb") as f:
        feature_matrix = np.array(pickle.load(f))

    labels = np.load(os.path.join(parent_dir, "labels.npy"))

    logging.info(f"Length of matrix is %d  %d  {labels.shape}", len(adj_matrix),
                 len(feature_matrix))
    print("labels: ", labels, labels[:, 0].sum(), labels[:, 1].sum(),
          sum(labels))

    # split 'adj_matrix', 'feature_matrix' and 'labels'
    (X, y, adj), stats = separate_data((feature_matrix, labels, adj_matrix),
                                       num_clients, num_classes, niid, balance,
                                       partition)
    train_data, test_data = split_data(sample=(X, y, adj))

    save_file(config_path, train_path, test_path, train_data, test_data,
              num_clients, num_classes, stats, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    load_clintox(zip_path, data_dir, num_clients, num_classes, niid, balance,
                 partition)
