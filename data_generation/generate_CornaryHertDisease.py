import os
import sys
import random
import pickle
import logging
import numpy as np
import pandas as pd

random.seed(1)
np.random.seed(1)
num_clients = 3
num_classes = 2

current_dir = os.path.dirname(__file__)
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

from data_generate import *

data_dir = os.path.join(parent_dir, "datasets/cornary_heart_disease/")
data_path = os.path.join(data_dir, "heart_disease.csv")


def preprocess(data_path):
    data = pd.read_csv(data_path,
                       engine='python',
                       encoding="utf-8",
                       header=None)

    label = data.pop(0)
    x = data.copy()

    float_cols = [9] + np.arange(11, 25).tolist()
    int_cols = np.arange(1, 9).tolist()

    for col in float_cols:
        mean_1 = x[col].mean()
        x[col].fillna(mean_1, inplace=True)

    return x.values, label


def load_heart_disease(disease_path, parent_dir, num_clients, num_classes, niid,
                       balance, partition):
    # Setup directory for train/test data
    config_path = os.path.join(parent_dir, "config.json")
    train_path = os.path.join(parent_dir, "train/")
    test_path = os.path.join(parent_dir, "test/")

    # fill NAs in disease file
    feature, label = preprocess(disease_path)

    (X, y), stats = separate_data((feature, label), num_clients, num_classes,
                                  niid, balance, partition)

    # split data to several clients
    train_data, test_data = [], []
    num_samples = {'train': [], 'test': []}

    for i in range(len(y)):
        X_train, X_test, y_train, y_test = train_test_split(
            X[i], y[i], train_size=train_size, shuffle=True)

        train_data.append({'x': X_train, 'y': y_train})
        test_data.append({'x': X_test, 'y': y_test})

        num_samples['train'].append(len(y_train))
        num_samples['test'].append(len(y_test))

    print("Total number of samples:",
          sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()

    del X, y
    # gc.collect()
    save_file(config_path, train_path, test_path, train_data, test_data,
              num_clients, num_classes, stats, niid, balance, partition)


if __name__ == "__main__":
    niid = True if sys.argv[1] == "noniid" else False
    balance = True if sys.argv[2] == "balance" else False
    partition = sys.argv[3] if sys.argv[3] != "-" else None

    load_heart_disease(data_path, data_dir, num_clients, num_classes, niid,
                       balance, partition)
