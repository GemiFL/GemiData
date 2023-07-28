import os
import pickle
import ujson
import numpy as np
import gc
import scipy.sparse as sp
import torch
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch.utils.data as data

batch_size = 10
train_size = 0.75  # merge original training set and test set, then split it manually.
least_samples = batch_size / (1 - train_size)  # least samples for each client
alpha = 0.1  # for Dirichlet distribution


class DefaultCollator(object):

    def __init__(self, normalize_features=True, normalize_adj=True):
        self.normalize_features = normalize_features
        self.normalize_adj = normalize_adj

    def __call__(self, molecule):
        adj_matrix, feature_matrix, label, _ = molecule[0]
        mask = np.where(np.isnan(label), 0.0, 1.0)
        label = np.where(np.isnan(label), 0.0, label)

        if self.normalize_features:
            mx = sp.csr_matrix(feature_matrix)
            rowsum = np.array(mx.sum(1))
            r_inv = np.power(rowsum, -1).flatten()
            r_inv[np.isinf(r_inv)] = 0.0
            r_mat_inv = sp.diags(r_inv)
            normalized_feature_matrix = r_mat_inv.dot(mx)
            normalized_feature_matrix = np.array(
                normalized_feature_matrix.todense())
        else:
            scaler = StandardScaler()
            scaler.fit(feature_matrix)
            normalized_feature_matrix = scaler.transform(feature_matrix)

        if self.normalize_adj:
            rowsum = np.array(adj_matrix.sum(1))
            r_inv_sqrt = np.power(rowsum, -0.5).flatten()
            r_inv_sqrt[np.isinf(r_inv_sqrt)] = 0.0
            r_mat_inv_sqrt = sp.diags(r_inv_sqrt)
            normalized_adj_matrix = (
                adj_matrix.dot(r_mat_inv_sqrt).transpose().dot(r_mat_inv_sqrt))
        else:
            normalized_adj_matrix = adj_matrix

        return (
            torch.as_tensor(np.array(normalized_adj_matrix.todense()),
                            dtype=torch.float32),
            torch.as_tensor(normalized_feature_matrix, dtype=torch.float32),
            torch.as_tensor(label, dtype=torch.float32),
            torch.as_tensor(mask, dtype=torch.float32),
        )


class MoleculesDataset(data.Dataset):

    def __init__(
        self,
        adj_matrices,
        feature_matrices,
        labels,
        compact=True,
        fanouts=[2, 2],
        split="train",
    ):
        self.adj_matrices = adj_matrices

        self.feature_matrices = feature_matrices
        self.labels = labels
        self.fanouts = [fanouts] * len(adj_matrices)

    def __getitem__(self, index):
        return (
            self.adj_matrices[index],
            self.feature_matrices[index],
            self.labels[index],
            self.fanouts[index],
        )

    def __len__(self):
        return len(self.adj_matrices)


def check(config_path,
          train_path,
          test_path,
          num_clients,
          num_classes,
          niid=False,
          balance=True,
          partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = ujson.load(f)
        if config['num_clients'] == num_clients and \
            config['num_classes'] == num_classes and \
            config['non_iid'] == niid and \
            config['balance'] == balance and \
            config['partition'] == partition and \
            config['alpha'] == alpha and \
            config['batch_size'] == batch_size:
            print("\nDataset already generated.\n")
            return True

    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    return False


def separate_data(data,
                  num_clients,
                  num_classes,
                  niid=False,
                  balance=False,
                  partition=None,
                  class_per_client=2):
    X = [[] for _ in range(num_clients)]
    y = [[] for _ in range(num_clients)]
    adj = [[] for _ in range(num_clients)]
    statistic = [[] for _ in range(num_clients)]

    if len(data) == 2:
        dataset_content, dataset_label = data
    elif len(data) == 3:
        dataset_content, dataset_label, dataset_adj = data
    else:
        raise ValueError("It's not implemented!")

    dataidx_map = {}

    if not niid:
        partition = 'pat'
        class_per_client = num_classes

    if partition == 'pat':
        idxs = np.array(range(len(dataset_label)))
        idx_for_each_class = []
        for i in range(num_classes):
            # idx_for_each_class.append(idxs[dataset_label == i])
            if len(data) == 3:
                idx_for_each_class.append(idxs[dataset_label[:, 1] == i])
            elif len(data) == 2:
                idx_for_each_class.append(idxs[dataset_label == i])
            else:
                raise ValueError("It is not implemented!")

        class_num_per_client = [class_per_client for _ in range(num_clients)]
        for i in range(num_classes):
            selected_clients = []
            for client in range(num_clients):
                if class_num_per_client[client] > 0:
                    selected_clients.append(client)
                selected_clients = selected_clients[:int(num_clients /
                                                         num_classes *
                                                         class_per_client)]

            num_all_samples = len(idx_for_each_class[i])
            num_selected_clients = len(selected_clients)
            num_per = num_all_samples / num_selected_clients
            if balance:
                num_samples = [
                    int(num_per) for _ in range(num_selected_clients - 1)
                ]
            else:
                num_samples = np.random.randint(
                    max(num_per / 10, least_samples / num_classes), num_per,
                    num_selected_clients - 1).tolist()
            num_samples.append(num_all_samples - sum(num_samples))

            idx = 0
            for client, num_sample in zip(selected_clients, num_samples):
                if client not in dataidx_map.keys():
                    dataidx_map[client] = idx_for_each_class[i][idx:idx +
                                                                num_sample]
                else:
                    dataidx_map[client] = np.append(
                        dataidx_map[client],
                        idx_for_each_class[i][idx:idx + num_sample],
                        axis=0)
                idx += num_sample
                class_num_per_client[client] -= 1

    elif partition == "dir":
        # https://github.com/IBM/probabilistic-federated-neural-matching/blob/master/experiment.py
        min_size = 0
        K = num_classes
        N = len(dataset_label)

        while min_size < least_samples:
            idx_batch = [[] for _ in range(num_clients)]
            for k in range(K):
                # idx_k = np.where(dataset_label == k)[0]
                idx_k = np.where(dataset_label[:, 1] == k)[0]
                np.random.shuffle(idx_k)
                proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
                proportions = np.array([
                    p * (len(idx_j) < N / num_clients)
                    for p, idx_j in zip(proportions, idx_batch)
                ])
                proportions = proportions / proportions.sum()
                proportions = (np.cumsum(proportions) *
                               len(idx_k)).astype(int)[:-1]
                idx_batch = [
                    idx_j + idx.tolist() for idx_j, idx in zip(
                        idx_batch, np.split(idx_k, proportions))
                ]
                min_size = min([len(idx_j) for idx_j in idx_batch])

        for j in range(num_clients):
            dataidx_map[j] = idx_batch[j]
    else:
        raise NotImplementedError

    # assign data
    for client in range(num_clients):
        idxs = dataidx_map[client]
        X[client] = dataset_content[idxs]
        y[client] = dataset_label[idxs]

        if len(data) == 3:
            adj[client] = dataset_adj[idxs]

        for i in np.unique(y[client]):
            # statistic[client].append((int(i), int(sum(y[client] == i))))
            if len(data) == 3:
                statistic[client].append((int(i), int(sum(y[client][:,
                                                                    1] == i))))
            elif len(data) == 2:
                statistic[client].append((int(i), int(sum(y[client] == i))))
            else:
                raise ValueError("It is not implemented!")

    # del data
    # gc.collect()

    for client in range(num_clients):
        print(f"Client {client}\t Size of data: {len(X[client])}\t Labels: ",
              np.unique(y[client]))
        print(f"\t\t Samples of labels: ", [i for i in statistic[client]])
        print("-" * 50)

    if len(data) == 3:
        return (X, y, adj), statistic
    elif len(data) == 2:
        return (X, y), statistic
    else:
        raise ValueError("It's not implemented!")


def split_data(sample, normalize_features=False, normalize_adj=False):
    if len(sample) == 2:
        X, y = sample
    elif len(sample) == 3:
        X, y, adj = sample
    else:
        raise ValueError("It's not implemented!")
    # Split dataset
    train_data, test_data = [], []
    num_samples = {'train': [], 'test': []}

    for i in range(len(y)):

        if len(sample) == 2:
            X_train, X_test, y_train, y_test = train_test_split(
                X[i], y[i], train_size=train_size, shuffle=True)

            train_torch_dataset = data.Data.TensorDataset(X_train, y_train)
            test_torch_dataset = data.Data.TensorDataset(X_test, y_test)

            train_dataloader = data.DataLoader(train_torch_dataset,
                                               batch_size=1,
                                               shuffle=True)

            test_dataloader = data.DataLoader(test_torch_dataset,
                                              batch_size=1,
                                              shuffle=False)

            # train_data.append({'x': X_train, 'y': y_train})
            train_data.append(train_dataloader)
            num_samples['train'].append(len(y_train))
            # test_data.append({'x': X_test, 'y': y_test})
            test_data.append(test_dataloader)
            num_samples['test'].append(len(y_test))

        elif len(sample) == 3:
            X_train, X_test, y_train, y_test, adj_train, adj_test = train_test_split(
                X[i], y[i], adj[i], train_size=train_size, shuffle=True)

            train_dataset = MoleculesDataset(adj_matrices=adj_train,
                                             feature_matrices=X_train,
                                             labels=y_train)

            test_dataset = MoleculesDataset(adj_matrices=adj_test,
                                            feature_matrices=X_test,
                                            labels=y_test)

            collator = DefaultCollator(normalize_features=normalize_features,
                                       normalize_adj=normalize_adj)

            train_dataloader = data.DataLoader(train_dataset,
                                               batch_size=1,
                                               shuffle=True,
                                               collate_fn=collator,
                                               pin_memory=True)

            test_dataloader = data.DataLoader(test_dataset,
                                              batch_size=1,
                                              shuffle=False,
                                              collate_fn=collator,
                                              pin_memory=True)

            # train_data.append({'x': train_dataloader, 'y': test_dataloader})
            train_data.append(train_dataloader)
            num_samples['train'].append(len(y_train))
            # test_data.append({'x': X_test, 'y': y_test, 'adj': adj_test})
            test_data.append(test_dataloader)
            num_samples['test'].append(len(y_test))

    print("Total number of samples:",
          sum(num_samples['train'] + num_samples['test']))
    print("The number of train samples:", num_samples['train'])
    print("The number of test samples:", num_samples['test'])
    print()

    del X, y
    # gc.collect()

    return train_data, test_data


def save_file(config_path,
              train_path,
              test_path,
              train_data,
              test_data,
              num_clients,
              num_classes,
              statistic,
              niid=False,
              balance=True,
              partition=None):
    config = {
        'num_clients': num_clients,
        'num_classes': num_classes,
        'non_iid': niid,
        'balance': balance,
        'partition': partition,
        'Size of samples for labels in clients': statistic,
        'alpha': alpha,
        'batch_size': batch_size,
    }

    # gc.collect()
    print("Saving to disk.\n")

    if not os.path.exists(train_path):
        os.makedirs(train_path)

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    for idx, train_dict in enumerate(train_data):

        with open(train_path + str(idx) + '.pkl', 'wb') as f:
            pickle.dump(train_dict, f)
            # np.savez_compressed(f, data=train_dict)
    for idx, test_dict in enumerate(test_data):
        with open(test_path + str(idx) + '.pkl', 'wb') as f:
            # np.savez_compressed(f, data=test_dict)
            pickle.dump(test_dict, f)
    with open(config_path, 'w') as f:
        ujson.dump(config, f)

    print("Finish generating dataset.\n")
