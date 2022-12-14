import os

import argparse
import pandas as pd
import torch


def prepare_dataset(itemnode, device, basepath, verbose=True, logger=None):
    data = load_data(itemnode, basepath)
    train_data, test_data = separate_data(data)
    id2index = indexing_data(itemnode, data)
    train_data_proc = process_data(itemnode, train_data, id2index, device)
    test_data_proc = process_data(itemnode, test_data, id2index, device)

    if verbose:
        print_data_stat(itemnode, train_data, "Train", logger=logger)
        print_data_stat(itemnode, test_data, "Test", logger=logger)

    return train_data_proc, test_data_proc, len(id2index)


def load_data(itemnode, basepath):
    path1 = os.path.join(basepath, "train_data.csv")
    path2 = os.path.join(basepath, "test_data.csv")
    data1 = pd.read_csv(path1) #(2266586, 6)
    data2 = pd.read_csv(path2) #(260114, 6)

    data = pd.concat([data1, data2]) #(2526700, 6)
    data.drop_duplicates(            #(2476706, 6)
        subset=["userID", itemnode], keep="last", inplace=True
    )

    return data


def separate_data(data):
    train_data = data[data.answerCode >= 0]
    test_data = data[data.answerCode < 0]

    return train_data, test_data


def indexing_data(itemnode, data):
    userid, itemid = (
        sorted(list(set(data.userID))),
        sorted(list(set(data[itemnode]))),
    )
    n_user, n_item = len(userid), len(itemid)

    userid_2_index = {v: i for i, v in enumerate(userid)}
    itemid_2_index = {v: i + n_user for i, v in enumerate(itemid)}
    id_2_index = dict(userid_2_index, **itemid_2_index)

    return id_2_index


def process_data(itemnode, data, id_2_index, device):
    edge, label = [], []
    for user, item, acode in zip(data.userID, data[itemnode], data.answerCode):
        uid, iid = id_2_index[user], id_2_index[item]
        edge.append([uid, iid])
        label.append(acode)

    edge = torch.LongTensor(edge).T  #train_data일 때 : len(shape) = (2475962,2) -> (2, 2475962)
    label = torch.LongTensor(label)  #train_data일 때 : len = 2475962

    return dict(edge=edge.to(device), label=label.to(device))


def print_data_stat(itemnode, data, name, logger):
    userid, itemid = list(set(data.userID)), list(set(data[itemnode]))
    n_user, n_item = len(userid), len(itemid)

    logger.info(f"{name} Dataset Info")
    logger.info(f" * Num. Users    : {n_user}")
    logger.info(f" * Max. UserID   : {max(userid)}")
    logger.info(f" * Num. Items    : {n_item}")
    logger.info(f" * Num. Records  : {len(data)}")
