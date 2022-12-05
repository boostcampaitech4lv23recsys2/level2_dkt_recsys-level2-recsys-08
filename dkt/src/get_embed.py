import os
import torch 
import torch.nn as nn

import pandas as pd
import numpy as np
import copy
from .lqtransformer import get_sinusoid_encoding_table, get_pos, Feed_Forward_block

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from lightgcn.config import CFG, logging_conf
from lightgcn.lightgcn.models import build, train, inference
from lightgcn.lightgcn.utils import class2dict, get_logger
from lightgcn.lightgcn.datasets import separate_data

logger = get_logger(logging_conf)
use_cuda = torch.cuda.is_available() and CFG.use_cuda_if_available
device = torch.device("cuda" if use_cuda else "cpu")

 ### lightGCN 모델 데이터셋 준비 (testId)
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

def indexing_data(itemnode, data):
    userid, itemid = (
        sorted(list(set(data.userID))),
        sorted(list(set(data[itemnode]))),
    )
    n_user, n_item = len(userid), len(itemid)

    userid_2_index = {v: i for i, v in enumerate(userid)}
    itemid_2_index = {v: i + n_user for i, v in enumerate(itemid)}
    id_2_index = dict(userid_2_index, **itemid_2_index)

    return n_user, id_2_index

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
    
def prepare_dataset(itemnode, device, basepath, verbose=True, logger=None):
    
    data = load_data(itemnode, basepath)
    train_data, test_data = separate_data(data)
    n_user, id2index = indexing_data(itemnode, data)
    train_data_proc = process_data(itemnode, train_data, id2index, device)
    test_data_proc = process_data(itemnode, test_data, id2index, device)

    return train_data_proc, test_data_proc, len(id2index), n_user

### lightGCN 모델에서 임베딩, 인덱스 가져오기 (testId)
def get_embed(itemnode):

    train_data, test_data, n_node, n_user = prepare_dataset(itemnode,
        device, CFG.basepath, verbose=CFG.loader_verbose, logger=logger.getChild("data")
    )

    if itemnode != 'assessmentItemID':
        weight = "/opt/ml/dkt_team/code/lightgcn/weight/" + itemnode + "_best_model.pt"
    else:
        weight = "/opt/ml/dkt_team/code/lightgcn/weight/best_model.pt"
        
    model = build(itemnode,
        n_node,
        embedding_dim=CFG.embedding_dim,
        num_layers=CFG.num_layers,
        alpha=CFG.alpha,
        weight=weight,
        logger=logger.getChild("build"),
        **CFG.build_kwargs
    )
    model.to(device)

    embed_matrix = model.get_embedding(train_data['edge']).to(device)
    
    return embed_matrix, n_user

# 문제 lgcn embedding 구하기
# def lgcn_embedding(itemnode, item):
#     item = item.detach().cpu().numpy()
    
#     embed_matrix, n_user = get_embed(itemnode)
#     embed_matrix = embed_matrix.detach().cpu().numpy()

#     len_user = n_user - 1

#     item_embed = []
#     for user in item:
#         user_li = []
#         for i in user:
#             user_li.append(embed_matrix[len_user + i])
#         item_embed.append(user_li)

#     item_embed = torch.Tensor(np.array(item_embed))
    
#     return item_embed
