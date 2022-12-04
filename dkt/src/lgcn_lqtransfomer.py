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
from lightgcn.lightgcn.datasets import load_data, separate_data, process_data


logger = get_logger(logging_conf)
use_cuda = torch.cuda.is_available() and CFG.use_cuda_if_available
device = torch.device("cuda" if use_cuda else "cpu")
print(device)

class lightGCN_LQTransformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.hidden_dim = self.args.hidden_dim

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim // 3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)
        self.embedding_big = nn.Embedding(self.args.n_big + 1, self.hidden_dim // 3)
        self.embedding_mid = nn.Embedding(self.args.n_mid + 1, self.hidden_dim // 3)
        self.embedding_problem = nn.Embedding(self.args.n_problem + 1, self.hidden_dim // 3)
        self.embedding_month = nn.Embedding(self.args.n_month + 1, self.hidden_dim // 3)
        self.embedding_day = nn.Embedding(self.args.n_day + 1, self.hidden_dim // 3)
        self.embedding_solvesec = nn.Embedding(self.args.n_solvesec + 1, self.hidden_dim // 3)
        self.embedding_bigacc = nn.Embedding(self.args.n_bigacc + 1, self.hidden_dim // 3)
        self.embedding_bigstd = nn.Embedding(self.args.n_bigstd + 1, self.hidden_dim // 3)
        
        # positioal Embedding
        # self.embedding_pos = get_sinusoid_encoding_table(args.max_seq_len, self.hidden_dim)
        # self.embedding_pos =  torch.FloatTensor(self.embedding_pos).to(args.device)
        self.embedding_pos = get_pos(args.max_seq_len).to(args.device)
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 12, self.hidden_dim) # 원하는 차원으로 줄이기

        # multihead attention(여기선 head를 1로 두었다.)
        self.multi_en = nn.MultiheadAttention( embed_dim= self.hidden_dim, num_heads= 1, dropout=0.1  )     # multihead attention    ## todo add dropout, LayerNORM
        
        #lstm
        self.lstm = nn.LSTM(input_size= args.hidden_dim, hidden_size = self.hidden_dim, num_layers=1)

        # layer norm
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)
        
        # feed-forward
        self.ffn_en = Feed_Forward_block(self.hidden_dim, 4*self.hidden_dim)  
        
        # 최종 Wo 곱하기
        self.out = nn.Linear(in_features= args.hidden_dim , out_features=1)                                          # feedforward block     ## todo dropout, LayerNorm
        
        # lightGCN embed matrix와 id2index
        self.embed_matrix, self.n_user = self.get_embed()
        # 다른 임베딩 벡터들과 차원 맞춰주기
        self.lgcn_linear = nn.Linear(CFG.embedding_dim, self.hidden_dim // 3)
    
    ### lightGCN 모델 데이터셋 준비
    def indexing_data(self, data):
        userid, itemid = (
            sorted(list(set(data.userID))),
            sorted(list(set(data.assessmentItemID))),
        )
        n_user, n_item = len(userid), len(itemid)

        userid_2_index = {v: i for i, v in enumerate(userid)}
        itemid_2_index = {v: i + n_user for i, v in enumerate(itemid)}
        id_2_index = dict(userid_2_index, **itemid_2_index)

        return n_user, id_2_index

    def prepare_dataset(self, device, basepath, verbose=True, logger=None):
        
        data = load_data(basepath)
        train_data, test_data = separate_data(data)
        n_user, id2index = self.indexing_data(data)
        train_data_proc = process_data(train_data, id2index, device)
        test_data_proc = process_data(test_data, id2index, device)

        return train_data_proc, test_data_proc, len(id2index), n_user
    
    ### lightGCN 모델에서 임베딩, 인덱스 가져오기
    def get_embed(self):

        train_data, test_data, n_node, n_user = self.prepare_dataset(
            device, CFG.basepath, verbose=CFG.loader_verbose, logger=logger.getChild("data")
        )

        model = build(
            n_node,
            embedding_dim=CFG.embedding_dim,
            num_layers=CFG.num_layers,
            alpha=CFG.alpha,
            weight="/opt/ml/dkt_team/code/lightgcn/weight/best_model.pt",
            logger=logger.getChild("build"),
            **CFG.build_kwargs
        )
        model.to(device)

        embed_matrix = model.get_embedding(train_data['edge']).to(device)
        
        return embed_matrix, n_user
    
    # 문제 lgcn embedding 구하기
    def lgcn_embedding_question(self, question):
        question = question.detach().cpu().numpy()
        embed_matrix = self.embed_matrix.detach().cpu().numpy()
        len_user = self.n_user - 1

        question_embed = []
        for user in question:
            user_li = []
            for item in user:
                user_li.append(embed_matrix[len_user + item])
            question_embed.append(user_li)

        question_embed = torch.Tensor(np.array(question_embed))
        
        return question_embed

    def forward(self, input):
        test, question, tag, _, mask, interaction, big, mid, problem, month, day, solvesec, bigacc, bigstd = input #(test, question, tag, correct, mask, interaction)

        # Embedding
        embed_test = self.embedding_test(test)                #shape = (64,20,21)
        # embed_test = nn.Dropout(0.1)(embed_test)
        
        # embed_question = self.embedding_question(question)
        # embed_question = nn.Dropout(0.1)(embed_question)

        embed_question = self.lgcn_embedding_question(question).to(device)
        embed_question = self.lgcn_linear(embed_question)
        
        embed_tag = self.embedding_tag(tag) 
        # embed_tag = nn.Dropout(0.1)(embed_tag)
        embed_interaction = self.embedding_interaction(interaction) #interaction의 값은 0/1/2 중 하나이다.
        # embed_interaction = nn.Dropout(0.1)(embed_interaction)

        embed_big = self.embedding_big(big)
        embed_mid = self.embedding_mid(mid)
        embed_problem = self.embedding_problem(problem)
        embed_month = self.embedding_month(month)
        embed_day = self.embedding_day(day)
        embed_solvesec = self.embedding_solvesec(solvesec)
        embed_bigacc = self.embedding_bigacc(bigacc)
        embed_bigstd = self.embedding_bigstd(bigstd)
        
        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
                embed_big,
                embed_mid,
                embed_problem,
                embed_month,
                embed_day,
                embed_solvesec,
                embed_bigacc,
                embed_bigstd
            ],
            2,
        )

        X = self.comb_proj(embed)
        X = X + self.embedding_pos #(64,20,64) (batch,seq,dim)
        # X = nn.Dropout(0.1)(X)

        X = X.permute(1,0,2)       #(20,64,64)
        X = self.layer_norm1(X)
        skip_X = X

        X, attn_wt = self.multi_en(X[-1:,:,:], X, X)         # Q,K,V
        
        X = X + skip_X #[20,64,64]


        X, _ = self.lstm(X)
        X = X[-1:,:,:]

        #feed forward
        X = X.permute(1,0,2)                                # (b,n,d)
        X = self.layer_norm2(X)                           # Layer norm 
        skip_X = X
        X = self.ffn_en( X )
        X = X + skip_X                                    # skip connection
        
        X = self.out( X )

        return X.squeeze(-1)