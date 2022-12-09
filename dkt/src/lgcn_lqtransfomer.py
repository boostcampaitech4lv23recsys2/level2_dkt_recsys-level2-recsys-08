import os
import torch 
import torch.nn as nn

import pandas as pd
import numpy as np
import copy
from .lqtransformer import get_sinusoid_encoding_table, get_pos, Feed_Forward_block
from .get_embed import get_embed

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from lightgcn.config import CFG, logging_conf
# from lightgcn.lightgcn.models import build, train, inference
from lightgcn.lightgcn.utils import class2dict, get_logger
# from lightgcn.lightgcn.datasets import load_data, separate_data, process_data


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

        # big, mid, problem, month, dayname
        self.embedding_big = nn.Embedding(self.args.n_big + 1, self.hidden_dim // 3)
        self.embedding_mid = nn.Embedding(self.args.n_mid + 1, self.hidden_dim // 3)
        self.embedding_problem = nn.Embedding(self.args.n_problem + 1, self.hidden_dim // 3)
        self.embedding_month = nn.Embedding(self.args.n_month + 1, self.hidden_dim // 3)
        self.embedding_dayname = nn.Embedding(self.args.n_dayname + 1, self.hidden_dim // 3)
        self.embedding_user_tag_cluster = nn.Embedding(self.args.n_user_tag_cluster + 1, self.hidden_dim // 3)
        
        # big, mid, problem, month, dayname
        self.cat_proj = nn.Linear((self.hidden_dim // 3) * 9, self.hidden_dim//2) 
        
        # solvesec_600, test_mean, test_std, test_sum, tag_mean, tag_std, tag_sum, big_mean, big_std, big_sum
        self.num_proj = nn.Sequential(nn.Linear(7, self.hidden_dim//2),
                                nn.LayerNorm(self.hidden_dim//2))
        
        # positioal Embedding
        # self.embedding_pos = get_sinusoid_encoding_table(args.max_seq_len, self.hidden_dim)
        # self.embedding_pos =  torch.FloatTensor(self.embedding_pos).to(args.device)
        self.embedding_pos = get_pos(args.max_seq_len).to(args.device)
        # self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim) # 원하는 차원으로 줄이기

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
        self.test_embed_matrix, self.test_n_user = get_embed('testId')
        self.question_embed_matrix, self.question_n_user = get_embed('assessmentItemID')
        
        # 다른 임베딩 벡터들과 차원 맞춰주기
        ## assessmentItemID
        self.lgcn_linear = nn.Linear(128, self.hidden_dim // 3)
        ## testId
        self.lgcn_linear_test = nn.Linear(256, self.hidden_dim // 3)
        
    
    # 문제 lgcn embedding 구하기
    def lgcn_embedding(self, itemnode, item):
        item = item.detach().cpu().numpy()
        
        if itemnode == 'testId':
            embed_matrix, n_user = self.test_embed_matrix, self.test_n_user
        else:
            embed_matrix, n_user = self.question_embed_matrix, self.question_n_user
        embed_matrix = embed_matrix.detach().cpu().numpy()
        
        len_user = n_user - 1

        item_embed = []
        for user in item:
            user_li = []
            for i in user:
                user_li.append(embed_matrix[len_user + i])
            item_embed.append(user_li)

        item_embed = torch.Tensor(np.array(item_embed))
        
        return item_embed

    def forward(self, input):
        test, question, tag, _, mask, interaction, big, mid, problem, month, dayname, user_tag_cluster, solvesec_600, test_mean, test_std, test_sum, tag_mean, tag_std, tag_sum, big_mean, big_std, big_sum, user_correct_answer, user_total_answer, user_acc = input #(test, question, tag, correct, mask, interaction)
        # test, question, tag, _, mask, interaction, new_feature = input

        batch_size = interaction.size(0)

        # Embedding
        embed_interaction = self.embedding_interaction(interaction) #interaction의 값은 0/1/2 중 하나이다.
        
        embed_test = self.embedding_test(test)                #shape = (64,20,21)
        embed_test_lgcn = self.lgcn_embedding('testId', test).to(device)
        embed_test_lgcn = self.lgcn_linear_test(embed_test_lgcn)
        
        embed_question = self.embedding_question(question)
        embed_question_lgcn = self.lgcn_embedding('assessmentItemID', question).to(device)
        embed_question_lgcn = self.lgcn_linear(embed_question_lgcn)
        
        embed_tag = self.embedding_tag(tag)

        # big, mid, problem, month, dayname
        embed_big = self.embedding_big(big)
        embed_mid = self.embedding_mid(mid)
        embed_problem = self.embedding_problem(problem)
        embed_month = self.embedding_month(month)
        embed_dayname = self.embedding_dayname(dayname)
        embed_user_tag_cluster = self.embedding_user_tag_cluster(user_tag_cluster)
        
        embed_cat = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_test_lgcn,
                embed_question,
                embed_question_lgcn,
                embed_tag,
                embed_big,
                # embed_mid,
                # embed_problem,
                embed_month,
                embed_dayname,
                # embed_user_tag_cluster
            ],
            2,
        ) #shape = (64,20,84)

        embed_cat = self.cat_proj(embed_cat)
        
        ## 범주형 변수만 사용 시
        # X = self.comb_proj(embed) #(64,20,64)
        
        #####😘 연속형 변수 추가 시 #####
        # 주의할 점 : cat_proj와 num_proj의 out_dim을 각각 hidden_dim//2로 하기,
        #           137줄 embed_cat 대신 embed로 바꾸기
        #           __init__의 self.num_proj도 수정하기
        
        # solvesec_600, test_mean, test_std, test_sum, tag_mean, tag_std, tag_sum, big_mean, big_std, big_sum
        embed_num = torch.cat(
            [
            solvesec_600.unsqueeze(2), #[64, 20, 1]
            test_mean.unsqueeze(2),
            test_std.unsqueeze(2),
            # test_sum.unsqueeze(2),
            tag_mean.unsqueeze(2),
            tag_std.unsqueeze(2),
            # tag_sum.unsqueeze(2),
            big_mean.unsqueeze(2),
            big_std.unsqueeze(2),
            # big_sum.unsqueeze(2),
            # user_correct_answer.unsqueeze(2),
            # user_total_answer.unsqueeze(2),
            # user_acc.unsqueeze(2),
            # elo_assessmentItemID.unsqueeze(2)
            ],
            2,
        )
        
        embed_num = embed_num.type(torch.FloatTensor).to(device)

        embed_num = self.num_proj(embed_num)
        
        X = torch.cat([embed_cat, embed_num], 2)
        
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