
import os
import torch
import torch.nn as nn
import numpy as np
from .get_embed import get_embed

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import (
        BertConfig,
        BertEncoder,
        BertModel,
    )

import sys
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))

from lightgcn.config import CFG, logging_conf
from lightgcn.lightgcn.utils import class2dict, get_logger

logger = get_logger(logging_conf)
use_cuda = torch.cuda.is_available() and CFG.use_cuda_if_available
device = torch.device("cuda" if use_cuda else "cpu")

class LSTM(nn.Module):
    #new_feature 를 추가하고 싶으시면, embedding 부분만 바꾸면 됩니다.
    #lqtransformer.py의 임베딩 부분을 참고해주세요! -> lqtransformer.py에서 "new_feature" 찾기 기능
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

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
        self.embedding_assIdx = nn.Embedding(self.args.n_assIdx + 1, self.hidden_dim // 3) 
        self.embedding_month = nn.Embedding(self.args.n_month + 1, self.hidden_dim // 3)
        self.embedding_day = nn.Embedding(self.args.n_day+ 1, self.hidden_dim // 3) 
        self.embedding_hour= nn.Embedding(self.args.n_hour + 1, self.hidden_dim // 3) 
        self.embedding_dayname = nn.Embedding(self.args.n_dayname + 1, self.hidden_dim // 3)
        self.embedding_time_category = nn.Embedding(self.args.n_time_category + 1, self.hidden_dim // 3) 
        self.embedding_solvecumsum_category = nn.Embedding(self.args.n_solvecumsum_category + 1, self.hidden_dim // 3) 
        # self.embedding_user_tag_cluster = nn.Embedding(self.args.n_user_tag_cluster + 1, self.hidden_dim // 3)        
        
        # big, mid, problem, month, dayname
        self.cat_proj = nn.Linear((self.hidden_dim // 3) * 13, self.hidden_dim//2) 
        
        # solvesec_600, test_mean, test_std, test_sum, tag_mean, tag_std, tag_sum, big_mean, big_std, big_sum
        self.num_proj = nn.Sequential(nn.Linear(17, self.hidden_dim//2),
                                nn.LayerNorm(self.hidden_dim//2))
        
        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim) # 원하는 차원으로 줄이기

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True #nn.LSTM(input_size, hidden_size, num_layers, ...)
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)
        
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

        # test, question, tag, _, mask, interaction, big, mid, problem, month, dayname, month_mean, solvesec_3600, test_mean, test_std, test_sum, tag_mean, tag_std, tag_sum, big_mean, big_std, big_sum, user_correct_answer, user_total_answer, user_acc = input #(test, question, tag, correct, mask, interaction)
        test, question, tag, _, mask, interaction, big_category, mid_category, problem_num, assIdx, month, day, hour, dayname, time_category, solvecumsum_category,  \
        solvesec_3600, test_mean, test_std, tag_mean, tag_std, big_mean, big_std, user_correct_answer, user_total_answer, user_acc, \
        solvesec_cumsum,  big_category_cumconut, big_category_user_acc, big_category_user_std, big_category_answer, big_category_answer_log1p, elo_assessmentItemID = input

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
        embed_big = self.embedding_big(big_category)
        embed_mid = self.embedding_mid(mid_category)
        embed_problem = self.embedding_problem(problem_num) 
        embed_assIdx= self.embedding_assIdx(assIdx)
        embed_month = self.embedding_month(month)
        embed_day = self.embedding_day(day)
        embed_hour = self.embedding_hour(hour)
        embed_dayname = self.embedding_dayname(dayname)
        embed_time_category = self.embedding_time_category(time_category)
        embed_solvecumsum_category = self.embedding_solvecumsum_category(solvecumsum_category)
        # embed_user_tag_cluster = self.embedding_user_tag_cluster(user_tag_cluster)    
            
        embed_cat = torch.cat(
            [
                embed_interaction,
                # embed_test,
                embed_test_lgcn,
                # embed_question,
                embed_question_lgcn,
                embed_tag,
                embed_big,
                # embed_mid,
                embed_problem,
                embed_assIdx,
                embed_month,
                embed_day,
                embed_hour,
                embed_dayname,
                embed_time_category,
                embed_solvecumsum_category,
                # embed_user_tag_cluster
            ],
            2,
        ) #shape = (64,20,84)

        embed_cat = self.cat_proj(embed_cat)
        
        ## 범주형 변수만 사용 시
        # X = self.comb_proj(embed_cat) #(64,20,64)
        
        #####😘 연속형 변수 추가 시 #####
        # 주의할 점 : cat_proj와 num_proj의 out_dim을 각각 hidden_dim//2로 하기,
        #           137줄 embed_cat 대신 embed로 바꾸기
        #           __init__의 self.num_proj도 수정하기
        
        embed_num = torch.cat(
            [
        #     month_mean.unsqueeze(2),
            solvesec_3600.unsqueeze(2), #[64, 20, 1]
            test_mean.unsqueeze(2),
            test_std.unsqueeze(2),
        #     test_sum.unsqueeze(2),
            tag_mean.unsqueeze(2),
            tag_std.unsqueeze(2),
        #     tag_sum.unsqueeze(2),
            big_mean.unsqueeze(2),
            big_std.unsqueeze(2),
        #     big_sum.unsqueeze(2),
            user_correct_answer.unsqueeze(2),
            user_total_answer.unsqueeze(2),
            user_acc.unsqueeze(2),
            solvesec_cumsum.unsqueeze(2),
            big_category_cumconut.unsqueeze(2),
            big_category_user_acc.unsqueeze(2),
            big_category_user_std.unsqueeze(2),
            big_category_answer.unsqueeze(2),
            big_category_answer_log1p.unsqueeze(2),
            elo_assessmentItemID.unsqueeze(2)
            ],
            2,
        )
        
        embed_num = embed_num.type(torch.FloatTensor).to(device)
        embed_num = self.num_proj(embed_num)
        X = torch.cat([embed_cat, embed_num], 2)
        
        out, _ = self.lstm(X) #hidden, cell state 반환                 #out.shape = (64,20,64)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out).view(batch_size, -1)                       #out.shape = (64,20)
        return out


class LSTMATTN(nn.Module):
    def __init__(self, args):
        super(LSTMATTN, self).__init__()
        self.args = args

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers
        self.n_heads = self.args.n_heads
        self.drop_out = self.args.drop_out

        # Embedding
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)
        # big, mid, problem, month, dayname
        self.embedding_big = nn.Embedding(self.args.n_big + 1, self.hidden_dim // 3)
        self.embedding_mid = nn.Embedding(self.args.n_mid + 1, self.hidden_dim // 3)
        self.embedding_problem = nn.Embedding(self.args.n_problem + 1, self.hidden_dim // 3)
        self.embedding_month = nn.Embedding(self.args.n_month + 1, self.hidden_dim // 3)
        self.embedding_dayname = nn.Embedding(self.args.n_dayname + 1, self.hidden_dim // 3)
        
        # big, mid, problem, month, dayname
        self.cat_proj = nn.Linear((self.hidden_dim // 3) * 11, self.hidden_dim//2) 
        
        # solvesec_600, test_mean, test_std, test_sum, tag_mean, tag_std, tag_sum, big_mean, big_std, big_sum
        self.num_proj = nn.Sequential(nn.Linear(10, self.hidden_dim//2),
                                nn.LayerNorm(self.hidden_dim//2))
        
        # embedding combination projection
        # self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True
        )

        ## 이 부분도 Attention에 의해 추가된 부분
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=1,
            num_attention_heads=self.n_heads,
            intermediate_size=self.hidden_dim,
            hidden_dropout_prob=self.drop_out,
            attention_probs_dropout_prob=self.drop_out,
        )
        self.attn = BertEncoder(self.config)

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        self.activation = nn.Sigmoid()
        
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
        test, question, tag, _, mask, interaction, big, mid, problem, month, dayname, solvesec_600, test_mean, test_std, test_sum, tag_mean, tag_std, tag_sum, big_mean, big_std, big_sum = input #(test, question, tag, correct, mask, interaction)
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
        
        embed_cat = torch.cat(
            [
                embed_interaction,
                embed_test, # 209줄 같이 바꿔줘야 함
                embed_test_lgcn,
                embed_question,
                embed_question_lgcn,
                embed_tag,
                embed_big,
                embed_mid,
                embed_problem,
                embed_month,
                embed_dayname
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
            test_sum.unsqueeze(2),
            tag_mean.unsqueeze(2),
            tag_std.unsqueeze(2),
            tag_sum.unsqueeze(2),
            big_mean.unsqueeze(2),
            big_std.unsqueeze(2),
            big_sum.unsqueeze(2),
            ],
            2,
        )
        
        embed_num = embed_num.type(torch.FloatTensor).to(device)

        embed_num = self.num_proj(embed_num)
        
        X = torch.cat([embed_cat, embed_num], 2)

        out, _ = self.lstm(X)
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)

        ## 이 부분이 특징적인  부분
        extended_attention_mask = mask.unsqueeze(1).unsqueeze(2) # 차원 늘려주기
        extended_attention_mask = extended_attention_mask.to(dtype=torch.float32)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0 # 마스킹 된 부분 가중치 낮게 줘서 학습 안되도록
        head_mask = [None] * self.n_layers

        encoded_layers = self.attn(out, extended_attention_mask, head_mask=head_mask) # BERT Encoder를 가져다 씀
        sequence_output = encoded_layers[-1]

        out = self.fc(sequence_output).view(batch_size, -1)
        return out


class Bert(nn.Module):
    def __init__(self, args):
        super(Bert, self).__init__()
        self.args = args

        # Defining some parameters
        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        # Embedding
        # interaction은 현재 correct으로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)

        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(
            self.args.n_questions + 1, self.hidden_dim // 3
        )

        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)

        # big, mid, problem, month, dayname
        self.embedding_big = nn.Embedding(self.args.n_big + 1, self.hidden_dim // 3)
        self.embedding_mid = nn.Embedding(self.args.n_mid + 1, self.hidden_dim // 3)
        self.embedding_problem = nn.Embedding(self.args.n_problem + 1, self.hidden_dim // 3)
        self.embedding_month = nn.Embedding(self.args.n_month + 1, self.hidden_dim // 3)
        self.embedding_dayname = nn.Embedding(self.args.n_dayname + 1, self.hidden_dim // 3)
        
        # big, mid, problem, month, dayname
        self.cat_proj = nn.Linear((self.hidden_dim // 3) * 9, self.hidden_dim//2) 
        
        # solvesec_600, test_mean, test_std, test_sum, tag_mean, tag_std, tag_sum, big_mean, big_std, big_sum
        self.num_proj = nn.Sequential(nn.Linear(10, self.hidden_dim//2),
                                nn.LayerNorm(self.hidden_dim//2))
        
        # embedding combination projection
        # self.comb_proj = nn.Linear((self.hidden_dim // 3) * 21, self.hidden_dim) # 원하는 차원으로 줄이기

        # Bert config
        self.config = BertConfig(
            3,  # not used
            hidden_size=self.hidden_dim,
            num_hidden_layers=self.args.n_layers,
            num_attention_heads=self.args.n_heads,
            max_position_embeddings=self.args.max_seq_len,
        )

        # Defining the layers
        # Bert Layer
        self.encoder = BertModel(self.config) # 버트라고 되어있지만, 트랜스포머 인코더부분

        # Fully connected layer
        self.fc = nn.Linear(self.args.hidden_dim, 1)

        self.activation = nn.Sigmoid()
        
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
        test, question, tag, _, mask, interaction, big, mid, problem, month, dayname, solvesec_600, test_mean, test_std, test_sum, tag_mean, tag_std, tag_sum, big_mean, big_std, big_sum = input #(test, question, tag, correct, mask, interaction)
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
        
        embed_cat = torch.cat(
            [
                embed_interaction,
                # embed_test,
                embed_test_lgcn,
                # embed_question,
                embed_question_lgcn,
                embed_tag,
                embed_big,
                embed_mid,
                embed_problem,
                embed_month,
                embed_dayname
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
            test_sum.unsqueeze(2),
            tag_mean.unsqueeze(2),
            tag_std.unsqueeze(2),
            tag_sum.unsqueeze(2),
            big_mean.unsqueeze(2),
            big_std.unsqueeze(2),
            big_sum.unsqueeze(2),
            ],
            2,
        )
        
        embed_num = embed_num.type(torch.FloatTensor).to(device)

        embed_num = self.num_proj(embed_num)
        
        X = torch.cat([embed_cat, embed_num], 2)
        
        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0] # 마지막 레이어의

        out = out.contiguous().view(batch_size, -1, self.hidden_dim) # 마지막 값만 뽑아서

        out = self.fc(out).view(batch_size, -1) # loss 계산
        return out
