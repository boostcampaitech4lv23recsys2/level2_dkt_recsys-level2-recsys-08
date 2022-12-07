import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import copy

def get_sinusoid_encoding_table(n_seq, d_hidn):
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

    return sinusoid_table

def get_pos(seq_len):
    return torch.arange( seq_len ).unsqueeze(0).unsqueeze(2)


class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, hidden_dim, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=hidden_dim , out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff , out_features=hidden_dim)

    def forward(self,ffn_in):
        return  self.layer2(F.relu(self.layer1(ffn_in)))

class LQTransformer(nn.Module):
    def init_hidden(self, batch_size):
        h = torch.zeros(
            self.args.n_layers,
            # 1,
            batch_size,
            self.hidden_dim)
        h = h.to(self.args.device)

        c = torch.zeros(
            self.args.n_layers,
            # 1,
            batch_size,
            self.hidden_dim)
        c = c.to(self.args.device)

        return (h, c)

    def __init__(self, args):
        super().__init__()

        self.args = args
        self.hidden_dim = self.args.hidden_dim
        self.third_hidden_dim = self.hidden_dim // 3
        self.cat_cols = args.used_cat_cols
        self.num_cols = args.used_num_cols


        ######## positioal Embedding ########
        ## sin, cos positional embedding
        # self.embedding_pos = get_sinusoid_encoding_table(args.max_seq_len, self.hidden_dim)
        # self.embedding_pos =  torch.FloatTensor(self.embedding_pos).to(args.device)
        ## arange positional embedding
        self.embedding_pos = get_pos(args.max_seq_len).to(args.device)

        ######## query, key, value ########
        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        ######## multihead attention(여기선 head를 1로 두었다.) ########
        self.attn = nn.MultiheadAttention( embed_dim= self.hidden_dim, num_heads= 1, batch_first = True, dropout=0.1)     # multihead attention    ## todo add dropout, LayerNORM
        
        ######## lstm ########
        self.lstm = nn.LSTM(input_size= self.hidden_dim, hidden_size = self.hidden_dim, num_layers=self.args.n_layers, batch_first = True)

        ######## layer norm ########
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)
        
        ######## feed-forward ########
        self.ffn = Feed_Forward_block(self.hidden_dim, 6*self.hidden_dim)  
        
        ######## fully connect ########
        # self.fc = nn.Sequential(
        #             nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim),
        #             nn.Linear(in_features=self.hidden_dim, out_features=1))
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=1)
       
        self.activation = nn.Sigmoid()
        
        ######## 신나는 Embedding ########
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        #😘1.FE할 때 여기
        for col in self.cat_cols:
            exec("self.embedding_" + col + '= nn.Embedding(self.args.n_' + col + '+1, self.third_hidden_dim)')
        self.embedding_interaction = nn.Embedding(3, self.third_hidden_dim)
        # self.embedding_testId = nn.Embedding(self.args.n_testId + 1, self.third_hidden_dim)
        
        #😘2.FE할 때 여기
        self.cat_proj = nn.Linear((self.third_hidden_dim) * (len(self.cat_cols)+1), self.hidden_dim//2) # 원하는 차원으로 줄이기
        self.num_proj = nn.Sequential(nn.Linear(len(self.num_cols), self.hidden_dim//2),
                                    nn.LayerNorm(self.hidden_dim//2))

        self.batch_norm = nn.BatchNorm1d(64, affine=True)

    def forward(self, input):

        #😘3.FE할 때 여기
        assessmentItemID, testId, KnowledgeTag, answerCode, \
        big_category, mid_category, problem_num, time_category, solvecumsum_category, \
        solvesec_3600, solvesec_cumsum, test_mean, test_std, \
        tag_mean, tag_std, big_mean, big_std, big_sum, assess_mean, assess_std, \
        user_mean, user_std, user_sum, assess_count, mask, interaction = input
        # test, question, tag, _, mask, interaction, new_feature = input
        batch_size = interaction.size(0) #(64, 20)

        ######## Embedding ########
        #😘4.FE할 때 여기
        embed_interaction = self.embedding_interaction(interaction.type(torch.cuda.IntTensor))
        embed_assessmentItemID = self.embedding_assessmentItemID(assessmentItemID.type(torch.cuda.IntTensor))
        embed_testId = self.embedding_testId(testId.type(torch.cuda.IntTensor))
        embed_KnowledgeTag = self.embedding_KnowledgeTag(KnowledgeTag.type(torch.cuda.IntTensor))
        embed_big_category = self.embedding_big_category(big_category.type(torch.cuda.IntTensor))
        embed_mid_category = self.embedding_mid_category(mid_category.type(torch.cuda.IntTensor))
        embed_problem_num = self.embedding_problem_num(problem_num.type(torch.cuda.IntTensor))
        embed_time_category = self.embedding_time_category(time_category.type(torch.cuda.IntTensor))
        embed_solvecumsum_category = self.embedding_solvecumsum_category(solvecumsum_category.type(torch.cuda.IntTensor))
        # embed_new_feature = self.embedding_new_feature(new_feature)

        #😘5.FE할 때 여기
        embed_cat = torch.cat(
            [
                embed_interaction,
                embed_assessmentItemID,
                embed_testId,
                embed_KnowledgeTag,
                embed_big_category,
                embed_mid_category,
                embed_problem_num,
                embed_time_category,
                embed_solvecumsum_category
            ],
            2,
        )
        embed_cat = self.cat_proj(embed_cat)

        embed_num = [solvesec_3600.type(torch.cuda.FloatTensor).unsqueeze(2),
                     solvesec_cumsum.type(torch.cuda.FloatTensor).unsqueeze(2),
                     test_mean.type(torch.cuda.FloatTensor).unsqueeze(2),
                     test_std.type(torch.cuda.FloatTensor).unsqueeze(2),
                     tag_mean.type(torch.cuda.FloatTensor).unsqueeze(2),
                     tag_std.type(torch.cuda.FloatTensor).unsqueeze(2),
                     big_mean.type(torch.cuda.FloatTensor).unsqueeze(2),
                     big_std.type(torch.cuda.FloatTensor).unsqueeze(2),
                     big_sum.type(torch.cuda.FloatTensor).unsqueeze(2),
                     assess_mean.type(torch.cuda.FloatTensor).unsqueeze(2),
                     assess_std.type(torch.cuda.FloatTensor).unsqueeze(2),
                     user_mean.type(torch.cuda.FloatTensor).unsqueeze(2),
                     user_std.type(torch.cuda.FloatTensor).unsqueeze(2),
                     user_sum.type(torch.cuda.FloatTensor).unsqueeze(2),
                     assess_count.type(torch.cuda.FloatTensor).unsqueeze(2),
                    ]
        embed_num = torch.cat(embed_num, 2)

        embed_num = self.num_proj(embed_num)
        embed_num = self.batch_norm(embed_num.permute(0, 2, 1))
        embed_num = embed_num.permute(0, 2, 1)

        embed = torch.cat([embed_cat, embed_num], 2)

        #========= 여기까지 복사!! ==========#

        embed = embed + self.embedding_pos #(64,20,64) (batch,seq,dim)

        ######## Encoder ########
        q = self.query(embed)[:, -1:, :]#.permute(1, 0, 2)
        k = self.key(embed)#.permute(1, 0, 2)
        v = self.value(embed)#.permute(1, 0, 2)

        # attention
        out, _ = self.attn(q, k, v)

        # residual, layer_norm
        #out = out.permute(1, 0, 2)
        out = embed + out
        out = self.layer_norm1(out)
        out_ = out

        # feed forward network
        out = self.ffn(out)

        # residual, layer_norm
        out = out_ + out
        out = self.layer_norm2(out) #[64,20,64]

        ######## LSTM ########
        hidden = self.init_hidden(batch_size)  #shape = [2, 1, 64, 64]
        out, hidden = self.lstm(out, hidden)   #out shape = [64, 20, 64]

        ######## DNN ########
        out = out.contiguous().view(batch_size, -1, self.hidden_dim)
        out = self.fc(out) #[64, 20, 1]

        preds = self.activation(out).view(batch_size, -1) #[64, 20]

        return preds