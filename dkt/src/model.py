import torch
import torch.nn as nn

try:
    from transformers.modeling_bert import BertConfig, BertEncoder, BertModel
except:
    from transformers.models.bert.modeling_bert import (
        BertConfig,
        BertEncoder,
        BertModel,
    )


class LSTM(nn.Module):
    #new_feature 를 추가하고 싶으시면, embedding 부분만 바꾸면 됩니다.
    #lqtransformer.py의 임베딩 부분을 참고해주세요! -> lqtransformer.py에서 "new_feature" 찾기 기능
    def __init__(self, args):
        super(LSTM, self).__init__()
        self.args = args

        self.hidden_dim = self.args.hidden_dim
        self.n_layers = self.args.n_layers

        self.lstm = nn.LSTM(
            self.hidden_dim, self.hidden_dim, self.n_layers, batch_first=True #nn.LSTM(input_size, hidden_size, num_layers, ...)
        )

        # Fully connected layer
        self.fc = nn.Linear(self.hidden_dim, 1)

        ######## 신나는 Embedding ########
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        #😘1.FE할 때 여기
        
        self.embedding_testId = nn.Embedding(self.args.n_testId + 1, self.hidden_dim // 6)
        self.embedding_assessmentItemID = nn.Embedding(self.args.n_assessmentItemID + 1, self.hidden_dim // 6)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 6)
        self.embedding_big_category = nn.Embedding(self.args.n_big_category + 1, self.hidden_dim // 6)
        self.embedding_mid_category = nn.Embedding(self.args.n_mid_category + 1, self.hidden_dim // 6)
        self.embedding_problem_num = nn.Embedding(self.args.n_problem_num + 1, self.hidden_dim // 6)
        self.embedding_month = nn.Embedding(self.args.n_month + 1, self.hidden_dim // 6)
        self.embedding_dayname = nn.Embedding(self.args.n_dayname + 1, self.hidden_dim // 6)
        self.embedding_KnowledgeTag = nn.Embedding(self.args.n_KnowledgeTag + 1, self.hidden_dim // 6)
    
        self.solvesec_600 = nn.Linear(in_features=1, out_features=self.hidden_dim//6)
        self.big_mean = nn.Linear(in_features=1, out_features=self.hidden_dim//6)
        self.big_std = nn.Linear(in_features=1, out_features=self.hidden_dim//6)
        self.tag_mean = nn.Linear(in_features=1, out_features=self.hidden_dim//6)
        self.tag_std = nn.Linear(in_features=1, out_features=self.hidden_dim//6)
        self.test_mean = nn.Linear(in_features=1, out_features=self.hidden_dim//6)
        self.test_std = nn.Linear(in_features=1, out_features=self.hidden_dim//6)
        self.month_mean = nn.Linear(in_features=1, out_features=self.hidden_dim//6)
        
        #😘2.FE할 때 여기
        self.comb_proj = nn.Linear((self.hidden_dim // 6) * (9+8-2), self.hidden_dim) # 원하는 차원으로 줄이기

    def forward(self, input):
        
        # test, question, tag, _, mask, interaction = input #(test, question, tag, correct, mask, interaction)
        #😘3.FE할 때 여기
        testId, assessmentItemID, big_category, answerCode, mid_category,\
        problem_num,  month, dayname, solvesec_600, \
        KnowledgeTag, big_mean, big_std, tag_mean, tag_std, \
        test_mean, test_std, month_mean, mask, interaction = input
        # test, question, tag, _, mask, interaction, new_feature = input
        batch_size = interaction.size(0) #(64, 20)

        ######## Embedding ########
        #😘4.FE할 때 여기
        
        embed_testId = self.embedding_testId(testId.type(torch.cuda.IntTensor))
        embed_assessmentItemID = self.embedding_assessmentItemID(assessmentItemID.type(torch.cuda.IntTensor))
        embed_big_category = self.embedding_big_category(big_category.type(torch.cuda.IntTensor))                #shape = (64,20,21)
        embed_mid_category = self.embedding_mid_category(mid_category.type(torch.cuda.IntTensor))
        embed_problem_num = self.embedding_problem_num(problem_num.type(torch.cuda.IntTensor)) 
        embed_interaction = self.embedding_interaction(interaction.type(torch.cuda.IntTensor))
        embed_month = self.embedding_month(month.type(torch.cuda.IntTensor))
        embed_dayname = self.embedding_dayname(dayname.type(torch.cuda.IntTensor))
        embed_KnowledgeTag = self.embedding_KnowledgeTag(KnowledgeTag.type(torch.cuda.IntTensor))
        
        # embed_new_feature = self.embedding_new_feature(new_feature)
        #😘5.FE할 때 여기
        embed = torch.cat(
            [
                embed_testId,
                embed_assessmentItemID,
                embed_big_category,
                embed_mid_category,
                embed_problem_num,
                embed_interaction,
                # embed_month,
                embed_dayname,
                embed_KnowledgeTag,

                self.big_mean(big_mean.unsqueeze(2).type(torch.cuda.FloatTensor)),
                self.solvesec_600(solvesec_600.unsqueeze(2).type(torch.cuda.FloatTensor)),
                self.big_std(big_std.unsqueeze(2).type(torch.cuda.FloatTensor)),
                self.tag_mean(solvesec_600.unsqueeze(2).type(torch.cuda.FloatTensor)),
                self.tag_std(tag_std.unsqueeze(2).type(torch.cuda.FloatTensor)),
                self.test_mean(test_mean.unsqueeze(2).type(torch.cuda.FloatTensor)),
                self.test_std(test_std.unsqueeze(2).type(torch.cuda.FloatTensor)),
                # self.month_mean(month_mean.unsqueeze(2).type(torch.cuda.FloatTensor)),
            ],
            2,
        )

        X = self.comb_proj(embed) #(64,20,64)

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

        ######## 신나는 Embedding ########
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        #😘1.FE할 때 여기
        
        self.embedding_testId = nn.Embedding(self.args.n_testId + 1, self.hidden_dim // 6)
        self.embedding_assessmentItemID = nn.Embedding(self.args.n_assessmentItemID + 1, self.hidden_dim // 6)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 6)
        self.embedding_big_category = nn.Embedding(self.args.n_big_category + 1, self.hidden_dim // 6)
        self.embedding_mid_category = nn.Embedding(self.args.n_mid_category + 1, self.hidden_dim // 6)
        self.embedding_problem_num = nn.Embedding(self.args.n_problem_num + 1, self.hidden_dim // 6)
        self.embedding_month = nn.Embedding(self.args.n_month + 1, self.hidden_dim // 6)
        self.embedding_dayname = nn.Embedding(self.args.n_dayname + 1, self.hidden_dim // 6)
        self.embedding_KnowledgeTag = nn.Embedding(self.args.n_KnowledgeTag + 1, self.hidden_dim // 6)
    
        self.solvesec_600 = nn.Linear(in_features=1, out_features=self.hidden_dim//6)
        self.big_mean = nn.Linear(in_features=1, out_features=self.hidden_dim//6)
        self.big_std = nn.Linear(in_features=1, out_features=self.hidden_dim//6)
        self.tag_mean = nn.Linear(in_features=1, out_features=self.hidden_dim//6)
        self.tag_std = nn.Linear(in_features=1, out_features=self.hidden_dim//6)
        self.test_mean = nn.Linear(in_features=1, out_features=self.hidden_dim//6)
        self.test_std = nn.Linear(in_features=1, out_features=self.hidden_dim//6)
        self.month_mean = nn.Linear(in_features=1, out_features=self.hidden_dim//6)
        
        #😘2.FE할 때 여기
        self.comb_proj = nn.Linear((self.hidden_dim // 6) * (9+8-2), self.hidden_dim) # 원하는 차원으로 줄이기

    def forward(self, input):
        
        # test, question, tag, _, mask, interaction = input #(test, question, tag, correct, mask, interaction)
        #😘3.FE할 때 여기
        testId, assessmentItemID, big_category, answerCode, mid_category,\
        problem_num,  month, dayname, solvesec_600, \
        KnowledgeTag, big_mean, big_std, tag_mean, tag_std, \
        test_mean, test_std, month_mean, mask, interaction = input
        # test, question, tag, _, mask, interaction, new_feature = input
        batch_size = interaction.size(0) #(64, 20)

        ######## Embedding ########
        #😘4.FE할 때 여기
        
        embed_testId = self.embedding_testId(testId.type(torch.cuda.IntTensor))
        embed_assessmentItemID = self.embedding_assessmentItemID(assessmentItemID.type(torch.cuda.IntTensor))
        embed_big_category = self.embedding_big_category(big_category.type(torch.cuda.IntTensor))                #shape = (64,20,21)
        embed_mid_category = self.embedding_mid_category(mid_category.type(torch.cuda.IntTensor))
        embed_problem_num = self.embedding_problem_num(problem_num.type(torch.cuda.IntTensor)) 
        embed_interaction = self.embedding_interaction(interaction.type(torch.cuda.IntTensor))
        embed_month = self.embedding_month(month.type(torch.cuda.IntTensor))
        embed_dayname = self.embedding_dayname(dayname.type(torch.cuda.IntTensor))
        embed_KnowledgeTag = self.embedding_KnowledgeTag(KnowledgeTag.type(torch.cuda.IntTensor))
        
        # embed_new_feature = self.embedding_new_feature(new_feature)
        #😘5.FE할 때 여기
        embed = torch.cat(
            [
                embed_testId,
                embed_assessmentItemID,
                embed_big_category,
                embed_mid_category,
                embed_problem_num,
                embed_interaction,
                # embed_month,
                embed_dayname,
                embed_KnowledgeTag,

                self.big_mean(big_mean.unsqueeze(2).type(torch.cuda.FloatTensor)),
                self.solvesec_600(solvesec_600.unsqueeze(2).type(torch.cuda.FloatTensor)),
                self.big_std(big_std.unsqueeze(2).type(torch.cuda.FloatTensor)),
                self.tag_mean(solvesec_600.unsqueeze(2).type(torch.cuda.FloatTensor)),
                self.tag_std(tag_std.unsqueeze(2).type(torch.cuda.FloatTensor)),
                self.test_mean(test_mean.unsqueeze(2).type(torch.cuda.FloatTensor)),
                self.test_std(test_std.unsqueeze(2).type(torch.cuda.FloatTensor)),
                # self.month_mean(month_mean.unsqueeze(2).type(torch.cuda.FloatTensor)),
            ],
            2,
        )

        X = self.comb_proj(embed)

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

        # embedding combination projection
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim)

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

    def forward(self, input):
        test, question, tag, _, mask, interaction = input #(test, question, tag, correct, mask, interaction)
        # test, question, tag, _, mask, interaction, new_feature = input

        batch_size = interaction.size(0)

        # 신나는 embedding

        embed_interaction = self.embedding_interaction(interaction)

        embed_test = self.embedding_test(test)
        embed_question = self.embedding_question(question)

        embed_tag = self.embedding_tag(tag)

        embed = torch.cat( # 여기는 Continuos 없고, 범주형만 존재
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
            ],
            2,
        )

        X = self.comb_proj(embed)

        # Bert
        encoded_layers = self.encoder(inputs_embeds=X, attention_mask=mask)
        out = encoded_layers[0] # 마지막 레이어의

        out = out.contiguous().view(batch_size, -1, self.hidden_dim) # 마지막 값만 뽑아서

        out = self.fc(out).view(batch_size, -1) # loss 계산
        return out
