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
    # use sine positional embeddinds
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
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.hidden_dim = self.args.hidden_dim

        ######## 신나는 Embedding ########
        # interaction은 현재 correct로 구성되어있다. correct(1, 2) + padding(0)
        self.embedding_interaction = nn.Embedding(3, self.hidden_dim // 3)
        self.embedding_test = nn.Embedding(self.args.n_test + 1, self.hidden_dim // 3)
        self.embedding_question = nn.Embedding(self.args.n_questions + 1, self.hidden_dim // 3)
        self.embedding_tag = nn.Embedding(self.args.n_tag + 1, self.hidden_dim // 3)
        # self.embedding_new_feature = nn.Embedding(self.args.n_new_feature + 1, self.hidden_dim // 3)

        ######## positioal Embedding ########
        ## sin, cos positional embedding
        # self.embedding_pos = get_sinusoid_encoding_table(args.max_seq_len, self.hidden_dim)
        # self.embedding_pos =  torch.FloatTensor(self.embedding_pos).to(args.device)
        ## arange positional embedding
        self.embedding_pos = get_pos(args.max_seq_len).to(args.device)
        self.comb_proj = nn.Linear((self.hidden_dim // 3) * 4, self.hidden_dim) # 원하는 차원으로 줄이기
        #new_feature : self.comb_proj = nn.Linear((self.hidden_dim // 3) * 5, self.hidden_dim) # 원하는 차원으로 줄이기

        ######## query, key, value ########
        self.query = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.key = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)
        self.value = nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim)

        ######## multihead attention(여기선 head를 1로 두었다.) ########
        self.attn = nn.MultiheadAttention( embed_dim= self.hidden_dim, num_heads= 1, batch_first = True, dropout=0.1)     # multihead attention    ## todo add dropout, LayerNORM
        
        ######## lstm ########
        self.lstm = nn.LSTM(input_size= self.hidden_dim, hidden_size = self.hidden_dim, num_layers=1, batch_first = True)

        ######## layer norm ########
        self.layer_norm1 = nn.LayerNorm(self.hidden_dim)
        self.layer_norm2 = nn.LayerNorm(self.hidden_dim)
        
        ######## feed-forward ########
        self.ffn = Feed_Forward_block(self.hidden_dim, 4*self.hidden_dim)  
        
        ######## fully connect ########
        self.fc = nn.Linear(in_features=self.hidden_dim, out_features=1)
       
        self.activation = nn.Sigmoid()

    def init_hidden(self, batch_size):
        h = torch.zeros(
            # self.args.n_layers,
            1,
            batch_size,
            self.args.hidden_dim)
        h = h.to(self.args.device)

        c = torch.zeros(
            # self.args.n_layers,
            1,
            batch_size,
            self.args.hidden_dim)
        c = c.to(self.args.device)

        return (h, c)

    def forward(self, input):
        test, question, tag, _, mask, interaction = input #(test, question, tag, correct, mask, interaction)
        # test, question, tag, _, mask, interaction, new_feature = input
        batch_size = interaction.size(0) #(64, 20)

        ######## Embedding ########
        embed_test = self.embedding_test(test)                #shape = (64,20,21)
        embed_question = self.embedding_question(question)
        embed_tag = self.embedding_tag(tag) 
        embed_interaction = self.embedding_interaction(interaction) #interaction의 값은 0/1/2 중 하나이다.
        # embed_new_feature = self.embedding_new_feature(new_feature)

        embed = torch.cat(
            [
                embed_interaction,
                embed_test,
                embed_question,
                embed_tag,
                # embed_new_feature,
            ],
            2,
        )

        embed = self.comb_proj(embed) #64,20,64
        embed = embed + self.embedding_pos #(64,20,64) (batch,seq,dim)
        # embed = nn.Dropout(0.1)(embed)

        ######## Encoder ########
        q = self.query(embed)[:, -1:, :]#.permute(1, 0, 2)
        k = self.key(embed)#.permute(1, 0, 2)
        v = self.value(embed)#.permute(1, 0, 2)

        # attention seq, batch, emb
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

'''
원본코드 참고용 (Riid 대회 1등 솔루션)
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import numpy as np
import copy

"""
Encoder --> LSTM --> dense
"""

class Feed_Forward_block(nn.Module):
    """
    out =  Relu( M_out*w1 + b1) *w2 + b2
    """
    def __init__(self, dim_ff):
        super().__init__()
        self.layer1 = nn.Linear(in_features=dim_ff , out_features=dim_ff)
        self.layer2 = nn.Linear(in_features=dim_ff , out_features=dim_ff)

    def forward(self,ffn_in):
        return  self.layer2(   F.relu( self.layer1(ffn_in) )   )


class last_query_model(nn.Module):
    """
    Embedding --> MLH --> LSTM
    """
    def __init__(self , dim_model, heads_en, total_ex ,total_cat, total_in,seq_len, use_lstm=True):
        super().__init__()
        self.seq_len = seq_len
        self.embd_ex =   nn.Embedding( total_ex , embedding_dim = dim_model )                   # embedings  q,k,v = E = exercise ID embedding, category embedding, and positionembedding.
        self.embd_cat =  nn.Embedding( total_cat, embedding_dim = dim_model )
        self.embd_in   = nn.Embedding(  total_in , embedding_dim = dim_model )                  #positional embedding

        self.multi_en = nn.MultiheadAttention( embed_dim= dim_model, num_heads= heads_en,dropout=0.1  )     # multihead attention    ## todo add dropout, LayerNORM
        self.ffn_en = Feed_Forward_block( dim_model )                                            # feedforward block     ## todo dropout, LayerNorm
        self.layer_norm1 = nn.LayerNorm( dim_model )
        self.layer_norm2 = nn.LayerNorm( dim_model )

        self.use_lstm = use_lstm
        if self.use_lstm:
            self.lstm = nn.LSTM(input_size= dim_model, hidden_size= dim_model , num_layers=1)

        self.out = nn.Linear(in_features= dim_model , out_features=1)

    def forward(self, in_ex, in_cat, in_in, first_block=True):
        first_block = True
        if first_block:
            in_ex = self.embd_ex( in_ex )
            in_ex = nn.Dropout(0.1)(in_ex)

            in_cat = self.embd_cat( in_cat )
            in_cat = nn.Dropout(0.1)(in_cat)

            #print("response embedding ", in_in.shape , '\n' , in_in[0])
            in_in = self.embd_in(in_in)
            in_in = nn.Dropout(0.1)(in_in)

            #in_pos = self.embd_pos( in_pos )
            #combining the embedings
            out = in_ex + in_cat + in_in #+ in_pos                      # (b,n,d)

        else:
            out = in_ex
        
        #in_pos = get_pos(self.seq_len)
        #in_pos = self.embd_pos( in_pos )
        #out = out + in_pos                                      # Applying positional embedding

        out = out.permute(1,0,2)                                # (n,b,d)  # print('pre multi', out.shape )
        
        #Multihead attention                            
        n,_,_ = out.shape
        out = self.layer_norm1( out )                           # Layer norm
        skip_out = out 

        out, attn_wt = self.multi_en( out[-1:,:,:] , out , out )         # Q,K,V
        #                        #attn_mask=get_mask(seq_len=n))  # attention mask upper triangular
        #print('MLH out shape', out.shape)
        out = out + skip_out                                    # skip connection

        #LSTM
        if self.use_lstm:
            out,_ = self.lstm( out )                                  # seq_len, batch, input_size
            out = out[-1:,:,:]

        #feed forward
        out = out.permute(1,0,2)                                # (b,n,d)
        out = self.layer_norm2( out )                           # Layer norm 
        skip_out = out
        out = self.ffn_en( out )
        out = out + skip_out                                    # skip connection

        out = self.out( out )

        return out.squeeze(-1)

def get_clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for i in range(N)])

def get_mask(seq_len):
    ##todo add this to device
    return torch.from_numpy( np.triu(np.ones((1 ,seq_len)), k=1).astype('bool'))

def get_pos(seq_len):
    # use sine positional embeddinds
    return torch.arange( seq_len ).unsqueeze(0) 



def random_data(bs, seq_len , total_ex, total_cat, total_in = 2):
    ex = torch.randint( 0 , total_ex ,(bs , seq_len) )
    cat = torch.randint( 0 , total_cat ,(bs , seq_len) )
    res = torch.randint( 0 , total_in ,(bs , seq_len) )
    return ex,cat, res

seq_len = 100
total_ex = 1200
total_cat = 234
total_in = 2
in_ex, in_cat, in_in = random_data(64, seq_len , total_ex, total_cat, total_in)
model = last_query_model(dim_model=128,
            heads_en=1,
            total_ex=total_ex,
            total_cat=total_cat,
            seq_len=seq_len,
            total_in=2
            )
outs = model(in_ex, in_cat,in_in)
print('Output lstm shape- ',outs[0].shape)
print(outs.shape)
'''