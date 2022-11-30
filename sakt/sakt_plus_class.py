import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn

from torch.utils.data import Dataset

MAX_SEQ = 100
D_MODEL = 256
N_LAYER = 2
BATCH_SIZE = 256
DROPOUT_RATE = 0.1
EPOCHS = 100

class SAINTDataset(Dataset):
    def __init__(self, group, n_skill, max_seq=MAX_SEQ):
        super(SAINTDataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill
        self.samples = {}
        
        self.user_ids = []
        for user_id in group.index:
            q, test_ID, qa, kT, uCA, uAcc, month, hour, bC, mC, testM, tagM = group[user_id]
            if len(q) < 2:
                continue
            
            if len(q) > self.max_seq:
                total_questions = len(q)
                initial = total_questions % self.max_seq
                if initial >= 2:
                    self.user_ids.append(f"{user_id}_0")
                    self.samples[f"{user_id}_0"] = (q[:initial], test_ID[:initial], qa[:initial], kT[:initial], uCA[:initial], 
                                                    uAcc[:initial], month[:initial], hour[:initial], bC[:initial], mC[:initial],
                                                    testM[:initial], tagM[:initial])
                for seq in range(total_questions // self.max_seq):
                    self.user_ids.append(f"{user_id}_{seq+1}")
                    start = initial + seq * self.max_seq
                    end = initial + (seq + 1) * self.max_seq
                    self.samples[f"{user_id}_{seq+1}"] = (q[start:end], test_ID[start:end], qa[start:end], kT[start:end], uCA[start:end], 
                                                    uAcc[start:end], month[start:end], hour[start:end], bC[start:end], mC[start:end],
                                                    testM[start:end], tagM[start:end])
            else:
                user_id = str(user_id)
                self.user_ids.append(user_id)
                self.samples[user_id] = (q, test_ID, qa, kT, uCA, uAcc, month, hour, bC, mC, testM, tagM)
    
    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        q_, test_ID_, qa_, kT_, uCA_, uAcc_, month_, hour_, bC_, mC_, testM_, tagM_ = self.samples[user_id]
        seq_len = len(q_)

        ## for zero padding
        # q_ = q_+1
        # pri_exp_ = pri_exp_ + 1
        # res_ = qa_ + 1
        
        q = np.zeros(self.max_seq, dtype=int)
        test_ID = np.zeros(self.max_seq, dtype=int)
        qa = np.zeros(self.max_seq, dtype=int)
        kT = np.zeros(self.max_seq, dtype=int)
        uCA = np.zeros(self.max_seq, dtype=int)
        uAcc = np.zeros(self.max_seq, dtype=int)
        month = np.zeros(self.max_seq, dtype=int)
        hour = np.zeros(self.max_seq, dtype=int)
        bC = np.zeros(self.max_seq, dtype=int)
        mC = np.zeros(self.max_seq, dtype=int)
        testM = np.zeros(self.max_seq, dtype=int)
        tagM = np.zeros(self.max_seq, dtype=int)
        
        if seq_len == self.max_seq:

            q[:] = q_
            test_ID[:] = test_ID_
            qa[:] = qa_
            kT[:] = kT_
            uCA[:] = uCA_
            uAcc[:] = uAcc_
            month[:] = month_
            hour[:] = hour_
            bC[:] = bC_
            mC[:] = mC_
            testM[:] = testM_
            tagM[:] = tagM_
            
        else:
            q[-seq_len:] = q_
            test_ID[-seq_len:] = test_ID_
            qa[-seq_len:] = qa_
            kT[-seq_len:] = kT_
            uCA[-seq_len:] = uCA_
            uAcc[-seq_len:] = uAcc_
            month[-seq_len:] = month_
            hour[-seq_len:] = hour_
            bC[-seq_len:] = bC_
            mC[-seq_len:] = mC_
            testM[-seq_len:] = testM_
            tagM[-seq_len:] = tagM_
        
        target = q[1:]
        test_ID = test_ID[1:]
        kT = kT[1:]        
        uCA = uCA[1:]        
        uAcc = uAcc[1:]        
        month = month[1:]        
        hour = hour[1:]        
        bC = bC[1:]        
        mC = mC[1:]        
        testM = testM[1:]        
        tagM = tagM[1:]        
        label = qa[1:]

        return target, test_ID, kT, uCA, uAcc, month, hour, bC, mC, testM, tagM, label

class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(DROPOUT_RATE)
    
    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)

def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)

def get_sinusoid_encoding_table(n_seq, d_hidn):
    def cal_angle(position, i_hidn):
        return position / np.power(10000, 2 * (i_hidn // 2) / d_hidn)
    def get_posi_angle_vec(position):
        return [cal_angle(position, i_hidn) for i_hidn in range(d_hidn)]

    sinusoid_table = np.array([get_posi_angle_vec(i_seq) for i_seq in range(n_seq)])
    sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # even index sin 
    sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # odd index cos

    return sinusoid_table


class SAINTModel(nn.Module):
    def __init__(self, n_skill, max_seq=MAX_SEQ, embed_dim= 128):
        super(SAINTModel, self).__init__()

        self.n_skill = n_skill
        self.embed_dim = embed_dim

        self.target_embedding = nn.Embedding(9454, self.embed_dim) ## target
        self.test_id_embedding = nn.Embedding(1537, self.embed_dim) ## target
        self.kT_embedding = nn.Embedding(912, self.embed_dim) ## target
        self.uCA_embedding = nn.Embedding(1552, self.embed_dim) ## target
        self.uAcc_embedding = nn.Embedding(196184, self.embed_dim) ## target
        self.month_embedding = nn.Embedding(12, self.embed_dim) ## target
        self.hour_embedding = nn.Embedding(24, self.embed_dim) ## target
        self.bc_embedding = nn.Embedding(9, self.embed_dim) ## target
        self.mC_embedding = nn.Embedding(198, self.embed_dim) ## target
        self.testM_embedding = nn.Embedding(2471, self.embed_dim) ## target
        self.tagM_embedding = nn.Embedding(1708, self.embed_dim) ## target
        # self.pos_embedding = nn.Embedding(max_seq-1, self.embed_dim) ## position
        self.pos_embedding = get_sinusoid_encoding_table(max_seq-1, self.embed_dim)
        self.pos_embedding =  torch.FloatTensor(self.pos_embedding).to('cuda')

        self.transformer = nn.Transformer(nhead=8, d_model = embed_dim, num_encoder_layers= N_LAYER, num_decoder_layers= N_LAYER, dropout = DROPOUT_RATE)

        self.dropout = nn.Dropout(DROPOUT_RATE)
        self.layer_normal = nn.LayerNorm(embed_dim) 
        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)
    
    def forward(self, question, test_ID, kT, uCA, uAcc, month, hour, bC, mC, testM, tagM):

        device = question.device  

        ## embedding layer
        question = self.target_embedding(question)
        test_ID = self.test_id_embedding(test_ID)
        kT = self.kT_embedding(kT)
        uCA = self.uCA_embedding(uCA)
        uAcc = self.uAcc_embedding(uAcc)
        month = self.month_embedding(month)
        hour = self.hour_embedding(hour)
        bC = self.bc_embedding(bC)
        mC = self.mC_embedding(mC)
        testM = self.testM_embedding(testM)
        tagM = self.tagM_embedding(tagM)
        # pos_id = torch.arange(question.size(1)).unsqueeze(0).to(device)
        # pos_id = self.pos_embedding(pos_id)
        pos_id = self.pos_embedding
        
        enc = question + pos_id
        dec = pos_id + test_ID + kT + uCA + uAcc + month + hour + bC + mC + testM + tagM       

        enc = enc.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
        dec = dec.permute(1, 0, 2)
        mask = future_mask(enc.size(0)).to(device)

        att_output = self.transformer(enc, dec, src_mask=mask, tgt_mask=mask, memory_mask = mask)
        att_output = self.layer_normal(att_output)
        att_output = att_output.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]

        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)

        return x.squeeze(-1)

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device="cuda"):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    for item in dataloader:
        q = item[0].to(device).long()
        test_id = item[1].to(device).long()
        label = item[2].to(device).float()
        kT = item[3].to(device).long()
        uCA = item[4].to(device).long()
        uAcc = item[5].to(device).long()
        month = item[6].to(device).long()
        hour = item[7].to(device).long()
        bC = item[8].to(device).long()
        mC = item[9].to(device).long()
        testM = item[10].to(device).long()
        tagM = item[11].to(device).long()
        target_mask = (q != 0)

        optimizer.zero_grad()
        output = model(q, test_id, kT, uCA, uAcc, month, hour, bC, mC, testM, tagM)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        scheduler.step()
        train_loss.append(loss.item())

        output = torch.masked_select(output, target_mask)
        label = torch.masked_select(label, target_mask)
        pred = (torch.sigmoid(output) >= 0.5).long()
        
        num_corrects += (pred == label).sum().item()
        num_total += len(label)

        labels.extend(label.view(-1).data.cpu().numpy())
        outs.extend(output.view(-1).data.cpu().numpy())

    acc = num_corrects / num_total
    auc = roc_auc_score(labels, outs)
    loss = np.mean(train_loss)

    return loss, acc, auc