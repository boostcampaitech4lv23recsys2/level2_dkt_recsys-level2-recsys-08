import numpy as np
import pandas as pd

from sklearn.metrics import roc_auc_score

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

train_df = pd.read_pickle('/opt/ml/input/data/after_fe_train_test.pkl')

assessmentItemID_to_idx = {v:k for k,v in enumerate(train_df['assessmentItemID'].unique())}
idx_to_assessmentItemID = {k:v for k,v in enumerate(train_df['assessmentItemID'].unique())}

testId_to_idx = {v:k for k,v in enumerate(train_df['testId'].unique())}
idx_to_testId = {k:v for k,v in enumerate(train_df['testId'].unique())}

train_df['assessmentItemID'] = train_df['assessmentItemID'].map(assessmentItemID_to_idx)
train_df['testId'] = train_df['testId'].map(testId_to_idx)

train_df['user_correct_answer'].fillna(0,inplace=True)
train_df['user_acc'].fillna(method='bfill',inplace= True)
train_df = train_df.loc[train_df['answerCode'] != -1]


MAX_SEQ = 100
D_MODEL = 256
N_LAYER = 2
BATCH_SIZE = 256
DROPOUT = 0.1
EPOCHS = 100
TIME_CAT_FLAG = True

train_df = train_df[['userID', 'assessmentItemID', 'testId', 'answerCode', 
       'KnowledgeTag', 'user_correct_answer', 'user_total_answer', 'user_acc',
       'month', 'day', 'hour', 'big_category', 'problem_num',
       'mid_category', 'test_mean', 'test_std', 'test_sum', 'tag_mean',
       'tag_std', 'tag_sum', 'time_category', 'solvesec_3600']]

skills = train_df['assessmentItemID'].unique()
n_skill = len(skills)
parts = train_df['testId'].unique()
n_part = len(parts)

group = train_df[['userID', 'assessmentItemID', 'testId', 'answerCode', 
       'KnowledgeTag', 'user_correct_answer', 'user_total_answer', 'user_acc',
       'month', 'day', 'hour', 'big_category', 'problem_num',
       'mid_category', 'test_mean', 'test_std', 'test_sum', 'tag_mean',
       'tag_std', 'tag_sum', 'time_category', 'solvesec_3600']].groupby('userID').apply(lambda r: (
            r['assessmentItemID'].values,
            r['testId'].values,
            r['answerCode'].values,
            r['KnowledgeTag'].values,
            r['user_correct_answer'].values,
            r['user_total_answer'].values,
            r['user_acc'].values,
            r['month'].values,
            r['day'].values,
            r['hour'].values,
            r['big_category'].values,
            r['problem_num'].values,
            r['mid_category'].values,
            r['test_mean'].values,
            r['test_std'].values,
            r['test_sum'].values,
            r['tag_mean'].values,
            r['tag_std'].values,
            r['tag_sum'].values,
            r['time_category'].values,
            r['solvesec_3600'].values))

train_indexes = list(group.index)[:int(len(train_df)*0.8)]
valid_indexes = list(group.index)[int(len(train_df)*0.8):]
train_group = group[group.index.isin(train_indexes)]
valid_group = group[group.index.isin(valid_indexes)]

class SAINTDataset(Dataset):
    def __init__(self, group, n_skill, max_seq=MAX_SEQ):
        super(SAINTDataset, self).__init__()
        self.max_seq = max_seq
        self.n_skill = n_skill
        self.samples = {}
        
        self.user_ids = []
        for user_id in group.index:
            assessmentItemID, testId, answerCode, KnowledgeTag, user_correct_answer, user_total_answer, user_acc, \
            month, day, hour, big_category, problem_num, \
            mid_category, test_mean, test_std, test_sum, tag_mean,\
            tag_std, tag_sum, time_category, solvesec_3600 = group[user_id]
            if len(assessmentItemID) < 2:
                continue
            
            if len(assessmentItemID) > self.max_seq:
                total_questions = len(assessmentItemID)
                initial = total_questions % self.max_seq
                if initial >= 2:
                    self.user_ids.append(f"{user_id}_0")
                    self.samples[f"{user_id}_0"] = (assessmentItemID[:initial], testId[:initial], answerCode[:initial], KnowledgeTag[:initial], user_correct_answer[:initial], 
                                                    user_total_answer[:initial], user_acc[:initial], month[:initial], day[:initial], hour[:initial], big_category[:initial], problem_num[:initial], 
                                                    mid_category[:initial], test_mean[:initial], test_std[:initial], test_sum[:initial], tag_mean[:initial],
                                                    tag_std[:initial], tag_sum[:initial], time_category[:initial], solvesec_3600[:initial])
                for seq in range(total_questions // self.max_seq):
                    self.user_ids.append(f"{user_id}_{seq+1}")
                    start = initial + seq * self.max_seq
                    end = initial + (seq + 1) * self.max_seq
                    self.samples[f"{user_id}_{seq+1}"] = (assessmentItemID[start:end], testId[start:end], answerCode[start:end], KnowledgeTag[start:end], user_correct_answer[start:end], 
                                                    user_total_answer[start:end], user_acc[start:end], month[start:end], day[start:end], hour[start:end], big_category[start:end], problem_num[start:end], 
                                                    mid_category[start:end], test_mean[start:end], test_std[start:end], test_sum[start:end], tag_mean[start:end],
                                                    tag_std[start:end], tag_sum[start:end], time_category[start:end], solvesec_3600[start:end])
            else:
                user_id = str(user_id)
                self.user_ids.append(user_id)
                self.samples[user_id] = (assessmentItemID, testId, answerCode, KnowledgeTag, 
                                        user_correct_answer, user_total_answer, user_acc,
                                        month, day, hour, big_category, problem_num, 
                                        mid_category, test_mean, test_std, test_sum, tag_mean,
                                        tag_std, tag_sum, time_category, solvesec_3600)
    
    def __len__(self):
        return len(self.user_ids)

    def __getitem__(self, index):
        user_id = self.user_ids[index]
        assessmentItemID_, testId_, answerCode_, KnowledgeTag_, user_correct_answer_, user_total_answer_, user_acc_, \
            month_, day_, hour_, big_category_, problem_num_, \
            mid_category_, test_mean_, test_std_, test_sum_, tag_mean_,\
            tag_std_, tag_sum_, time_category_, solvesec_3600_ = self.samples[user_id]
        seq_len = len(assessmentItemID_)

        ## for zero padding
        # assessmentItemID_ = assessmentItemID_+1
        # pri_exp_ = pri_exp_ + 1
        # res_ = answerCode_ + 1

        # res = np.zeros(self.max_seq, dtype=int) 
        assessmentItemID = np.zeros(self.max_seq, dtype=int) 
        testId = np.zeros(self.max_seq, dtype=int) 
        answerCode = np.zeros(self.max_seq, dtype=int)
        KnowledgeTag = np.zeros(self.max_seq, dtype=int) 
        user_correct_answer = np.zeros(self.max_seq, dtype=int) 
        user_total_answer = np.zeros(self.max_seq, dtype=int) 
        user_acc = np.zeros(self.max_seq, dtype=int) 
        month = np.zeros(self.max_seq, dtype=int) 
        day = np.zeros(self.max_seq, dtype=int) 
        hour = np.zeros(self.max_seq, dtype=int) 
        big_category = np.zeros(self.max_seq, dtype=int) 
        problem_num = np.zeros(self.max_seq, dtype=int) 
        mid_category = np.zeros(self.max_seq, dtype=int) 
        test_mean = np.zeros(self.max_seq, dtype=int) 
        test_std = np.zeros(self.max_seq, dtype=int) 
        test_sum = np.zeros(self.max_seq, dtype=int) 
        tag_mean = np.zeros(self.max_seq, dtype=int)
        tag_std = np.zeros(self.max_seq, dtype=int) 
        tag_sum = np.zeros(self.max_seq, dtype=int) 
        time_category = np.zeros(self.max_seq, dtype=int) 
        solvesec_3600 = np.zeros(self.max_seq, dtype=int) 
        
        if seq_len == self.max_seq:

            # res[:] = res_
            assessmentItemID[:] = assessmentItemID_
            testId[:] = testId_
            answerCode[:] = answerCode_
            KnowledgeTag[:] = KnowledgeTag_
            user_correct_answer[:] = user_correct_answer_
            user_total_answer[:] = user_total_answer_
            user_acc[:] = user_acc_
            month[:] = month_
            day[:] = day_
            hour[:] = hour_
            big_category[:] = big_category_
            problem_num[:] = problem_num_
            mid_category[:] = mid_category_
            test_mean[:] = test_mean_
            test_std[:] = test_std_
            test_sum[:] = test_sum_
            tag_mean[:] = tag_mean_
            tag_std[:] = tag_std_
            tag_sum[:] = tag_sum_
            time_category[:] = time_category_
            solvesec_3600[:] = solvesec_3600_
            
        else:
            # res[-seq_len:] = res_
            assessmentItemID[-seq_len:] = assessmentItemID_
            testId[-seq_len:] = testId_
            answerCode[-seq_len:] = answerCode_
            KnowledgeTag[-seq_len:] = KnowledgeTag_
            user_correct_answer[-seq_len:] = user_correct_answer_
            user_total_answer[-seq_len:] = user_total_answer_
            user_acc[-seq_len:] = user_acc_
            month[-seq_len:] = month_
            day[-seq_len:] = day_
            hour[-seq_len:] = hour_
            big_category[-seq_len:] = big_category_
            problem_num[-seq_len:] = problem_num_
            mid_category[-seq_len:] = mid_category_
            test_mean[-seq_len:] = test_mean_
            test_std[-seq_len:] = test_std_
            test_sum[-seq_len:] = test_sum_
            tag_mean[-seq_len:] = tag_mean_
            tag_std[-seq_len:] = tag_std_
            tag_sum[-seq_len:] = tag_sum_
            time_category[-seq_len:] = time_category_
            solvesec_3600[-seq_len:] = solvesec_3600_
        
        # respon = res_
        target_qids = assessmentItemID[1:]
        label = answerCode[1:]
        part = testId[1:]

        KnowledgeTag = KnowledgeTag[:-1]
        answerCode = answerCode[:-1]
        user_correct_answer = user_correct_answer[:-1]
        user_total_answer = user_total_answer[:-1]
        user_acc = user_acc[:-1]
        month = month[:-1]
        day = day[:-1]
        hour = hour[:-1]
        big_category = big_category[:-1]
        problem_num = problem_num[:-1]
        mid_category = mid_category[:-1]
        test_mean = test_mean[:-1]
        test_std = test_std[:-1]
        test_sum = test_sum[:-1]
        tag_mean = tag_mean[:-1]
        tag_std = tag_std[:-1]
        tag_sum = tag_sum[:-1]
        time_category = time_category[:-1]
        solvesec_3600 = solvesec_3600[:-1]

        inputs = {
        'answerCode': answerCode,
        'KnowledgeTag': KnowledgeTag,
        'user_correct_answer': user_correct_answer,
        'user_total_answer': user_total_answer,
        'user_acc': user_acc,
        'month': month,
        'day': day,
        'hour': hour,
        'big_category': big_category,
        'problem_num': problem_num,
        'mid_category': mid_category,
        'test_mean': test_mean,
        'test_std': test_std,
        'test_sum': test_sum,
        'tag_mean': tag_mean,
        'tag_std': tag_std,
        'tag_sum': tag_sum,
        'time_category': time_category,
        'solvesec_3600': solvesec_3600
        }

        return inputs, target_qids, part, label

train_dataset = SAINTDataset(train_group, n_skill)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)


val_dataset = SAINTDataset(valid_group, n_skill)
val_dataloader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FFN(nn.Module):
    def __init__(self, state_size=200):
        super(FFN, self).__init__()
        self.state_size = state_size

        self.lr1 = nn.Linear(state_size, state_size)
        self.relu = nn.ReLU()
        self.lr2 = nn.Linear(state_size, state_size)
        self.dropout = nn.Dropout(DROPOUT)
    
    def forward(self, x):
        x = self.lr1(x)
        x = self.relu(x)
        x = self.lr2(x)
        return self.dropout(x)

def future_mask(seq_length):
    future_mask = np.triu(np.ones((seq_length, seq_length)), k=1).astype('bool')
    return torch.from_numpy(future_mask)

class SAINTModel(nn.Module):
    def __init__(self, n_skill, n_part, max_seq=MAX_SEQ, embed_dim= 128):
        super(SAINTModel, self).__init__()

        self.n_skill = n_skill
        self.embed_dim = embed_dim
        self.n_cat = n_part

        self.target_embedding = nn.Embedding(self.n_skill+1, embed_dim) 
        self.c_embedding = nn.Embedding(self.n_cat+1, embed_dim) 
        self.pos_embedding = nn.Embedding(max_seq-1, embed_dim) ## position

        self.test_id_embedding = nn.Embedding(1537+1, embed_dim) 
        self.kT_embedding = nn.Embedding(912+1, embed_dim) 
        self.uCA_embedding = nn.Embedding(1552+1, embed_dim) 
        self.uTA_embedding = nn.Embedding(1860+1, embed_dim) 
        self.uAcc_embedding = nn.Embedding(196184+1, embed_dim) 
        self.month_embedding = nn.Embedding(12+1, embed_dim) 
        self.day_embedding = nn.Embedding(31+1, embed_dim) 
        self.hour_embedding = nn.Embedding(24+1, embed_dim) 
        self.bC_embedding = nn.Embedding(9+1, embed_dim) 
        self.problem_num_embedding = nn.Embedding(13+1, embed_dim) 
        self.mC_embedding = nn.Embedding(198+1, embed_dim) 
        self.test_m_embedding = nn.Embedding(2471+1, embed_dim) 
        self.test_std_embedding = nn.Embedding(3023+1, embed_dim) 
        self.test_sum_embedding = nn.Embedding(1030+1, embed_dim) 
        self.tag_m_embedding = nn.Embedding(1708+1, embed_dim) 
        self.tag_std_embedding = nn.Embedding(1805+1, embed_dim) 
        self.tag_sum_embedding = nn.Embedding(1058+1, embed_dim) 
        self.timeC_embedding = nn.Embedding(10+1, embed_dim) 
        self.solvesec_3600_embedding = nn.Embedding(3598+1, embed_dim) 
        

        # self.transformer = nn.Transformer(nhead=8, d_model = embed_dim, num_encoder_layers= N_LAYER, num_decoder_layers= N_LAYER, dropout = DROPOUT)

        self.multi_att = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=8, dropout=DROPOUT)

        self.dropout = nn.Dropout(DROPOUT)
        self.layer_normal = nn.LayerNorm(embed_dim) 
        self.ffn = FFN(embed_dim)
        self.pred = nn.Linear(embed_dim, 1)
    
    def forward(self, assessmentItemID, testId, KnowledgeTag, user_correct_answer, user_total_answer, user_acc, \
            month, day, hour, big_category, problem_num, \
            mid_category, test_mean, test_std, test_sum, tag_mean,\
            tag_std, tag_sum, time_category, solvesec_3600):

        # device = question.device  

        ## embedding layer

        assessmentItemID = self.target_embedding(assessmentItemID)
        KnowledgeTag = self.c_embedding(KnowledgeTag)
        pos_id = torch.arange(assessmentItemID.size(1)).unsqueeze(0).to(device)
        pos_id = self.pos_embedding(pos_id)

        test_id = self.test_id_embedding(testId)
        # kT = self.kT_embedding(kT)
        user_correct_answer = self.uCA_embedding(user_correct_answer)
        user_total_answer = self.uTA_embedding(user_total_answer)
        user_acc = self.uAcc_embedding(user_acc)
        month = self.month_embedding(month)
        day = self.day_embedding(day)
        hour = self.hour_embedding(hour)
        big_category = self.bC_embedding(big_category)
        problem_num = self.problem_num_embedding(problem_num)
        mid_category = self.mC_embedding(mid_category) 
        test_mean = self.test_m_embedding(test_mean)
        test_std = self.test_std_embedding(test_std)
        test_sum = self.test_sum_embedding(test_sum)
        tag_mean = self.tag_m_embedding(tag_mean)
        tag_std = self.tag_std_embedding(tag_std)
        tag_sum = self.tag_sum_embedding(tag_sum)
        time_category = self.timeC_embedding(time_category)
        solvesec_3600 = self.solvesec_3600_embedding(solvesec_3600)
        
        
        enc = (assessmentItemID + KnowledgeTag + pos_id + test_id + user_correct_answer + user_total_answer + user_acc
                + month + day + hour + big_category + problem_num + mid_category + test_mean + test_std + test_sum + tag_mean
                + tag_std + tag_sum + time_category + solvesec_3600)

        enc = enc.permute(1, 0, 2) # x: [bs, s_len, embed] => [s_len, bs, embed]
        # dec = dec.permute(1, 0, 2)
        mask = future_mask(enc.size(0)).to(device)

        att_output, att_weight = self.multi_att(enc, attn_mask=mask)
        # att_output = self.transformer(enc, dec, src_mask=mask, tgt_mask=mask, memory_mask = mask)
        att_output = self.layer_normal(att_output)
        att_output = att_output.permute(1, 0, 2) # att_output: [s_len, bs, embed] => [bs, s_len, embed]

        x = self.ffn(att_output)
        x = self.layer_normal(x + att_output)
        x = self.pred(x)

        return x.squeeze(-1)

model = SAINTModel(n_skill, n_part, embed_dim= D_MODEL)

## AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=2e-3, steps_per_epoch=len(train_dataloader), epochs=EPOCHS
)

model.to(device)
criterion.to(device)

def train_epoch(model, dataloader, optimizer, scheduler, criterion, device="cuda"):
    model.train()

    train_loss = []
    num_corrects = 0
    num_total = 0
    labels = []
    outs = []

    for inputs, tar_qids, part, label in dataloader:

        assessmentItemID = tar_qids.to(device).long()
        testId = part.to(device).long()
        label = label.to(device).long()

        KnowledgeTag = inputs['KnowledgeTag'].to(device).long()
        user_correct_answer = inputs['user_correct_answer'].to(device).long()
        user_total_answer = inputs['user_total_answer'].to(device).long()
        user_acc = inputs['user_acc'].to(device).long()
        month = inputs['month'].to(device).long()
        day = inputs['day'].to(device).long()
        hour = inputs['hour'].to(device).long()
        big_category = inputs['big_category'].to(device).long()
        problem_num = inputs['problem_num'].to(device).long()
        mid_category = inputs['mid_category'].to(device).long()
        test_mean = inputs['test_mean'].to(device).long()
        test_std = inputs['test_std'].to(device).long()
        test_sum = inputs['test_sum'].to(device).long()
        tag_mean = inputs['tag_mean'].to(device).long()
        tag_std = inputs['tag_std'].to(device).long()
        tag_sum = inputs['tag_sum'].to(device).long()
        time_category = inputs['time_category'].to(device).long()
        solvesec_3600 = inputs['solvesec_3600'].to(device).long()

        target_mask = (assessmentItemID != 0)

        optimizer.zero_grad()
        output = model(assessmentItemID, testId, KnowledgeTag, user_correct_answer, user_total_answer, user_acc,
            month, day, hour, big_category, problem_num,
            mid_category, test_mean, test_std, test_sum, tag_mean,\
            tag_std, tag_sum, time_category, solvesec_3600)
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

best_auc = 0
over_fit = 0
last_auc = 0

for epoch in range(EPOCHS):
    print(f'------- Epoch {epoch} ---------')
    train_loss, train_acc, train_auc = train_epoch(model, train_dataloader, optimizer, scheduler,criterion)
    print('\t Train')
    print("\tepoch - {} train_loss - {:.2f} acc - {:.3f} auc - {:.3f}".format(epoch, train_loss, train_acc, train_auc))
        
    if train_auc > best_auc:
        best_auc = train_auc
        torch.save(model.state_dict(), '2nd_sakt_model.pt')
        over_fit = 0
    else:
        over_fit += 1
            
    if over_fit >= 3: # 3번 동안 auc 향상이 없으면 eearly stop
        print("early stop epoch ", epoch)
        break