import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sakt_plus_class import SAINTDataset, SAINTModel, train_epoch

MAX_SEQ = 100
D_MODEL = 256
N_LAYER = 2
BATCH_SIZE = 256
DROPOUT_RATE = 0.1
EPOCHS = 100

train_df = pd.read_csv('/opt/ml/input/data/after_fe_train_test.csv')

train_df['user_correct_answer'].fillna(0,inplace=True)
train_df['user_acc'].fillna(method='bfill',inplace= True)
train_df = train_df.loc[train_df['answerCode'] != -1]

train_df = train_df[['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag', 'user_correct_answer', 
       'user_acc', 'month', 'hour', 'big_category', 'mid_category', 'test_mean', 'tag_mean']]

assessmentItemID_to_idx = {v:k for k,v in enumerate(train_df['assessmentItemID'].unique())}
idx_to_assessmentItemID = {k:v for k,v in enumerate(train_df['assessmentItemID'].unique())}

testId_to_idx = {v:k for k,v in enumerate(train_df['testId'].unique())}
idx_to_testId = {k:v for k,v in enumerate(train_df['testId'].unique())}

train_df['assessmentItemID'] = train_df['assessmentItemID'].map(assessmentItemID_to_idx)
train_df['testId'] = train_df['testId'].map(testId_to_idx)

# -------------------------------------

skills = train_df['assessmentItemID'].unique()
n_skill = len(skills)

train_group = train_df[['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag', 'user_correct_answer', 
       'user_acc', 'month', 'hour', 'big_category', 'mid_category', 'test_mean', 'tag_mean']].groupby('userID').apply(lambda r: (
            r['assessmentItemID'].values,
            r['testId'].values,
            r['answerCode'].values,
            r['KnowledgeTag'].values,
            r['user_correct_answer'].values,
            r['user_acc'].values,
            r['month'].values,
            r['hour'].values,
            r['big_category'].values,
            r['mid_category'].values,
            r['test_mean'].values,
            r['tag_mean'].values))

train_dataset = SAINTDataset(train_group, n_skill)
train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SAINTModel(n_skill, embed_dim= D_MODEL)

## AdamW
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4)
criterion = nn.BCEWithLogitsLoss()
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=2e-3, steps_per_epoch=len(train_dataloader), epochs=EPOCHS
)

model.to(device)
criterion.to(device)

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