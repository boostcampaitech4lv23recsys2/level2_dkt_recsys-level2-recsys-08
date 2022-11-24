import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sakt_class import SAKTDataset, SAKTModel, train_epoch, valid_epoch

# Hyper Parameters   -----------------------------

TRAIN_SAMPLES = 5953 # # 전체 유저 수 : 7442, 전체의 약 80%
MAX_SEQ = 100
MIN_SAMPLES = 5
EMBED_DIM = 128
DROPOUT_RATE = 0.2
LEARNING_RATE = 1e-3
MAX_LEARNING_RATE = 2e-3
TRAIN_BATCH_SIZE = 2048
EPOCHS = 30

# -------------------------------------------------


# Data load
train_df = pd.read_csv('/opt/ml/input/data/train_data.csv')
test_df = pd.read_csv('/opt/ml/input/data/test_data.csv')

ref_train_df = train_df.copy()
ref_test_df = test_df.copy()

ref_test_df = ref_test_df[ref_test_df['answerCode'] != -1]
ref_train_df = pd.concat([ref_train_df, ref_test_df])

# train data

problems = ref_train_df['assessmentItemID'].unique()
train_num_problems = len(problems)

t_assessmentItemID_to_idx = {v:k for k,v in enumerate(ref_train_df['assessmentItemID'].unique())}
t_idx_to_assessmentItemID = {k:v for k,v in enumerate(ref_train_df['assessmentItemID'].unique())}


ref_train_df['assessmentItemID'] = ref_train_df['assessmentItemID'].map(t_assessmentItemID_to_idx)

# 현재 'userID', 'assessmentItemID', 'answerCode'만을 사용
group = ref_train_df[['userID', 'assessmentItemID', 'answerCode']].groupby('userID').apply(lambda r: (
            r['assessmentItemID'].values,
            r['answerCode'].values))

# -------------------------------------------------


# Train / Valid split ----------------------
train_indexes = list(group.index)[:TRAIN_SAMPLES]
valid_indexes = list(group.index)[TRAIN_SAMPLES:]
train_group = group[group.index.isin(train_indexes)]
valid_group = group[group.index.isin(valid_indexes)]
del group, train_indexes, valid_indexes
# -------------------------------------------------


train_dataset = SAKTDataset(train_group, train_num_problems, min_samples=MIN_SAMPLES, max_seq=MAX_SEQ)
train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, num_workers=8)

valid_dataset = SAKTDataset(valid_group, train_num_problems, max_seq=MAX_SEQ)
valid_dataloader = DataLoader(valid_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_ = SAKTModel(train_num_problems, max_seq=MAX_SEQ, embed_dim=EMBED_DIM, dropout_rate=DROPOUT_RATE)

# optimizer = torch.optim.SGD(model.parameters(), lr=1e-3, momentum=0.99, weight_decay=0.005)
optimizer_ = torch.optim.Adam(model_.parameters(), lr=LEARNING_RATE)
criterion_ = nn.BCEWithLogitsLoss() # Binary Cross Entropy Loss에 Sigmod가 결합된 형태

scheduler_ = torch.optim.lr_scheduler.OneCycleLR(
    optimizer_, max_lr=MAX_LEARNING_RATE, steps_per_epoch=len(train_dataloader), epochs=EPOCHS
)

model_.to(device)
criterion_.to(device)

best_auc = 0
over_fit = 0
last_auc = 0
for epoch in range(EPOCHS):
    print(f'------- Epoch {epoch} ---------')
    train_loss, train_acc, train_auc = train_epoch(model_, train_dataloader, optimizer_, scheduler_,criterion_)
    print('\t Train')
    print("\tepoch - {} train_loss - {:.2f} acc - {:.3f} auc - {:.3f}".format(epoch, train_loss, train_acc, train_auc))
    
    print(f'\t Validation')
    val_loss, avl_acc, val_auc = valid_epoch(model_, valid_dataloader, criterion_)
    print("\tepoch - {} val_loss - {:.2f} acc - {:.3f} auc - {:.3f}".format(epoch, val_loss, avl_acc, val_auc))
    
    if val_auc > best_auc:
        best_auc = val_auc
        torch.save(model_.state_dict(), 'very_1st_sakt_model.pt')
        over_fit = 0
    else:
        over_fit += 1
            
    if over_fit >= 3: # 3번 동안 auc 향상이 없으면 eearly stop
        print("early stop epoch ", epoch)
        break