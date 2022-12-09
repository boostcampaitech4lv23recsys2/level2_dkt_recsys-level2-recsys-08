import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sakt_class import SAKTDataset, SAKTModel, get_output

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
test_df = pd.read_csv('/opt/ml/input/data/test_data.csv')
ref_test_df = test_df.copy()

# train data

problems = ref_test_df['assessmentItemID'].unique()
train_num_problems = len(problems)

t_assessmentItemID_to_idx = {v:k for k,v in enumerate(ref_test_df['assessmentItemID'].unique())}
t_idx_to_assessmentItemID = {k:v for k,v in enumerate(ref_test_df['assessmentItemID'].unique())}


ref_test_df['assessmentItemID'] = ref_test_df['assessmentItemID'].map(t_assessmentItemID_to_idx)

# 현재 'userID', 'assessmentItemID', 'answerCode'만을 사용
group = ref_test_df[['userID', 'assessmentItemID', 'answerCode']].groupby('userID').apply(lambda r: (
            r['assessmentItemID'].values,
            r['answerCode'].values))

# -------------------------------------------------
testing_dataset = SAKTDataset(group, train_num_problems, max_seq=MAX_SEQ)
testing_dataloader = DataLoader(testing_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=False, num_workers=8)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_ = SAKTModel(train_num_problems, max_seq=MAX_SEQ, embed_dim=EMBED_DIM, dropout_rate=DROPOUT_RATE)

criterion_ = nn.BCEWithLogitsLoss() # Binary Cross Entropy Loss에 Sigmod가 결합된 형태

model_.to(device)
criterion_.to(device)

# --------- get output --------
model = model_.cuda()
model.load_state_dict(torch.load('/opt/ml/input/code/sakt/very_1st_sakt_model.pt'))

lab, output = get_output(model, testing_dataloader, criterion_, device="cuda")

output_df = pd.DataFrame(zip(lab,output),columns=['label','prediction'])
pre_output = output_df.loc[output_df['label'] == -1]
real_output = pre_output.drop(['label'], axis=1)
real_output = real_output.reset_index(drop=True)
# real_output.loc[real_output['prediction'] < 0,'prediction'] = 0 # 0보다 작은거 다 0 으로
# real_output.loc[real_output['prediction'] > 1,'prediction'] = 1 # 0보다 작은거 다 0 으로
real_output.to_csv('/opt/ml/input/code/sakt/sakt_output/sakt_output.csv')