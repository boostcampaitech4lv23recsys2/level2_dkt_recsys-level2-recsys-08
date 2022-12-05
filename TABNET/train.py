import os
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
import mlflow

import time
from datetime import datetime
from pytz import timezone
import wandb

from PIL import Image
from args import parse_args

from utils import seed_everything, plot_explain, data_load, explain_image,fi_image
from data import categorical_feature,assess_count,\
    feature_engineering,custom_train_test_split,percentile,time_feature_engineering

from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
from pytorch_tabnet.tab_model import TabNetClassifier, TabNetRegressor
from pytorch_tabnet.callbacks import Callback

from mlflow_util import MLCallback,connect_server

def main(args):
    # wandb + seed 고정
    wandb.login()
    seed_everything(args.SEED)

    wandb.init(project="TabNet", entity = "recsys8", config=vars(args))
    wandb.run.name = "TabNet_test"
    wandb.run.save()

    now = datetime.now(tz = timezone('Asia/Seoul'))
    date_str = now.strftime('%m-%d-%H:%M:%S')

    cat_features = ['userID','assessmentItemID','testId','KnowledgeTag','assess_count']
    # data load
    df = data_load(args.DATA_PATH)

    # time feature engineering
    df = assess_count(df)
    df = time_feature_engineering(df)

    # feature engineering
    df = feature_engineering(df)
    df.fillna(0,inplace = True)
    df = categorical_feature(df,cat_features)

    # train test split
    train = df[df['answerCode']>-1]
    test = df[df['answerCode']==-1]
    train,valid = custom_train_test_split(train)

    # FEATS 선언, categorical features idx선언
    FEATS = ['userID','assessmentItemID','testId','KnowledgeTag','Time','Timediff','Timepassed','Time_answer_rate','assess_count',\
         'user_acc', 'test_mean', 'test_sum', 'tag_mean','tag_sum','answerCode']
    train_set = train[FEATS]
    valid_set = valid[FEATS]
    cat_idxs = [train_set.columns.get_loc(col) for col in cat_features]
    cat_dims = [df[col].nunique() for col in cat_features]

    # model 선언
    model = TabNetClassifier(
        seed = args.SEED,
        n_d = args.ND,
        n_a = args.NA,
        n_steps = args.N_STEPS,
        gamma = args.GAMMA, 
        n_independent = args.N_INDEPENDENT,
        n_shared = args.N_SHARED,
        cat_emb_dim=args.CAT_EMB_DIM,
        optimizer_params=dict(lr=args.LR),
        momentum=args.MOMENTUM,
        cat_idxs=cat_idxs,
        cat_dims=cat_dims,
        verbose=1,
        #scheduler_params=dict(milestones=[20, 50, 80], gamma=0.5), 
        #scheduler_fn=torch.optim.lr_scheduler.MultiStepLR,
        lambda_sparse = args.LAMBDA,
        clip_value = args.CLIP,
    )

    # mlflow 연결
    remote_server_uri,experiment_id = connect_server()
    ml_callback = MLCallback
    run_name="tabent"+date_str
    desc="tabnet emergency"
    mlflow.end_run()

    # model 학습
    model.fit(
        X_train = train_set.drop(columns = 'answerCode').values,
        y_train = train_set['answerCode'].values,
        eval_set = [(train_set.drop(columns = 'answerCode').values,train_set['answerCode'].values),(valid_set.drop(columns = 'answerCode').values,valid_set['answerCode'].values)],
        eval_name = ['train','valid'],
        eval_metric = ['accuracy','auc'],
        max_epochs = args.N_EPOCHS, 
        patience = args.PATIENCE,
        batch_size = args.BATCH_SZ, 
        virtual_batch_size = args.VIRTUAL_BS,
        num_workers = 0,
        weights = 1,
        drop_last = False,
        callbacks=[ml_callback(remote_server_uri, experiment_id, run_name, desc,model.get_params())]
        
    )
    
    # model 저장
    saving_path_name = "./saved/tabnet_" + date_str
    model.save_model(saving_path_name)

    # metric 계산
    valid_proba =  model.predict_proba(valid_set[FEATS].drop(columns = 'answerCode').values)
    valid_preds =  model.predict(valid_set[FEATS].drop(columns = 'answerCode').values)
    valid_auc = roc_auc_score(valid_set['answerCode'],valid_proba[:,1])
    valid_acc = accuracy_score(valid_set['answerCode'],valid_preds)

    # wandb logging
    for i in range(len(model.history['valid_auc'])):
        wandb.log({
            'epoch' : i,
            'loss' : model.history['loss'][i],
            'valid_auc' : model.history['valid_auc'][i],
            'valid_acc' : model.history['valid_accuracy'][i],
        },step = i)

    wandb.run.summary['best_auc'] = valid_auc
    wandb.run.summary['best_acc'] = valid_acc

    # mlflow image logging
    feat_importances = model.feature_importances_
    indices = np.argsort(feat_importances)
    fi_image(feat_importances,indices,FEATS,experiment_id)

    explain,masks = model.explain(valid_set.drop(columns = 'answerCode').values)
    explain_image(explain,masks,FEATS,experiment_id)
    

    # submission 저장
    total_preds = model.predict_proba(test[FEATS].drop(columns = 'answerCode').values)[:,1]
    submission = pd.read_csv('../../data/sample_submission.csv')
    submission['prediction'] = total_preds
    submission.to_csv('./submission/submission'+date_str+'.csv')



if __name__ == "__main__":
    args = parse_args()
    main(args)