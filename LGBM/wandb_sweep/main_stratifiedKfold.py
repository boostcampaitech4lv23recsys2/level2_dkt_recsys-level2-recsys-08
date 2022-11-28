import argparse
import pandas as pd
import os
import random
import warnings
warnings.filterwarnings('ignore')
import lightgbm as lgb
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import numpy as np
from utils import *
from sklearn.model_selection import StratifiedGroupKFold
import wandb

def main(args):

    ## wandb setting
    wandb.login()
    wandb.init()
    wandb.run.name = f"LGBM_LR_{args.learning_rate}_num_leaves_{args.num_leaves}_feature_fraction_{args.feature_fraction}_bagging_fraction_{args.bagging_fraction}_bagging_freq_{args.bagging_freq}"
    wandb.run.save()
    # wandb.config.update(args)

    ## 1. 데이터 로딩
    data_dir = '/opt/ml/input/data' # 경로
    # train_file_path = os.path.join(data_dir, 'train_data.csv') # 데이터
    # test_file_path = os.path.join(data_dir, 'test_data.csv')
    # df_train = pd.read_csv(train_file_path)
    df = pd.read_csv(os.path.join(data_dir, 'train_test_last2.csv'))
    # df_test = pd.read_csv(test_file_path)
    # df_test = df_test[df_test.answerCode!=-1]  # answer_code -1 제외

    ## 2. FE
    # train_fe = feature_engineering(df_train)
    # test_fe = feature_engineering(df_test)

    # ## 3. train test split
    # ### 3.1 train - test merge
    # test_to_train = test_fe[test_fe['userID'] == test_fe['userID'].shift(-1)]
    # test_to_train.shape
    # train_all = pd.concat([train_fe,test_to_train])
    # valid = test_fe[test_fe['userID'] != test_fe['userID'].shift(-1)]

    # ### 3.2 Custom train test split
    # # X, y 값 분리
    # y_train_main = train_all['answerCode']
    # train_main = train_all.drop(['answerCode'], axis=1)
    # y_test_main = valid['answerCode']
    # test_main = valid.drop(['answerCode'], axis=1)

    # X, y 값 분리
    n_splits = 6
    k_auc_list = list()

    cv = StratifiedGroupKFold(n_splits=n_splits)

    x = df.drop('answerCode',axis=1)
    y = df['answerCode']

    ## 4. train
    # 파라미터 설정
    params = {
        # "max_depth": args.max_depth, # default=-1 (no limit)
        "learning_rate": args.learning_rate,  # default = 0.1, [0.0005 ~ 0.5]
        "boosting": "gbdt",
        "objective": args.objective,
        "metric": args.metric,
        "num_leaves": args.num_leaves,  # default = 31, [10, 20, 31, 40, 50]
        "feature_fraction": args.feature_fraction,  # default = 1.0, [0.4, 0.6, 0.8, 1.0]
        "bagging_fraction": args.bagging_fraction,  # default = 1.0, [0.4, 0.6, 0.8, 1.0]
        "bagging_freq": args.bagging_freq,  # default = 0, [0, 1, 2, 3, 4]
        "seed": 42,
        "verbose": -1,
    }
    # 사용할 Feature 설정
    FEATS = ['KnowledgeTag', 'user_correct_answer', 'user_total_answer',
            'big_category',
            'mid_category',
            'problem_num',
            'month','day','dayname','hour',
            'user_acc',
            'test_mean',
            'test_sum',
            'test_std', 'tag_std',
            'tag_mean',
            'tag_sum',
            ]

    for k, (train_idx, valid_idx) in enumerate(cv.split(x,y,df.group),1):
        
        # train 분리
        train = x.iloc[train_idx]
        y_train = y.iloc[train_idx]
        
        # valid 분리
        val = x.iloc[valid_idx]
        valid = val[val.valid==1]
        y_valid = y.iloc[valid.index]

        model = lgb.LGBMClassifier(
        **params,
        n_estimators=10000,
        silent=-1,
        )

        model.fit(
            X=train[FEATS],
            y=y_train,
            early_stopping_rounds=100,
            eval_set=[(train[FEATS], y_train), (valid[FEATS], y_valid)],
            eval_names=["train", "valid"],
            eval_metric="roc_auc",
            verbose=100,
        )
        
        preds = model.predict_proba(valid[FEATS])[:, 1]
        acc = accuracy_score(y_valid, np.where(preds >= 0.5, 1, 0))
        auc = roc_auc_score(y_valid, preds)
        k_auc_list.append(auc)
        metric={
        "Valid/AUC": auc,
        "Valid/ACC": acc,
        "best_iters":model.best_iteration_,
        }
        wandb.log(metric)
        print(f'VALID AUC : {auc} ACC : {acc}\n')

    kfold_auc = sum(k_auc_list) / n_splits
    wandb.log({"kfold_auc":kfold_auc})
    wandb.finish()



if __name__ == "__main__":
    ######################## BASIC ENVIRONMENT SETUP
    parser = argparse.ArgumentParser(description='parser')
    arg = parser.add_argument

    ############### BASIC OPTION
    parser.add_argument("--seed", default=42, type=int, help="seed")

    parser.add_argument("--device", default="cpu", type=str, help="cpu or gpu")

    parser.add_argument(
        "--data_dir",
        default="/opt/ml/input/data/",
        type=str,
        help="data directory",
    )
    parser.add_argument(
        "--asset_dir", default="asset/", type=str, help="data directory"
    )

    parser.add_argument(
        "--file_name",
        default="train_data.csv",
        type=str,
        help="train file name",  # 원래 train_data.csv
    )

    parser.add_argument(
        "--model_dir", default="models/", type=str, help="model directory"
    )
    parser.add_argument(
        "--model_name", default="model.pt", type=str, help="model file name"
    )

    parser.add_argument(
        "--output_dir", default="output/", type=str, help="output directory"
    )
    parser.add_argument(
        "--test_file_name", default="test_data.csv", type=str, help="test file name"
    )

    parser.add_argument(
        "--max_seq_len", default=100, type=int, help="max sequence length"
    )
    parser.add_argument("--num_workers", default=1, type=int, help="number of workers")

    # 모델
    parser.add_argument(
        "--hidden_dim", default=64, type=int, help="hidden dimension size"
    )
    parser.add_argument("--n_layers", default=2, type=int, help="number of layers")
    parser.add_argument("--n_heads", default=2, type=int, help="number of heads")
    parser.add_argument("--drop_out", default=0.2, type=float, help="drop out rate")
    # 훈련
    parser.add_argument("--max_depth", default=-1, type=int)
    parser.add_argument("--boosting", default="gbdt", type=str)
    parser.add_argument("--learning_rate", default=0.1, type=float)
    parser.add_argument("--objective", default="binary", type=str)
    parser.add_argument("--metric", default="auc", type=str)
    parser.add_argument("--num_leaves", default=31, type=int)
    parser.add_argument("--feature_fraction", default=1.0, type=float)
    parser.add_argument("--bagging_fraction", default=1.0, type=float)
    parser.add_argument("--bagging_freq", default=1, type=int)

    parser.add_argument(
        "--log_steps", default=50, type=int, help="print log per n steps"
    )
    
    # wandb config
    parser.add_argument(
        "--project_name", default="dkt-LGBM", type=str, help="wandb project name"
    )
    parser.add_argument(
        "--run_name", default="run_LGBM", type=str, help="wandb run name"
    )

    ### 중요 ###
    parser.add_argument("--model", default="lstmattn", type=str, help="model type")
    parser.add_argument("--optimizer", default="adam", type=str, help="optimizer type")
    parser.add_argument(
        "--scheduler", default="plateau", type=str, help="scheduler type"
    )

    args = parser.parse_args()
    main(args)