import os

import torch
import mlflow
import wandb
from args import parse_args
from src import trainer
from src.dataloader import Preprocess
from src.utils import setSeeds
import numpy as np


def main(args):
    wandb.login()
    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    print(args.device)

    preprocess = Preprocess(args)
    preprocess.load_train_data(args)
    train_data = preprocess.get_train_data()  #shape = (6698, 4, interaction수_가변적)  
                                              #shape의 6698은 train.csv의 유저 수
                                              #shape 중간의 4는 ["testID","assessmentItemID","knowledgeTag","answerCode"]+solvesec
    train_data, valid_data = preprocess.split_data(train_data)
    # answerCode가 -1제외 test데이터도 같이 학습
    # preprocess.load_train_test_data(args)
    # train_test_data = preprocess.get_train_test_data()
    # train_data = np.concatenate((train_data, train_test_data), axis=0)

    wandb.init(project="Sequential", entity = "recsys8", config=vars(args))
    wandb.run.name = f"{args.model}" # 표시되는 이름을 바꾸고 싶다면 해당 줄을 바꿔주세요
    wandb.run.save()

    model = trainer.get_model(args).to(args.device)
    wandb.watch(model,log = 'all')
    trainer.run(args, train_data, valid_data, model)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)

    # config = ConfigParser.from_args(args)
    remote_server_uri ="http://118.67.134.110:30005"
    mlflow.set_tracking_uri(remote_server_uri)
    
    experiment_name = "같은 피처 여러 시퀀스모델들"
    # experiment_name = args.model
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment == None:
        experiment = mlflow.set_experiment(experiment_name)
    client = mlflow.tracking.MlflowClient()
    
    run = client.create_run(experiment.experiment_id)

    #🙂1. FE할 때 여기 고치세요!
    args.base_cols = ['userID','Timestamp','answerCode']
    args.cat_cols = ['assessmentItemID', 'testId', 'KnowledgeTag', 'big_category', 'mid_category', 'problem_num',\
                     'time_category', 'solvecumsum_category']
    args.num_cols =['solvesec_3600', 'solvesec_cumsum', 'test_mean', 'test_std', \
                    'tag_mean', 'tag_std', 'big_mean', 'big_std', 'big_sum', 'assess_mean', 'assess_std', \
                    'user_mean', 'user_std', 'user_sum', 'assess_count']
    
    args.used_cat_cols = ['assessmentItemID', 'testId', 'KnowledgeTag', 'big_category', 'mid_category', 'problem_num',\
                        'time_category', 'solvecumsum_category']
    args.used_num_cols = ['solvesec_3600', 'solvesec_cumsum', 'test_mean', 'test_std', \
                        'tag_mean', 'tag_std', 'big_mean', 'big_std', 'big_sum', 'assess_mean', 'assess_std', \
                        'user_mean', 'user_std', 'user_sum', 'assess_count']
    args.train_df_csv = "/opt/ml/input/main_dir/dkt/asset/train_fe_df.csv"
    args.test_df_csv = "/opt/ml/input/main_dir/dkt/asset/test_fe_df.csv"

    run_name = "🌈(12/06 Tue)["+args.model+"]augment:+"+str(args.window)+" 피처: "+str(len(args.used_cat_cols)+len(args.used_num_cols))+"개)"
    desc = '사용한 피처 :' + ', '.join(args.used_cat_cols + args.used_num_cols)

    with mlflow.start_run(run_name="tmp", run_id=run.info.run_id, description=desc):
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.set_tag('mlflow.user', 'lnh')
        params = {"model":args.model,
                  "lr":args.lr,
                  "epoch":args.n_epochs,
                  "hidden_dim":args.hidden_dim,
                  "seq_len":args.max_seq_len,
                  "batch":args.batch_size,
                  "drop_out":args.drop_out,
                  "patience":args.patience,
                  }
        mlflow.log_params(params)

        main(args)