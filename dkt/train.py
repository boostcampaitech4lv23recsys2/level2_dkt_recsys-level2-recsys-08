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
    preprocess = Preprocess(args)
    preprocess.load_train_data(args)
    train_data = preprocess.get_train_data()  #shape = (6698, 4, interaction수_가변적)  
                                              #shape의 6698은 train.csv의 유저 수
                                              #shape 중간의 4는 ["testID","assessmentItemID","knowledgeTag","answerCode"]+solvesec
    train_data, valid_data = preprocess.split_data(train_data)
    # answerCode가 -1제외 test데이터도 같이 학습
    # preprocess.load_train_test_data(args)
    # train_test_data = preprocess.get_train_test_data()
    # train_data = train_test_data
    
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
    
    experiment_name = args.model
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment == None:
        experiment = mlflow.set_experiment(experiment_name)
    client = mlflow.tracking.MlflowClient()
    
    run = client.create_run(experiment.experiment_id)
    run_name = "🌈(12/05 Mon)["+args.model+"] 총 피처: 3개 / lgcn 임베딩 피처 : 2개)"

    #🙂1. FE할 때 여기 고치세요!
    # columns = ['assessmentItemID', 'testId','KnowledgeTag', 
    #         'big_category', 'mid_category', 'problem_num', 'month',
    #         'dayname', 'month_mean', 'solvesec_600', 'test_mean', 'test_std', 'test_sum',
    #         'tag_mean', 'tag_std', 'tag_sum', 'big_mean', 'big_std', 'big_sum', 'user_correct_answer', 'user_total_answer', 'user_acc']
    columns = ["assessmentItemID", "testId", "KnowledgeTag", "big_category", "mid_category", "problem_num",
               "assIdx", "month", "day", "hour", "dayname", "time_category", "solvecumsum_category",
               "solvesec_3600", "test_mean", 'test_std', "tag_mean", 'tag_std', "big_mean", 'big_std',
               "user_correct_answer", "user_total_answer", "user_acc", "solvesec_cumsum", "big_category_cumconut", "big_category_user_acc", "big_category_user_std", "big_category_answer", "big_category_answer_log1p", "elo_assessmentItemID"]
    
    desc = '사용한 피처 :' + ', '.join(columns)

    with mlflow.start_run(run_id=run.info.run_id, run_name=run_name, description=desc):
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.set_tag('mlflow.user', 'jhl')
        params = {"lr":args.lr,
                  "epoch":args.n_epochs,
                  "hidden_dim":args.hidden_dim,
                  "seq_len":args.max_seq_len,
                  "batch":args.batch_size,
                  "drop_out":args.drop_out,
                  "patience":args.patience,
                  }
        mlflow.log_params(params)

        main(args)
