import torch
import argparse
import os
import random
import wandb
import mlflow

import pandas as pd
import numpy as np

from catboost_dataset import CBDataset
from catboost_model import CatBoostModel

# MLflow setting

# MLflow 서버 연결
remote_server_uri="http://118.67.134.110:30005"
mlflow.set_tracking_uri(remote_server_uri)

client = mlflow.tracking.MlflowClient()
experiment_name = "Catboost" # 튜토
try:
    experiment_id = client.create_experiment(experiment_name)
except:
    experiment = client.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id

mlflow.pytorch.autolog()

# ----------------

def seed_everythings(seed):
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    # MLflow ---------------
    run_name="DKT Catboost"
    desc="DKT Catboost w/ n features"
    # -------------------------
    with mlflow.start_run(run_name=run_name, description=desc, experiment_id=experiment_id) as run:
        # dataset
        dataset = CBDataset(args.data_dir)
        X_train, X_valid, y_train, y_valid = dataset.split_data()
        
        args.output_dir = os.path.join(args.output_dir)
        os.makedirs(args.output_dir, exist_ok=True)

        # model setting & run
        model = CatBoostModel(args, output_dir=args.output_dir)
        model.fit(
            X_train, y_train,
            cat_features=dataset.cat_features,
            eval_set=(X_valid, y_valid),
            verbose=args.verbose,
        )
        best_s= model.gat_best_score()
        best_iter = model.gat_best_iter()
        featrue_importance = model.gat_feature_importances()
        featrue_name = model.gat_feature_names()
        mlflow.log_metric('setted iter', args.iteration)
        mlflow.log_metric('learning rate', args.lr)
        mlflow.log_metric('random seed', args.seed)
        mlflow.log_metric('best score(valid AUC)', best_s['validation']['AUC'])
        mlflow.log_metric('best loss(valid)', best_s['validation']['Logloss'])
        mlflow.log_metric('best_iter', best_iter)
        # mlflow.log_metric('feature importance', featrue_importance)
        mlflow.log_metric('used feature num', len(featrue_importance))
        # mlflow.pytorch.log_model(model, artifact_path="model") # 모델 기록  <- TypeError: Argument 'pytorch_model' should be a torch.nn.Module
     
    # save model feature information
    # model.save_features(dataset.features, dataset.cat_features)#, args.feature_descript)
    # if args.save_model:
    #     # model.save_model(args.output_dir)
    #     torch.save(model,args.output_dir)

    # cmd 에서 확인하는 코드 -----------------------------------------------
    # featrue_importance = model.gat_feature_importances()
    # featrue_name = model.gat_feature_names()
    print('featrue_importance')
    print(featrue_importance)
    print('featrue_importance len : ', len(featrue_importance))
    print('featrue_importance shape : ', featrue_importance.shape)
    print(featrue_name)
    # ------------------------------------------------------------
        

    submission = pd.read_csv(args.inference_dir)
    submission.prediction = model.inference(dataset.get_test_data())
    submission.to_csv(os.path.join(args.output_dir, "catboost_submission.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="/opt/ml/input/data/ref_FE_1.pkl")
    parser.add_argument("--inference_dir", type=str, default="/opt/ml/input/data/sample_submission.csv")

    parser.add_argument("--output_dir", type=str, default="/opt/ml/input/code/CatBoost_model/catboost_output")
    parser.add_argument("--lr",type=float, default=0.01)
    parser.add_argument("--iteration", type=int, default=100) 
    parser.add_argument("--early_stopping", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42) # 13
    parser.add_argument("--verbose", type=int, default=100)
    parser.add_argument("--save_model", type=bool, default=True)
    
    
    args = parser.parse_args()
    
    seed_everythings(args.seed)
    main(args)