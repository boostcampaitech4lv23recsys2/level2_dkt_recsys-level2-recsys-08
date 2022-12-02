import torch
import argparse
import os
import random
import wandb

import pandas as pd
import numpy as np

from catboost_dataset import CBDataset
from catboost_model import CatBoostModel


def seed_everythings(seed):
    np.random.seed(seed)
    random.seed(seed)


def main(args):
    # wandb setting
    # wandb.login()
    # wandb.init(
    #     project='Catboost',
    #     # config=args
    #     name='catboost_cms'
    # )
    # wandb.run.name = 
    # wandb.run.save()

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
    # save model feature information
    # model.save_features(dataset.features, dataset.cat_features)#, args.feature_descript)
    # if args.save_model:
    #     # model.save_model(args.output_dir)
    #     torch.save(model,args.output_dir)
    # -----------------------------------------------
    featrue_importance = model.gat_feature_importances()
    featrue_name = model.gat_feature_names()
    print('featrue_importance')
    print(featrue_importance)
    print('featrue_importance len : ', len(featrue_importance))
    print('featrue_importance shape : ', featrue_importance.shape)
    print(featrue_name)
    # bs, bi, all_para, er = model.gat_()
    # wandb.log({
    #     # 'epoch': args.iteration,
    #     # 'lr': args.lr,
    #     'best_score': bs,
    #     'best_iteration': bi,
    #     'auc': er['validation']['AUC']
    # })
        

    submission = pd.read_csv(args.inference_dir)
    submission.prediction = model.inference(dataset.get_test_data())
    submission.to_csv(os.path.join(args.output_dir, "catboost_submission.csv"), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_dir", type=str, default="/opt/ml/input/data/ref_FE_1.pkl")
    parser.add_argument("--inference_dir", type=str, default="/opt/ml/input/data/sample_submission.csv")

    parser.add_argument("--output_dir", type=str, default="/opt/ml/input/code/CatBoost_model/catboost_output")
    parser.add_argument("--lr",type=float, default=0.01)
    parser.add_argument("--iteration", type=int, default=1000) 
    parser.add_argument("--early_stopping", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42) # 13
    parser.add_argument("--verbose", type=int, default=100)
    parser.add_argument("--save_model", type=bool, default=True)
    
    
    args = parser.parse_args()
    
    seed_everythings(args.seed)
    main(args)