import os

import torch
from args import parse_args
from src import trainer
from src.dataloader import Preprocess


def main(args):
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess = Preprocess(args)
    preprocess.load_test_data(args)
    test_data = preprocess.get_test_data()
    model = trainer.load_model(args).to(args.device)
    trainer.inference(args, test_data, model)


if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    
    args.base_cols = ['userID','Timestamp','answerCode']
    args.cat_cols = ['testId','assessmentItemID','KnowledgeTag','big_category','mid_category','problem_num', 'month', 'dayname']
    args.num_cols = ['solvesec_600', 'big_mean', 'big_std','tag_mean', 'tag_std','test_mean', 'test_std','month_mean']
    
    args.used_cat_cols = ['TestId','assessmentItemID','KnowledgeTag','big_category','mid_category','problem_num', 'month', 'dayname']
    args.used_num_cols = ['solvesec_600', 'big_mean', 'big_std','tag_mean', 'tag_std','test_mean', 'test_std','month_mean']
    args.train_df_csv = "/opt/ml/input/main_dir/dkt/asset/train_fe_df.csv"
    args.test_df_csv = "/opt/ml/input/main_dir/dkt/asset/test_fe_df.csv"

    main(args)
