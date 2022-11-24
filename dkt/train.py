import os

import torch
import wandb
from args import parse_args
from src import trainer
from src.dataloader import Preprocess
from src.utils import setSeeds


def main(args):
    wandb.login()
    setSeeds(args.seed)
    args.device = "cuda" if torch.cuda.is_available() else "cpu"
    preprocess = Preprocess(args)
    preprocess.load_train_data(args.file_name)
    train_data = preprocess.get_train_data()  #shape = (6698, 4, interaction수_가변적)  
                                              #shape의 6698은 train.csv의 유저 수
                                              #shape 중간의 4는 ["testID","assessmentItemID","knowledgeTag","answerCode"]                          
    # user수만큼 🌕🌕🌕
    train_data = preprocess.get_partial_data(train_data, 640)
    train_data, valid_data = preprocess.split_data(train_data)
    wandb.init(project="Sequential", entity = "recsys8", config=vars(args))
    wandb.run.name = f"{args.model}_juj" # 표시되는 이름을 바꾸고 싶다면 해당 줄을 바꿔주세요
    wandb.run.save()

    model = trainer.get_model(args).to(args.device)
    wandb.watch(model,log = 'all')
    trainer.run(args, train_data, valid_data, model)

if __name__ == "__main__":
    args = parse_args()
    os.makedirs(args.model_dir, exist_ok=True)
    main(args)
