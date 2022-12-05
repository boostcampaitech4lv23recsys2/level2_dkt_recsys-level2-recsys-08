import pandas as pd
import torch
import mlflow
from config import CFG, logging_conf
from lightgcn.datasets import prepare_dataset
from lightgcn.models import build, train
from lightgcn.utils import class2dict, get_logger

if CFG.user_wandb:
    import wandb

    wandb.init(**CFG.wandb_kwargs, config=class2dict(CFG))
    wandb.run.name = f"lightGCN_jhl" # 표시되는 이름을 바꾸고 싶다면 해당 줄을 바꿔주세요
    wandb.run.save()


logger = get_logger(logging_conf)
use_cuda = torch.cuda.is_available() and CFG.use_cuda_if_available
device = torch.device("cuda" if use_cuda else "cpu")
print(f'device : {device}')


def main():
    logger.info("Task Started")

    logger.info("[1/1] Data Preparing - Start")
    train_data, test_data, n_node = prepare_dataset(
        device, CFG.basepath, verbose=CFG.loader_verbose, logger=logger.getChild("data")
    )
    logger.info("[1/1] Data Preparing - Done")

    logger.info("[2/2] Model Building - Start")
    model = build(
        n_node,
        embedding_dim=CFG.embedding_dim,
        num_layers=CFG.num_layers,
        alpha=CFG.alpha,
        logger=logger.getChild("build"),
        **CFG.build_kwargs
    )
    model.to(device)

    if CFG.user_wandb:
        wandb.watch(model)

    logger.info("[2/2] Model Building - Done")

    logger.info("[3/3] Model Training - Start")
    train(
        model,
        train_data,
        n_epoch=CFG.n_epoch,
        learning_rate=CFG.learning_rate,
        use_wandb=CFG.user_wandb,
        weight=CFG.weight_basepath,
        logger=logger.getChild("train"),
    )
    logger.info("[3/3] Model Training - Done")

    logger.info("Task Complete")


if __name__ == "__main__":
    
    remote_server_uri ="http://118.67.134.110:30005"
    mlflow.set_tracking_uri(remote_server_uri)
    
    experiment_name = "lightGCN"
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment == None:
        experiment = mlflow.set_experiment(experiment_name)
    client = mlflow.tracking.MlflowClient()
    
    run = client.create_run(experiment.experiment_id)
    run_name = "🌈(12/05 Mon)[lightGCN]"

    #🙂1. FE할 때 여기 고치세요!
    columns = ["assessmentItemID"]
    desc = '사용한 피처 :' + ', '.join(columns)

    with mlflow.start_run(run_id=run.info.run_id, run_name=run_name, description=desc):
        mlflow.set_tag("mlflow.runName", run_name)
        mlflow.set_tag('mlflow.user', 'jhl')
        params = {"lr":CFG.learning_rate,
                  "epoch":CFG.n_epoch,
                  "embedding_dim":CFG.embedding_dim,
                  "num_layers":CFG.num_layers,
                  }
        mlflow.log_params(params)

        main()
