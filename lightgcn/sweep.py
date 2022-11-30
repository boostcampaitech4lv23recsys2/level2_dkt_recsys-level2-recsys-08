import pandas as pd
import torch
import wandb
from args import parse_args
from lightgcn.datasets import prepare_dataset
from lightgcn.models import build, train
from lightgcn.utils import class2dict, get_logger

logging_conf = {  # only used when 'user_wandb==False'
    "version": 1,
    "formatters": {
        "basic": {"format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s"}
    },
    "handlers": {
        "console": {
            "class": "logging.StreamHandler",
            "level": "INFO",
            "formatter": "basic",
            "stream": "ext://sys.stdout",
        },
        "file_handler": {
            "class": "logging.FileHandler",
            "level": "DEBUG",
            "formatter": "basic",
            "filename": "run.log",
        },
    },
    "root": {"level": "INFO", "handlers": ["console", "file_handler"]},
}


def main(args):
    logger = get_logger(logging_conf)
    use_cuda = torch.cuda.is_available() and args.use_cuda_if_available
    device = torch.device("cuda" if use_cuda else "cpu")
    print(device)
    
    wandb.login()
    wandb.init(project="GNN", entity = "recsys8", config=vars(args))

    logger.info("Task Started")

    logger.info("[1/1] Data Preparing - Start")
    train_data, test_data, n_node = prepare_dataset(
        device, args.basepath, verbose=args.loader_verbose, logger=logger.getChild("data")
    )
    logger.info("[1/1] Data Preparing - Done")

    logger.info("[2/2] Model Building - Start")
    model = build(
        n_node,
        embedding_dim=args.embedding_dim,
        num_layers=args.num_layers,
        alpha=args.alpha,
        logger=logger.getChild("build"),
        **args.build_kwargs
    )
    model.to(device)

    if args.user_wandb:
        wandb.watch(model)

    logger.info("[2/2] Model Building - Done")

    logger.info("[3/3] Model Training - Start")
    train(
        model,
        train_data,
        n_epoch=args.n_epoch,
        learning_rate=args.learning_rate,
        use_wandb=args.user_wandb,
        weight=args.weight_basepath,
        logger=logger.getChild("train"),
    )

if __name__ == "__main__":
    args = parse_args()
    main(args)
