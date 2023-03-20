import os

import pandas as pd
import torch
from config import CFG, logging_conf
from lightgcn.datasets import prepare_dataset
from lightgcn.models import build, inference
from lightgcn.utils import get_logger

logger = get_logger(logging_conf)
use_cuda = torch.cuda.is_available() and CFG.use_cuda_if_available
device = torch.device("cuda" if use_cuda else "cpu")

if not os.path.exists(CFG.output_dir):
    os.makedirs(CFG.output_dir)


def main():
    logger.info("Task Started")

    logger.info("[1/4] Data Preparing - Start")
    train_data, test_data, n_node = prepare_dataset(
        device, CFG.basepath, verbose=CFG.loader_verbose, logger=logger.getChild("data")
    )
    logger.info("[1/4] Data Preparing - Done")

    logger.info("[2/4] Model Building - Start")
    
    itemnode = CFG.itemnode
    
    if itemnode != 'assessmentItemID':
        weight = "/opt/ml/dkt_team/code/lightgcn/weight/" + itemnode + "_best_model.pt"
    else:
        weight = "/opt/ml/dkt_team/code/lightgcn/weight/best_model.pt"
        
    model = build(itemnode,
        n_node,
        embedding_dim=128 if itemnode == 'assessmentItemID' else 256 ,
        num_layers=1 if itemnode == 'assessmentItemID' else 2,
        alpha=CFG.alpha,
        weight=weight,
        logger=logger.getChild("build"),
        **CFG.build_kwargs
    )
    model.to(device)
    logger.info("[2/4] Model Building - Done")

    logger.info("[3/4] Inference - Start")
    pred = inference(model, test_data, logger=logger.getChild("infer"))
    logger.info("[3/4] Inference - Done")

    logger.info("[4/4] Result Dump - Start")
    pred = pred.detach().cpu().numpy()
    pd.DataFrame({"prediction": pred}).to_csv(
        os.path.join(CFG.output_dir, CFG.pred_file), index_label="id"
    )
    logger.info("[4/4] Result Dump - Done")

    logger.info("Task Complete")


if __name__ == "__main__":
    main()
