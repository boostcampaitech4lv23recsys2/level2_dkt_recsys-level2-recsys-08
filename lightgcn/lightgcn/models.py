import os

import numpy as np
import torch
import mlflow
import argparse
from sklearn.metrics import accuracy_score, roc_auc_score
from torch_geometric.nn.models import LightGCN
from .utils import setSeeds


def build(itemnode, n_node, weight=None, logger=None, **kwargs):
    model = LightGCN(n_node, **kwargs)
    
    # if itemnode != "assessmentItemID":
    #     weight = "/opt/ml/dkt_team/code/lightgcn/weight/" + itemnode + "_best_model.pt"
    # else :
    #     weight = "/opt/ml/dkt_team/code/lightgcn/weight/best_model.pt"
        
    if weight:
        if not os.path.isfile(weight):
            logger.fatal("Model Weight File Not Exist")
        logger.info("Load model")
        state = torch.load(weight)["model"]
        model.load_state_dict(state)
        return model
    else:
        logger.info("No load model")
        return model


def train(itemnode,
    model,
    train_data,
    valid_data=None,
    n_epoch=100,
    learning_rate=0.01,
    use_wandb=None,
    weight=None,
    logger=None,
):
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    if not os.path.exists(weight):
        os.makedirs(weight)

    if valid_data is None:
        setSeeds() # 랜덤시드 설정
        eids = np.arange(len(train_data["label"])) #(2475962,)
        eids = np.random.permutation(eids)[:1000]  #(1000,)
        edge, label = train_data["edge"], train_data["label"]    #edge=(2,2475962), label=(2475962)
        label = label.to("cpu").detach().numpy()
        valid_data = dict(edge=edge[:, eids], label=label[eids]) #edge=(2,1000), label=(1000,)

    logger.info(f"Training Started : n_epoch={n_epoch}")
    best_auc, best_epoch = 0, -1
    for e in range(n_epoch):
        # forward
        pred = model(train_data["edge"])
        loss = model.link_pred_loss(pred, train_data["label"])

        # backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            prob = model.predict_link(valid_data["edge"], prob=True)
            prob = prob.detach().cpu().numpy()
            acc = accuracy_score(valid_data["label"], prob > 0.5)
            auc = roc_auc_score(valid_data["label"], prob)
            logger.info(
                f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}"
            )
            if use_wandb:
                import wandb
                wandb.log({'loss':loss, 'acc':acc, 'auc':auc})

        if weight:
            if auc > best_auc:
                logger.info(
                    f" * In epoch {(e+1):04}, loss={loss:.03f}, acc={acc:.03f}, AUC={auc:.03f}, ✨Best AUC✨"
                )
                best_auc, best_epoch = auc, e
                if itemnode != "assessmentItemID":
                    torch.save(
                        {"model": model.state_dict(), "epoch": e + 1},
                        os.path.join(weight, itemnode + "_best_model.pt"),
                    )
                else:
                    torch.save(
                        {"model": model.state_dict(), "epoch": e + 1},
                        os.path.join(weight, f"best_model.pt"),
                        )
                early_stopping_counter = 0
            else:
                early_stopping_counter += 1
                if early_stopping_counter >= 10:
                    print(
                        f"EarlyStopping counter: {early_stopping_counter} out of 10"
                    )
                    break
                
            if use_wandb:
                wandb.log({'best_auc':best_auc})
                wandb.run.summary['best_auc'] = best_auc
                
            mlflow.log_metric("BEST AUC",best_auc)
            mlflow.log_metric("ACC",acc)
            mlflow.log_metric("AUC",auc)
            mlflow.log_metric("LOSS",loss)
            
        mlflow.pytorch.log_model(model, artifact_path="model") # 모델 기록
    
    torch.save(
        {"model": model.state_dict(), "epoch": e + 1},
        os.path.join(weight, f"last_model.pt"),
    )
    logger.info(f"Best Weight Confirmed : {best_epoch+1}'th epoch")


def inference(model, data, logger=None):
    model.eval()
    with torch.no_grad():
        pred = model.predict_link(data["edge"], prob=True)
        return pred
