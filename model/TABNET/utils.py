import os
import numpy as np
import pandas as pd
import torch
import random
import matplotlib.pyplot as plt
import mlflow
from PIL import Image
from sklearn.preprocessing import LabelEncoder

def seed_everything(seed_value):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ['PYTHONHASHSEED'] = str(seed_value)

    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False

def data_load(path):
    train = pd.read_csv(path+'train_data.csv',parse_dates=['Timestamp'])
    test = pd.read_csv(path+'test_data.csv',parse_dates=['Timestamp'])
    df = pd.concat([train,test],axis = 0).reset_index(drop = True)
    return df

def plot_explain(explain,FEATS):
    if type(explain) == dict:
        fig, axes = plt.subplots(1,3,figsize = (15,5))

        for i,ax in enumerate(axes):
            pcm = ax.pcolor(explain[i],vmin = 0, vmax = 1)
            ax.set_xticks(np.arange(len(FEATS[:-1]))+0.5)
            ax.set_xticklabels(FEATS[:-1], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('./temp')
    else :
        fig, ax = plt.subplots(1,1,figsize = (10,5))

        pcm = ax.pcolor(explain,vmin = 0, vmax = 1)
        ax.set_xticks(np.arange(len(FEATS[:-1]))+0.5)
        ax.set_xticklabels(FEATS[:-1], rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig('./temp')

def fi_image(feat_importances,indices,FEATS,experiment_id):
    plt.figure()
    plt.title("Feature importances")
    plt.barh(range(len(feat_importances)), feat_importances[indices],
        color="r", align="center")
    # If you want to define your own labels,
    # change indices to a list of labels on the following line.
    plt.yticks(range(len(feat_importances)), [FEATS[idx] for idx in indices])
    plt.ylim([-1, len(feat_importances)])
    plt.tight_layout()
    plt.savefig('./temp',pad_inches = 0.5)
    image = Image.open('./temp.png')
    run_id = mlflow.last_active_run().info.run_id
    with mlflow.start_run(run_id = run_id,experiment_id = experiment_id):
        mlflow.log_image(image, 'Feature_importance'+'.png')

def explain_image(explain,masks,FEATS,experiment_id):
    a = []
    a.append(explain)
    b = ['explain']
    for i in range(len(masks)):
        a.append(masks[i])
        b.append('mask'+str(i))
    run_id = mlflow.last_active_run().info.run_id
    for img,name in zip(a,b):
        plot_explain(img,FEATS)
        image = Image.open('./temp.png')
        run_id = mlflow.last_active_run().info.run_id
        with mlflow.start_run(run_id = run_id,experiment_id = experiment_id):
            mlflow.log_image(image, name+'.png')