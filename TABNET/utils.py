import os
import numpy as np
import pandas as pd
import torch
import random
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
    train = pd.read_csv(path+'train_time.csv')
    test = pd.read_csv(path+'test_time.csv')
    return train,test
