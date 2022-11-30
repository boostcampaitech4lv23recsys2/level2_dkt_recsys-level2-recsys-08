import os
import random
import time
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn.preprocessing import LabelEncoder


class Preprocess:
    def __init__(self, args):
        self.args = args
        self.train_data = None
        self.train_test_data = None
        self.test_data = None

    def get_train_data(self):
        return self.train_data

    def get_train_test_data(self):
        return self.train_test_data

    def get_test_data(self):
        return self.test_data

    def split_data(self, data, ratio=0.7, shuffle=True, seed=0):
        """
        split data into two parts with a given ratio.
        """
        if shuffle:
            random.seed(seed)  # fix to default seed 0
            random.shuffle(data)

        size = int(len(data) * ratio)
        data_1 = data[:size]
        data_2 = data[size:]

        return data_1, data_2

    def __save_labels(self, encoder, name):
        le_path = os.path.join(self.args.asset_dir, name + "_classes.npy")
        np.save(le_path, encoder.classes_)

    def __preprocessing(self, df, args, is_train=True):
        cate_cols = args.custom_columns
        if args.partial_user: #640명에 대해서 자른다.
            df = df[df['userID'] < 717]
        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)

        for col in cate_cols:

            le = LabelEncoder()
            if is_train:
                # For UNKNOWN class
                a = df[col].unique().tolist() + ["unknown"]
                le.fit(a)
                self.__save_labels(le, col)
            else:
                label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
                le.classes_ = np.load(label_path)

                df[col] = df[col].apply(
                    lambda x: x if str(x) in le.classes_ else "unknown"
                )

            # 모든 컬럼이 범주형이라고 가정
            df[col] = df[col].astype(str)
            test = le.transform(df[col])
            df[col] = test

        def convert_time(s):
            timestamp = time.mktime(
                datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
            )
            return int(timestamp)

        df["Timestamp"] = df["Timestamp"].apply(convert_time)

        return df

    def __feature_engineering(self, df, args):
        if args.df_file_name:
            data_dir = '/opt/ml/input/data' # 경로
            after_fe_path = os.path.join(data_dir, args.df_file_name)
            df = pd.read_pickle(after_fe_path)
        return df[args.columns]

    def load_data_from_file(self, args, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        if is_train == True:
            df = df[df['answerCode'] != -1]
        df = self.__feature_engineering(df, args)
        df = self.__preprocessing(df, args, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용

        non_categorical = set([])
        for col in args.custom_columns:
            if col not in non_categorical:
                exec("self.args.n_"+ col +'= len(np.load(os.path.join(self.args.asset_dir,"' + col + '_classes.npy")))')
        
        # self.args.n_questions = len(
        #     np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        # )
        # self.args.n_test = len(
        #     np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        # )
        # self.args.n_tag = len(
        #     np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        # )

        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        columns = args.columns
        tmp_columns = ["answerCode"] + args.custom_columns
        group = (
            df[columns]
            .groupby("userID")
            .apply(lambda r: [r[col].values for col in columns if col in tmp_columns]
                # lambda r: (
                #     r["testId"].values,
                #     r["assessmentItemID"].values,
                #     r["KnowledgeTag"].values,
                #     r["answerCode"].values,
                #     r["solvesec"].values,
                # )
            )
        )
        return group.values

    def load_train_data(self, args):
        self.train_data = self.load_data_from_file(args, args.file_name)

    def load_train_test_data(self, args):
        self.train_test_data = self.load_data_from_file(args, args.test_file_name)

    def load_test_data(self, args):
        self.test_data = self.load_data_from_file(args, args.test_file_name, is_train=False)


class DKTDataset(torch.utils.data.Dataset):
    def __init__(self, data, args):
        self.data = data
        self.args = args

    def __getitem__(self, index):
        row = self.data[index]

        # 각 data의 sequence length
        seq_len = len(row[0])
        #👀
        feature_len = len(row)

        for i in range(feature_len):
            globals()[self.args.custom_columns[i]] = row[i]
        # test, question, tag, correct, solvesec = row[0], row[1], row[2], row[3], row[4]

        cate_cols = self.args.custom_columns
        # cate_cols = [test, question, tag, correct, solvesec]

        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cate_cols):
                cate_cols[i] = col[-self.args.max_seq_len :] # 자르기
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else: # 아니면, 그냥 냅두기
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cate_cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cate_cols):
            cate_cols[i] = torch.tensor(col)

        return cate_cols

    def __len__(self):
        return len(self.data)


from torch.nn.utils.rnn import pad_sequence


def collate(batch):
    col_n = len(batch[0])
    col_list = [[] for _ in range(col_n)]
    max_seq_len = len(batch[0][-1])

    # batch의 값들을 각 column끼리 그룹화
    ## padding
    for row in batch:
        for i, col in enumerate(row):
            pre_padded = torch.zeros(max_seq_len)
            pre_padded[-len(col) :] = col # 앞에가 패딩, 뒤에가 값
            col_list[i].append(pre_padded)

    for i, _ in enumerate(col_list):
        col_list[i] = torch.stack(col_list[i])

    return tuple(col_list)


def get_loaders(args, train, valid):

    pin_memory = False
    train_loader, valid_loader = None, None

    if train is not None:
        trainset = DKTDataset(train, args)
        train_loader = torch.utils.data.DataLoader(
            trainset,
            num_workers=args.num_workers,
            shuffle=True,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )
    if valid is not None:
        valset = DKTDataset(valid, args)
        valid_loader = torch.utils.data.DataLoader(
            valset,
            num_workers=args.num_workers,
            shuffle=False,
            batch_size=args.batch_size,
            pin_memory=pin_memory,
            collate_fn=collate,
        )

    return train_loader, valid_loader
