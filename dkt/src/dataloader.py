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
        # cate_cols = ["assessmentItemID", "testId", "KnowledgeTag"]
        cate_cols = args.cat_cols
        if args.partial_user: #640명에 대해서 자른다.
            df = df[df['userID'] < 717]
        if not os.path.exists(self.args.asset_dir):
            os.makedirs(self.args.asset_dir)
        all_df = pd.read_csv("/opt/ml/input/main_dir/dkt/asset/all_fe_df.csv")
        
        for col in args.cat_cols:
            exec("self.args.n_" + col + '= all_df["' + col + '"].nunique()')
        
        # for col in cate_cols:
        #     exec(col + '2idx = {v:k for k,v in enumerate(all_df["' + col + '"].unique())}')
        #     exec('df["' + col + '"] = df["' + col + '"].map(' + col + '2idx)')

        # for col in cate_cols:

        #     le = LabelEncoder()
        #     if is_train:
        #         # For UNKNOWN class
        #         a = df[col].unique().tolist() + ["unknown"]
        #         le.fit(a)
        #         self.__save_labels(le, col)
        #     else:
        #         label_path = os.path.join(self.args.asset_dir, col + "_classes.npy")
        #         le.classes_ = np.load(label_path)

        #         df[col] = df[col].apply(
        #             lambda x: x if str(x) in le.classes_ else "unknown"
        #         )

        #     # 모든 컬럼이 범주형이라고 가정
        #     df[col] = df[col].astype(str)
        #     test = le.transform(df[col])
        #     df[col] = test

        # def convert_time(s):
        #     timestamp = time.mktime(
        #         datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
        #     )
        #     return int(timestamp)

        # df["Timestamp"] = df["Timestamp"].apply(convert_time)

        return df


    def __feature_engineering(self, df, args, is_train):
        if is_train == True and args.train_df_csv:
            df = pd.read_csv(args.train_df_csv)
            return df[args.base_cols + args.cat_cols + args.num_cols]
        elif is_train == False and args.test_df_csv:
            df = pd.read_csv(args.test_df_csv)
            return df[args.base_cols + args.cat_cols + args.num_cols]

        return df

    def load_data_from_file(self, args, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        if is_train == True:
            df = df[df['answerCode'] != -1]
        df = self.__feature_engineering(df, args, is_train)
        df = self.__preprocessing(df, args, is_train)


        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
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
        
        #🙂2. FE할 때 여기 고치세요! 주의할 점 : userID와 answerCode 잊지마세요
        columns = ['userID', 'answerCode'] + args.cat_cols + args.num_cols
        # columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag','new_feature']
        
        #🙂3. FE할 때 여기 고치세요! 주의할 점 : answerCode 위치는 4번째에 적어주세요
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["assessmentItemID"].values,
                    r["testId"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
                    r["big_category"].values,
                    r["mid_category"].values,
                    r["problem_num"].values,
                    r["time_category"].values,
                    r["solvecumsum_category"].values,

                    r["solvesec_3600"].values,
                    r["solvesec_cumsum"].values,
                    r["test_mean"].values,
                    r["test_std"].values,
                    r["tag_mean"].values,
                    r["tag_std"].values,
                    r["big_mean"].values,
                    r["big_std"].values,
                    r["big_sum"].values,
                    r["assess_mean"].values,
                    r["assess_std"].values,
                    r["user_mean"].values,
                    r["user_std"].values,
                    r["user_sum"].values,
                    r["assess_count"].values,
                    
                )
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
        
        #🙂4. FE할 때 여기 고치세요! 주의할 점 : 3.과정(group) 순서 그대로 적어주세요!
        assessmentItemID, testId, KnowledgeTag, answerCode, \
        big_category, mid_category, problem_num, time_category, solvecumsum_category, \
        solvesec_3600, solvesec_cumsum, test_mean, test_std, \
        tag_mean, tag_std, big_mean, big_std, big_sum, assess_mean, assess_std, \
        user_mean, user_std, user_sum, assess_count = row

        cols = [assessmentItemID, testId, KnowledgeTag, answerCode, \
        big_category, mid_category, problem_num, time_category, solvecumsum_category, \
        solvesec_3600, solvesec_cumsum, test_mean, test_std, \
        tag_mean, tag_std, big_mean, big_std, big_sum, assess_mean, assess_std, \
        user_mean, user_std, user_sum, assess_count]
        
        #test, question, tag, correct, new_feature = row[0], row[1], row[2], row[3], row[4]
        #cols = [test, question, tag, correct, new_feature]


        # max seq len을 고려하여서 이보다 길면 자르고 아닐 경우 그대로 냅둔다
        if seq_len > self.args.max_seq_len:
            for i, col in enumerate(cols):
                cols[i] = col[-self.args.max_seq_len :] # 자르기
            mask = np.ones(self.args.max_seq_len, dtype=np.int16)
        else: # 아니면, 그냥 냅두기
            mask = np.zeros(self.args.max_seq_len, dtype=np.int16)
            mask[-seq_len:] = 1

        # mask도 columns 목록에 포함시킴
        cols.append(mask)

        # np.array -> torch.tensor 형변환
        for i, col in enumerate(cols):
            cols[i] = torch.tensor(col)

        return cols

    def __len__(self):
        return len(self.data)


def data_augmentation(data, args):
    if args.window == True:
        data = slidding_window(data, args)

    return data


def slidding_window(data, args):
    window_size = args.max_seq_len
    stride = 1000
    shuffle_tf = False
    print("data_augmentation 적용!, stride = "+str(stride))

    augmented_datas = []
    for row in data:
        seq_len = len(row[0])

        # 만약 window 크기보다 seq len이 같거나 작으면 augmentation을 하지 않는다
        if seq_len <= window_size:
            augmented_datas.append(row)
        else:
            total_window = ((seq_len - window_size) // stride) + 1
            
            # 앞에서부터 slidding window 적용
            for window_i in range(total_window):
                # window로 잘린 데이터를 모으는 리스트
                window_data = []
                for col in row:
                    window_data.append(col[window_i*stride:window_i*stride + window_size])

                # Shuffle
                # 마지막 데이터의 경우 shuffle을 하지 않는다
                if shuffle_tf and window_i + 1 != total_window:
                    shuffle_datas = shuffle(window_data, window_size, args)
                    augmented_datas += shuffle_datas
                else:
                    augmented_datas.append(tuple(window_data))

            # slidding window에서 뒷부분이 누락될 경우 추가
            total_len = window_size + (stride * (total_window - 1))
            if seq_len != total_len:
                window_data = []
                for col in row:
                    window_data.append(col[-window_size:])
                augmented_datas.append(tuple(window_data))


    return augmented_datas

def shuffle(data, data_size, args):
    shuffle_n = 3
    shuffle_datas = []
    for i in range(shuffle_n):
        # shuffle 횟수만큼 window를 랜덤하게 계속 섞어서 데이터로 추가
        shuffle_data = []
        random_index = np.random.permutation(data_size)
        for col in data:
            shuffle_data.append(col[random_index])
        shuffle_datas.append(tuple(shuffle_data))
    return shuffle_datas


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
