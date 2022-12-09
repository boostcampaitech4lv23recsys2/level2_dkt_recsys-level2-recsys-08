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
        #🙂2. FE할 때 여기 고치세요! 주의할 점 :범주형 변수에 대해서만 추가해주세요
        cate_cols = ["assessmentItemID", "testId", "KnowledgeTag", "big_category", "mid_category", "problem_num",
                     "assIdx", "month", "day", "hour", "dayname", "time_category", "solvecumsum_category"]
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

    def __feature_engineering(self, df, file_name):
        
        df1 = pd.read_pickle("/opt/ml/input/data/df1_FE.pkl")
        df2 = pd.read_pickle("/opt/ml/input/data/df2_FE.pkl")
        
        if file_name == "train_data.csv":
            return df1
        elif file_name == "test_data.csv":
            return df2
        
        # df2 = df.copy()
        
        # df2['big_category'] = df2.testId.map(lambda x:x[2]).astype(int)
        # df2['mid_category'] = df2.testId.map(lambda x: int(x[-3:]))
        # df2['problem_num'] = df2.assessmentItemID.map(lambda x: int(x[-3:]))
        
        # correct_t = df2.groupby(['testId'])['answerCode'].agg(['mean', 'std'])
        # correct_t.columns = ["test_mean", "test_std"]
        # correct_k = df2.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'std'])
        # correct_k.columns = ["tag_mean", 'tag_std']
        # correct_b = df2.groupby(['big_category'])['answerCode'].agg(['mean', 'std'])
        # correct_b.columns = ["big_mean", 'big_std']

        # df2 = pd.merge(df2, correct_t, on=['testId'], how="left")
        # df2 = pd.merge(df2, correct_k, on=['KnowledgeTag'], how="left")
        # df2 = pd.merge(df2, correct_b, on=['big_category'], how="left")

        # df2['user_correct_answer'] = df2.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
        # df2['user_total_answer'] = df2.groupby('userID')['answerCode'].cumcount()
        # df2['user_acc'] = df2['user_correct_answer']/df2['user_total_answer']
            
        # df2['Timestamp_start'] = pd.to_datetime(df['Timestamp'])
        # df2['Timestamp_fin'] = df2.groupby('userID')['Timestamp_start'].shift(-1)
        # df2['solvetime'] = df2.Timestamp_fin - df2.Timestamp_start
        # df2['solvesec_600'] = df2.solvetime.map(lambda x : x.total_seconds()).shift(1).fillna(0)
        # df2.loc[df2.solvesec_600>=600,'solvesec_600']=0
        # df2.loc[df2.solvesec_600<0,'solvesec_600']=0    
            
        # df2.sort_values(by=['userID','Timestamp'], inplace=True)
        
        # return df2 

    def load_data_from_file(self, args, file_name, is_train=True):
        csv_file_path = os.path.join(self.args.data_dir, file_name)
        df = pd.read_csv(csv_file_path)  # , nrows=100000)
        if is_train == True:
            df = df[df['answerCode'] != -1]
        df = self.__feature_engineering(df, file_name)
        df = self.__preprocessing(df, args, is_train)

        # 추후 feature를 embedding할 시에 embedding_layer의 input 크기를 결정할때 사용
        #🙂3. FE할 때 여기 고치세요! 주의할 점: 2번과정에서 쓴 feature에 대해서만 바꾸세요
        self.args.n_questions = len(
            np.load(os.path.join(self.args.asset_dir, "assessmentItemID_classes.npy"))
        )
        self.args.n_test = len(
            np.load(os.path.join(self.args.asset_dir, "testId_classes.npy"))
        )
        self.args.n_tag = len(
            np.load(os.path.join(self.args.asset_dir, "KnowledgeTag_classes.npy"))
        )
        
        # 'big_category','mid_category','problem_num', 'month', 'dayname'
        self.args.n_big = len(
            np.load(os.path.join(self.args.asset_dir, "big_category_classes.npy"))
        )
        self.args.n_mid = len(
            np.load(os.path.join(self.args.asset_dir, "mid_category_classes.npy"))
        )
        self.args.n_problem = len(
            np.load(os.path.join(self.args.asset_dir, "problem_num_classes.npy"))
        )
        # assIdx, month, day, hour, dayname, time_category, solvecumsum_category
        self.args.n_assIdx = len(
            np.load(os.path.join(self.args.asset_dir, "assIdx_classes.npy"))
        )
        self.args.n_month = len(
            np.load(os.path.join(self.args.asset_dir, "month_classes.npy"))
        )
        self.args.n_day = len(
            np.load(os.path.join(self.args.asset_dir, "day_classes.npy"))
        )
        self.args.n_hour = len(
            np.load(os.path.join(self.args.asset_dir, "hour_classes.npy"))
        )
        self.args.n_dayname = len(
            np.load(os.path.join(self.args.asset_dir, "dayname_classes.npy"))
        )
        self.args.n_time_category = len(
            np.load(os.path.join(self.args.asset_dir, "time_category_classes.npy"))
        )
        self.args.n_solvecumsum_category = len(
            np.load(os.path.join(self.args.asset_dir, "solvecumsum_category_classes.npy"))
        )
        # self.args.n_user_tag_cluster = len(
        #     np.load(os.path.join(self.args.asset_dir, "user_tag_cluster_classes.npy"))
        # )
        
        df = df.sort_values(by=["userID", "Timestamp"], axis=0)
        
        #🙂4. FE할 때 여기 고치세요! 주의할 점: userID와 answerCode 잊지마세요
        columns = ["userID", "assessmentItemID", "testId", "answerCode", "KnowledgeTag", "big_category", "mid_category", "problem_num",
                   "assIdx", "month", "day", "hour", "dayname", "time_category", "solvecumsum_category",
                   "solvesec_3600", "test_mean", 'test_std', "tag_mean", 'tag_std', "big_mean", 'big_std', "user_correct_answer", "user_total_answer", "user_acc",
                   "solvesec_cumsum", "big_category_cumconut", "big_category_user_acc", "big_category_user_std", "big_category_answer", "big_category_answer_log1p", "elo_assessmentItemID"]
        # columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag',
                # 'big_category', 'mid_category', 'problem_num', 'month', 'dayname',
                # 'month_mean', 'solvesec_600', 'test_mean', 'test_std', 'test_sum',
                # 'tag_mean', 'tag_std', 'tag_sum', 'big_mean', 'big_std', 'big_sum', 'user_correct_answer', 'user_total_answer', 'user_acc']
        # columns = ['userID', 'assessmentItemID', 'testId', 'answerCode', 'KnowledgeTag','new_feature']
        
        #🙂5. FE할 때 여기 고치세요! 주의할 점: answerCode 위치는 4번째에 적어주세요
        group = (
            df[columns]
            .groupby("userID")
            .apply(
                lambda r: (
                    r["testId"].values,
                    r["assessmentItemID"].values,
                    r["KnowledgeTag"].values,
                    r["answerCode"].values,
                    r["big_category"].values,
                    r["mid_category"].values,
                    r["problem_num"].values,
                    r["assIdx"].values, 
                    r["month"].values,
                    r["day"].values,
                    r["hour"].values,
                    # r["month_mean"].values, 
                    r["dayname"].values,
                    r["time_category"].values,
                    r["solvecumsum_category"].values,
                    # r["user_tag_cluster"].values,
                    r["solvesec_3600"].values,
                    r["test_mean"].values,
                    r["test_std"].values,
                    # r["test_sum"].values,
                    r["tag_mean"].values,
                    r["tag_std"].values,
                    # r["tag_sum"].values,
                    r["big_mean"].values,
                    r["big_std"].values,
                    # r["big_sum"].values,
                    r['user_correct_answer'].values,
                    r['user_total_answer'].values,
                    r['user_acc'].values, 
                    r['solvesec_cumsum'].values,
                    r['big_category_cumconut'].values,
                    r['big_category_user_acc'].values,
                    r['big_category_user_std'].values,
                    r['big_category_answer'].values,
                    r['big_category_answer_log1p'].values,
                    r['elo_assessmentItemID'].values,
                    # r["new_feature"].values,
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

        #🙂6. FE할 때 여기 고치세요! 주의할 점: 5.과정(group) 순서 그대로 적어주세요!
        test, question, tag, correct, big_category, mid_category, problem_num, \
        assIdx, month, day, hour, dayname, time_category, solvecumsum_category, \
        solvesec_3600, test_mean, test_std, tag_mean, tag_std, big_mean, big_std, user_correct_answer, user_total_answer, user_acc, \
        solvesec_cumsum,  big_category_cumconut, big_category_user_acc, big_category_user_std, big_category_answer, big_category_answer_log1p, elo_assessmentItemID \
        = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8], row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[22], row[23], row[24], row[25], row[26], row[27], row[28], row[29], row[30]
        cols = [test, question, tag, correct, big_category, mid_category, problem_num,
                assIdx, month, day, hour, dayname, time_category, solvecumsum_category,
                solvesec_3600, test_mean, test_std, tag_mean, tag_std, big_mean, big_std, user_correct_answer, user_total_answer, user_acc,
                solvesec_cumsum,  big_category_cumconut, big_category_user_acc, big_category_user_std, big_category_answer, big_category_answer_log1p, elo_assessmentItemID]
        
        # test, question, tag, correct, big, mid, problem, month, dayname = row[0], row[1], row[2], row[3], row[4], row[5], row[6], row[7], row[8]
        # month_mean, solvesec_600, test_mean, test_std, test_sum, tag_mean, tag_std, tag_sum, big_mean, big_std, big_sum, user_correct_answer, user_total_answer, user_acc = row[9], row[10], row[11], row[12], row[13], row[14], row[15], row[16], row[17], row[18], row[19], row[20], row[21], row[22]
        # cols = [test, question, tag, correct, big, mid, problem, month, dayname, month_mean, solvesec_600, test_mean, test_std, test_sum, tag_mean, tag_std, tag_sum, big_mean, big_std, big_sum, user_correct_answer, user_total_answer, user_acc]

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
