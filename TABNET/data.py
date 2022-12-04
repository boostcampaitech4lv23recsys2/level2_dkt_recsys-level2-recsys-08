import pandas as pd
import time
import numpy as np
import random
from sklearn.preprocessing import LabelEncoder

def percentile(s):
    return np.sum(s) / len(s)

def categorical_feature(data : pd.DataFrame,columns : list) -> pd.DataFrame:
    for col in columns:
        X = data[col]
        enc = LabelEncoder()
        enc.fit(X)
        data[col] = enc.transform(X)
    return data

def assess_count(data):
    data['cnt'] = 1
    data['assess_count'] = data.groupby(['userID', 'assessmentItemID'])['cnt'].cumsum()
    return data.drop(columns = 'cnt')

def feature_engineering(df):
    #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
    df.sort_values(by=['userID','Timestamp'], inplace=True)
    
    #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
    ## 유저별 맞춘 개수를 한칸 아래로 땡기기(shift 1)
    df['user_correct_answer'] = df.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
    ## 유저별 문제 푼 개수 cumcount
    df['user_total_answer'] = df.groupby('userID')['answerCode'].cumcount()
    # 유저별 푼 문제수에 따른 정확도
    df['user_acc'] = df['user_correct_answer']/df['user_total_answer']

    # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
    # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
    
    ## 시험지별 정답률과 맞춘 횟수
    correct_t = df.groupby(['testId'])['answerCode'].agg(['mean', 'sum'])
    correct_t.columns = ["test_mean", 'test_sum']
    ## 문제 유형별 정답률과 맞춘 횟수
    correct_k = df.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'sum'])
    correct_k.columns = ["tag_mean", 'tag_sum']

    ## df에 시험지별 정답률과 맞춘 횟수 추가
    df = pd.merge(df, correct_t, on=['testId'], how="left")
    ## df에 문제 유형별 정답률과 맞춘 횟수 추가
    df = pd.merge(df, correct_k, on=['KnowledgeTag'], how="left")
    
    return df

def time_feature_engineering(df):
    df['Time'] = df['Timestamp'].apply(lambda x: int(time.mktime(x.timetuple())))
    df['Timediff'] = df.groupby(['userID','testId','assess_count'])['Time'].diff()
    df = df.fillna(0)
    df['Timediff'] = df['Timediff'].apply(lambda x : x if x < 650 else 650)
    df['Timepassed'] = df.groupby(['userID','testId','assess_count'])['Timediff'].cumsum()
    timediff = df.groupby('Timediff').agg({'answerCode':percentile}).reset_index()
    timediff.columns =['Timediff','Time_answer_rate']
    df = df.merge(timediff,how='left',on='Timediff')

    return df

# train과 test 데이터셋은 사용자 별로 묶어서 분리를 해주어야함
def custom_train_test_split(df, ratio=0.7, split=True):
    
    users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
    random.shuffle(users)
    
    max_train_data_len = ratio*len(df)
    sum_of_train_data = 0
    user_ids =[]

    for user_id, count in users:
        sum_of_train_data += count
        if max_train_data_len < sum_of_train_data:
            break
        user_ids.append(user_id)


    train = df[df['userID'].isin(user_ids)]
    test = df[df['userID'].isin(user_ids) == False]

    #test데이터셋은 각 유저의 마지막 interaction만 추출
    test = test[test['userID'] != test['userID'].shift(-1)]
    return train, test