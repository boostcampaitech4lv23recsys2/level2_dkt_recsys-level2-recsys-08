
import pandas as pd
import numpy as np

import time
from datetime import datetime

import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import random
import sys
import os
plt.style.use('seaborn')
from IPython.core.interactiveshell import InteractiveShell
InteractiveShell.ast_node_interactivity = "all"
# import missingno
import pandas as pd
pd.set_option('display.min_rows', 500)
import warnings
warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))))
from LGBM.utils import apply_elo_granularity_feature_name

import time
from datetime import datetime

day_dict = {'Tuesday': 0,
'Thursday': 1,
'Friday': 2,
'Wednesday' : 3,
'Monday': 4,
'Saturday': 5,
'Sunday': 6}

def convert_time(s):
    timestamp = time.mktime(
        datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
    )
    return int(timestamp)

def get_month(s):
    return s.month

def get_dayname(s):
    return s.dayofweek

df2 = df.copy()
#유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
df2.sort_values(by=['userID','Timestamp'], inplace=True)

df2['big_category'] = df2.testId.map(lambda x:x[2]).astype(int)
df2['mid_category'] = df2.testId.map(lambda x: int(x[-3:]))
df2['problem_num'] = df2.assessmentItemID.map(lambda x: int(x[-3:]))

df2['month'] = pd.to_datetime(df2.Timestamp, format="%Y-%m-%d %H:%M:%S").apply(get_month)
df2['month_mean'] = df2.groupby(['month'])['answerCode'].agg(['mean'])
# correct_m = df2.groupby(['month'])['answerCode'].agg(['mean'])
# correct_m.columns = ['month_mean']
# df2 = pd.merge(df2, correct_m, on=['month'], how="left")

df2['dayname'] = pd.to_datetime(df2.Timestamp, format="%Y-%m-%d %H:%M:%S").apply(get_dayname)

df2['Timestamp_start'] = pd.to_datetime(df['Timestamp'])
df2['Timestamp_fin'] = df2.groupby('userID')['Timestamp_start'].shift(-1)
df2['solvetime'] = df2.Timestamp_fin - df2.Timestamp_start
df2['solvesec_600'] = df2.solvetime.map(lambda x : x.total_seconds()).shift(1).fillna(0)
# df2['solvesec_cat'] = pd.to_datetime(df2.Timestamp).dt.day_name().map(day_dict)
df2.loc[df2.solvesec_600>=600,'solvesec_600']=0
df2.loc[df2.solvesec_600<0,'solvesec_600']=0

correct_t = df2.groupby(['testId'])['answerCode'].agg(['mean', 'std', 'sum'])
correct_t.columns = ["test_mean", "test_std", 'test_sum']
correct_k = df2.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'std', 'sum'])
correct_k.columns = ["tag_mean", 'tag_std', 'tag_sum']
correct_b = df2.groupby(['big_category'])['answerCode'].agg(['mean', 'std', 'sum'])
correct_b.columns = ["big_mean", 'big_std', 'big_sum']

df2 = pd.merge(df2, correct_t, on=['testId'], how="left")
df2 = pd.merge(df2, correct_k, on=['KnowledgeTag'], how="left")
df2 = pd.merge(df2, correct_b, on=['big_category'], how="left")

df2['user_correct_answer'] = df2.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
df2['user_total_answer'] = df2.groupby('userID')['answerCode'].cumcount()
df2['user_acc'] = df2['user_correct_answer']/df2['user_total_answer']
    
df2.sort_values(by=['userID','Timestamp'], inplace=True)
# df["Timestamp"] = df["Timestamp"].apply(convert_time)


# day_dict = {'Tuesday': 0,
#  'Thursday': 1,
#  'Friday': 2,
#  'Wednesday' : 3,
#  'Monday': 4,
#  'Saturday': 5,
#  'Sunday': 6}

# def convert_time(s):
#     timestamp = time.mktime(
#         datetime.strptime(s, "%Y-%m-%d %H:%M:%S").timetuple()
#     )
#     return int(timestamp)

# def feature_engineering(df):
#     # big_category, mid_category, problem_num, month, dayname, Timestamp_start, Timestamp_fin, solvetime, solvesec_600,
#     # test_mean, test_std, test_sum, tag_mean, tag_std, tad_sum, big_mean, big_std, big_sum
#     uid2idx = {v:k for k,v in enumerate(sorted(df.userID.unique()))}
#     ass2idx = {v:k for k,v in enumerate(sorted(df.assessmentItemID.unique()))}
#     test2idx = {v:k for k,v in enumerate(sorted(df.testId.unique()))}
    
#     df2 = df.copy()
#     #유저별 시퀀스를 고려하기 위해 아래와 같이 정렬
#     df2.sort_values(by=['userID','Timestamp'], inplace=True)
    
#     # df2['big_category'] = df2.testId.map(lambda x:x[2]).astype(int)
#     # df2['mid_category'] = df2.testId.map(lambda x: int(x[-3:]))
#     # df2['problem_num'] = df2.assessmentItemID.map(lambda x: int(x[-3:]))
    
#     # df2['month'] = pd.to_datetime(df2.Timestamp).dt.month
#     # correct_m = df2.groupby(['month'])['answerCode'].agg(['mean'])
#     # correct_m.columns = ['month_mean']
#     # df2 = pd.merge(df2, correct_m, on=['month'], how="left")
    
#     # df2['dayname'] = pd.to_datetime(df2.Timestamp).dt.day_name().map(day_dict)
    
#     # df2['Timestamp_start'] = pd.to_datetime(df['Timestamp'])
#     # df2['Timestamp_fin'] = df2.groupby('userID')['Timestamp_start'].shift(-1)
#     # df2['solvetime'] = df2.Timestamp_fin - df2.Timestamp_start
#     # df2['solvesec_600'] = df2.solvetime.map(lambda x : x.total_seconds()).shift(1).fillna(0)
#     # # df2['solvesec_cat'] = pd.to_datetime(df2.Timestamp).dt.day_name().map(day_dict)
#     # df2.loc[df2.solvesec_600>=600,'solvesec_600']=0
#     # df2.loc[df2.solvesec_600<0,'solvesec_600']=0
    
#     # correct_t = df2.groupby(['testId'])['answerCode'].agg(['mean', 'std', 'sum'])
#     # correct_t.columns = ["test_mean", "test_std", 'test_sum']
#     # correct_k = df2.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'std', 'sum'])
#     # correct_k.columns = ["tag_mean", 'tag_std', 'tag_sum']
#     # correct_b = df2.groupby(['big_category'])['answerCode'].agg(['mean', 'std', 'sum'])
#     # correct_b.columns = ["big_mean", 'big_std', 'big_sum']
    
#     # df2 = pd.merge(df2, correct_t, on=['testId'], how="left")
#     # df2 = pd.merge(df2, correct_k, on=['KnowledgeTag'], how="left")
#     # df2 = pd.merge(df2, correct_b, on=['big_category'], how="left")
    

#     # df["Timestamp"] = df["Timestamp"].apply(convert_time)
    
    
#     ### 여기부턴 원준님 base fe
#     # assIdx, big_category_user_cum_acc, big_category_cumconut, user_acc, 
#     # user_correct_answer, day, time_category, elo_assessmentItemID, hour, problem_num, assess_count, user_tag_cluster
    
#     df2['uidIdx'] = df2.userID.map(uid2idx)
#     df2['assIdx'] = df2.assessmentItemID.map(ass2idx)
#     df2['testIdx'] = df2.testId.map(test2idx)

#     #유저들의 문제 풀이수, 정답 수, 정답률을 시간순으로 누적해서 계산
#     df2['user_correct_answer'] = df2.groupby('userID')['answerCode'].transform(lambda x: x.cumsum().shift(1))
#     df2['user_total_answer'] = df2.groupby('userID')['answerCode'].cumcount()
#     df2['user_acc'] = df2['user_correct_answer']/df2['user_total_answer']
#     df2['month'] = pd.to_datetime(df2.Timestamp).dt.month
#     df2['day'] = pd.to_datetime(df2.Timestamp).dt.day
#     df2['hour'] = pd.to_datetime(df2.Timestamp).dt.hour
#     df2['dayname'] = pd.to_datetime(df2.Timestamp).dt.day_name().map(day_dict)
#     df2['big_category'] = df2.testId.map(lambda x:x[2]).astype(int)
#     df2['problem_num'] = df2.assessmentItemID.map(lambda x: int(x[-3:]))
#     df2['mid_category'] = df2.testId.map(lambda x: int(x[-3:]))

#     # testId와 KnowledgeTag의 전체 정답률은 한번에 계산
#     # 아래 데이터는 제출용 데이터셋에 대해서도 재사용
#     correct_t = df2.groupby(['testId'])['answerCode'].agg(['mean', 'std', 'sum'])
#     correct_t.columns = ["test_mean", "test_std", 'test_sum']
#     correct_k = df2.groupby(['KnowledgeTag'])['answerCode'].agg(['mean', 'std', 'sum'])
#     correct_k.columns = ["tag_mean", 'tag_std', 'tag_sum']

#     df2 = pd.merge(df2, correct_t, on=['testId'], how="left")
#     df2 = pd.merge(df2, correct_k, on=['KnowledgeTag'], how="left")
    
#     # 유저별 문제푼 시간, solvesec_3600, time_category
#     df2['Timestamp2'] = pd.to_datetime(df2.Timestamp)
#     df2['solvetime'] = df2.groupby('userID')['Timestamp2'].diff().fillna(pd.Timedelta(seconds=0))
#     df2['solvesec'] = df2.solvetime.map(lambda x : x.total_seconds())
#     df2['solvesec_3600'] = df2.solvesec
#     df2.loc[df2.solvesec>=3600,'solvesec_3600']=3600

#     df2['time_category'] = ''
#     tc = [0,5,7,10,60,600,1200,2400,3600]
#     df2.loc[(df2.solvesec==0), 'time_category'] = "0 - [0,0]"
#     for i in range(len(tc)-1):
#         s,e = tc[i],tc[i+1]
#         df2.loc[(df2.solvesec>s) & (df2.solvesec<=e),'time_category']=f"{i+1} - ({s}, {e}]"
#     df2.loc[(df2.solvesec>=tc[-1]),'time_category'] = f"{i+2} - ({e}, )"
#     timecat2idx={k:v for v,k in enumerate(sorted(df2.time_category.unique()))}
#     df2['time_category'] = df2.time_category.map(timecat2idx)
    
#     # 유저별 문제푼 시간 Cumsum
#     df2['solvesec_cumsum'] = df2.groupby(['userID','testId'])['solvesec_3600'].cumsum()%3601
#     df2['solvecumsum_category'] = ''
#     tc = [0,5,7,10,60,600,1200,2400,3600,7200]
#     df2.loc[(df2.solvesec_cumsum==0), 'solvecumsum_category'] = "0 - [0,0]"
#     for i in range(len(tc)-1):
#         s,e = tc[i],tc[i+1]
#         df2.loc[(df2.solvesec_cumsum>s) & (df2.solvesec_cumsum<=e),'solvecumsum_category']=f"{i+1} - ({s}, {e}]"
#     df2.loc[(df2.solvesec_cumsum>=tc[-1]),'solvecumsum_category'] = f"{i+2} - ({e}, )"
#     solvecumsum_category2idx={k:v for v,k in enumerate(sorted(df2.solvecumsum_category.unique()))}
#     df2['solvecumsum_category'] = df2.solvecumsum_category.map(solvecumsum_category2idx) 
    
#     # 유저별 big category 문제 푼 횟수, 맞춤 횟수, 누적 정답률
#     df2['big_category_cumconut'] = df2.groupby(['userID','big_category']).answerCode.cumcount()
#     df2['big_category_answer'] = df2.groupby(['userID','big_category']).answerCode.transform(lambda x: x.cumsum().shift(1)).fillna(0)
#     df2['big_category_user_cum_acc'] = (df2['big_category_answer'] / df2['big_category_cumconut']).fillna(0) 
    
#     # 유저별 mid category 문제 푼 횟수, 맞춤 횟수, 누적 정답률
#     df2['mid_category_cumconut'] = df2.groupby(['userID','mid_category']).answerCode.cumcount()
#     df2['mid_category_answer'] = df2.groupby(['userID','mid_category']).answerCode.transform(lambda x: x.cumsum().shift(1)).fillna(0)
#     df2['mid_category_user_cum_acc'] = (df2['mid_category_answer'] / df2['mid_category_cumconut']).fillna(0)
    
#     ass_acc_dict = dict(df2[df2.answerCode!=-1].groupby('assessmentItemID').answerCode.mean())
#     df2['ass_acc_mean'] = df2.assessmentItemID.map(ass_acc_dict)
#     df2['ass_difficulty'] = 1 - df2['ass_acc_mean']
    
#     ass_acc_std_dict = dict(df2[df2.answerCode!=-1].groupby('assessmentItemID').answerCode.std())
#     df2['ass_acc_std'] = df2.assessmentItemID.map(ass_acc_std_dict)

#     ### 문제 번호별 난이도
#     pb_num_dict = dict(df2[df2.answerCode!=-1].groupby('problem_num').answerCode.mean())
#     df2['pb_num_acc_mean'] = df2.problem_num.map(pb_num_dict)
#     df2['pb_num_difficulty'] = 1 - df2['pb_num_acc_mean']
    
#     pb_num_std_dict = dict(df2[df2.answerCode!=-1].groupby('problem_num').answerCode.std())
#     df2['pb_num_acc_std'] = df2.problem_num.map(pb_num_std_dict)
    
#     ## assess_count
#     df2['assess_count'] = df2.groupby(['userID','assessmentItemID']).answerCode.cumcount()
    
#     df2 = apply_elo_granularity_feature_name(df2, "assessmentItemID")
    
#     df2.sort_values(by=['userID','Timestamp'], inplace=True)    

#     return df2
