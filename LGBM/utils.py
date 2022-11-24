import random
import pandas as pd
import os
from dotenv import load_dotenv
import subprocess as sp

load_dotenv('../.env')
breakpoint()

random.seed(42)
def custom_train_test_split(df, ratio=0.7, split=True):
    """df를 입력받아서, userid들을 랜덤해서 1-ratio만큼의 user만 선별하고, 마지막 제출껀만 만들어서 train, test로 리턴 """
    
    ## (user_id, 문제푼 횟수)를 원소로 갖는 리스트
    users = list(zip(df['userID'].value_counts().index, df['userID'].value_counts()))
    random.shuffle(users) # 셔플
    
    max_train_data_len = ratio*len(df) # train data 길이
    sum_of_train_data = 0
    user_ids =[]

    for user_id, count in users: # for문으로 전체의 ratio 퍼센트만 user_ids에 추가
        sum_of_train_data += count
        if max_train_data_len < sum_of_train_data:
            break
        user_ids.append(user_id)

    ## train, test split
    train = df[df['userID'].isin(user_ids)] # train
    test = df[df['userID'].isin(user_ids) == False] # test

    #test데이터셋은, train에서 각 유저의 마지막 interaction만 추출
    test = test[test['userID'] != test['userID'].shift(-1)]
    return train, test


def lgbm_predict(test_df, model, FEATS, submission):
    """test_df(FE된 테스트 전체 데이터), model, FEATS, submission(test.csv 형식) 입력받아서 예측하고 output/{submission}.csv 저장"""
    # LEAVE LAST INTERACTION ONLY
    test_df = test_df[test_df['userID'] != test_df['userID'].shift(-1)]

    # DROP ANSWERCODE
    test_df = test_df.drop(['answerCode'], axis=1)

    # MAKE PREDICTION
    if 'classifier' in model.__doc__: # LGBM doc: 'LightGBM classifier.'
        total_preds = model.predict_proba(test_df[FEATS])[:, 1]
    elif 'Booster' in model.__doc__: # lgb doc: 'Booster in LightGBM.'
        total_preds = model.predict(test_df[FEATS])
    else:
        raise "Unable Model Please get LGBM Model"

    # SAVE OUTPUT
    output_dir = 'output/'
    write_path = os.path.join(output_dir, f"{submission}")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with open(write_path, 'w', encoding='utf8') as w:
        print("writing prediction : {}".format(write_path))
        w.write("id,prediction\n")
        for id, p in enumerate(total_preds):
            w.write('{},{}\n'.format(id,p))


def post_slack(message):
    API=os.environ['API']
    NAME='cwj'
    message=f'curl -s -d "payload={{\\"username\\":\\"{NAME}\\", \\"text\\":\\"\`\`\`{message}\`\`\`\\"}}" "{API}"'
    # post message
    sp.getstatusoutput(message)