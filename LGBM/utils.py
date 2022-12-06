import random
import pandas as pd
import numpy as np
import os
from dotenv import load_dotenv
import subprocess as sp
import re
from tqdm import tqdm

load_dotenv('../.env')

def custom_train_test_split(df, ratio=0.7, seed=41, split=True):
    """df를 입력받아서, userid들을 랜덤해서 1-ratio만큼의 user만 선별하고, 마지막 제출껀만 만들어서 train, test로 리턴 """
    random.seed(seed)
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
    return total_preds


def post_slack(message):
    API=os.environ['API']
    NAME='cwj'
    message=f'curl -s -d "payload={{\\"username\\":\\"{NAME}\\", \\"text\\":\\"\`\`\`{message}\`\`\`\\"}}" "{API}"'
    # post message
    sp.getstatusoutput(message)


def title2filename(title):
    new_str = re.sub(r"[^\uAC00-\uD7A30-9a-zA-Z\s]", " ", title)
    file_str = '_'.join(new_str.split())
    return file_str


def apply_elo_granularity_feature_name(df, granularity_feature_name):
    def get_new_theta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
        return theta + learning_rate_theta(nb_previous_answers) * (
            is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
        )

    def get_new_beta(is_good_answer, beta, left_asymptote, theta, nb_previous_answers):
        return beta - learning_rate_beta(nb_previous_answers) * (
            is_good_answer - probability_of_good_answer(theta, beta, left_asymptote)
        )

    def learning_rate_theta(nb_answers):
        return max(0.3 / (1 + 0.01 * nb_answers), 0.04)

    def learning_rate_beta(nb_answers):
        return 1 / (1 + 0.05 * nb_answers)

    def probability_of_good_answer(theta, beta, left_asymptote):
        return left_asymptote + (1 - left_asymptote) * sigmoid(theta - beta)

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def estimate_parameters(answers_df, granularity_feature_name=granularity_feature_name):
        item_parameters = {
            granularity_feature_value: {"beta": 0, "nb_answers": 0}
            for granularity_feature_value in np.unique(
                answers_df[granularity_feature_name]
            )
        }
        student_parameters = {
            student_id: {"theta": 0, "nb_answers": 0}
            for student_id in np.unique(answers_df.userID)
        }

        print("Parameter estimation is starting...", flush=True)

        for student_id, item_id, left_asymptote, answered_correctly in tqdm(
            zip(
                answers_df.userID.values,
                answers_df[granularity_feature_name].values,
                answers_df.left_asymptote.values,
                answers_df.answerCode.values,
            ),
            total=len(answers_df),
        ):
            theta = student_parameters[student_id]["theta"]
            beta = item_parameters[item_id]["beta"]

            item_parameters[item_id]["beta"] = get_new_beta(
                answered_correctly,
                beta,
                left_asymptote,
                theta,
                item_parameters[item_id]["nb_answers"],
            )
            student_parameters[student_id]["theta"] = get_new_theta(
                answered_correctly,
                beta,
                left_asymptote,
                theta,
                student_parameters[student_id]["nb_answers"],
            )

            item_parameters[item_id]["nb_answers"] += 1
            student_parameters[student_id]["nb_answers"] += 1

        print(f"Theta & beta estimations on {granularity_feature_name} are completed.")
        return student_parameters, item_parameters

    def gou_func(theta, beta):
        return 1 / (1 + np.exp(-(theta - beta)))

    df["left_asymptote"] = 0

    print(f"Dataset of shape {df.shape}")
    print(f"Columns are {list(df.columns)}")

    student_parameters, item_parameters = estimate_parameters(df)

    prob = [
        gou_func(student_parameters[student]["theta"], item_parameters[item]["beta"])
        for student, item in zip(df.userID.values, df[granularity_feature_name].values)
    ]

    df[f"elo_{granularity_feature_name}"] = prob

    return df