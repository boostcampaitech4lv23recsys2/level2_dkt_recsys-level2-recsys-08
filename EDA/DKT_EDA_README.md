# 탐색적 분석 및 전처리

## A. 데이터 소개

train/test 합쳐서 총 7,442명의 사용자가 존재합니다. 한 행은 한 사용자가 한 문항을 풀었을 때의 정보와 그 문항을 맞췄는지에 대한 정보가 담겨져 있습니다. 데이터는 모두 Timestamp 기준으로 정렬되어 있습니다. 이 때 이 사용자가 푼 마지막 문항의 정답을 맞출 것인지 예측하는 것이 저희의 최종 목표입니다.

- `userID` 사용자의 고유번호입니다.
- `assessmentItemID` 문항의 고유번호입니다.
- `testId` 시험지의 고유번호입니다.
- `answerCode` 사용자가 해당 문항을 맞췄는지 여부에 대한 이진 데이터입니다.
- `Timestamp` 사용자가 해당문항을 풀기 시작한 시점의 데이터입니다.
- `KnowledgeTag` 문항 당 하나씩 배정되는 태그로, 일종의 중분류 역할을 합니다.


## B. 데이터 분석 및 Feature Engineering

|분류|세부 내용|
| --- | --- |
| Base | userID, assessmentItemID, testId, Timestamp, KnowledgeTag |
| Static | user_acc, problem_num, test_mean, test_std, test_sum, tag_mean, tag_std, tag_sum,solvesec_cumsum, solvecumsum_category, big_category_acc, big_category_std, big_category_cumconut, big_category_user_acc,big_category_user_std |
| Category | big_category, mid_category, time_category |
| Time | month, day, hour, dayname, solvetime, solvesec, solvesec_3600, problem_elapsed_time, prob_elapsed_time, normalized_p_elapsed_time, user_prob_elapsed_time_aver, test_prob_elapsed_time_aver, BC_prob_elapsed_time_aver, MC_prob_elapsed_time_aver, KT_prob_elapsed_time_aver, normalized_solvesec, |
| answer | user_correct_answer, user_total_answer, big_category_answer, big_category_answer_log1p, user_kt_ans_rate, prev_answer, prev_answer_rate, correctRatio__by_user, correctRatio__by_test_paper,correctRatio__by_tag, correctRatio__by_prob, correctRatio__by_BC, correctRatio__by_TC |
| Difficulty | elo_assessmentItemID, elo_problem_num |
| Clutering | user_tag_cluster, tag_cluster |
