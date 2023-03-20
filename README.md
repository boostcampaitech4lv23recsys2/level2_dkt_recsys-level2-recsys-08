# Deep Knowledge Tracing
<div align="center">
  <img width="720" alt="image" src="https://user-images.githubusercontent.com/79351899/226336872-4e1e0d28-8dee-40b3-ba4d-591daca0d6f9.png">

</div>
<div align="center"><br>
</div>

# Contents

- [🏆️ 프로젝트 개요](#️-프로젝트-개요)
- [💻 활용 장비](#-활용-장비)
- [🗓️ 프로젝트 수행 기간](#️-프로젝트-수행-기간)
- [🙋🏻‍♂️🙋🏻‍♀️ 팀 소개 및 역할](#️️-팀-소개-및-역할)
- [🔔 프로젝트 수행 결과](#-프로젝트-수행-결과)
- [📔 프로젝트 결과물](#-프로젝트-결과물)

<br>

# 🏆️ 프로젝트 개요
    
DKT는 Deep Knowledge Tracing의 약자로 이번 DKT 프로젝트를 통해 저희는 "지식 상태"를 추적하는 딥러닝 방법론을 사용하여 학생의 이해도를 측정하고 미래 학습 예측을 목표로 하였습니다.
    
   ![dkt_image](https://user-images.githubusercontent.com/79351899/226337521-3f6c1286-bb43-4ed5-a2cd-eb29f45bcc66.png)
    
    
<br>

# 💻 활용 장비

- 개발환경 : vscode, Jupyter
- 개발 언어 : Python (Pytorch)
- GPU : V100
- 협업툴 : GitHub, Wandb, MLflow
- 의사소통툴 : Slack, Notion, Zoom, Trello, Gather Town

<br>

# 🗓️ 프로젝트 수행 기간 
- 2022.11 ~ 2022.12  

<br>

# 🙋🏻‍♂️🙋🏻‍♀️ 팀 소개 및 역할
<table align="center">
    <tr>
        <td align="center"><b>정의준</b></td>
        <td align="center"><b>채민수</b></td>
        <td align="center"><b>전해리</b></td>
        <td align="center"><b>이나현</b></td>
        <td align="center"><b>조원준</b></td>
    </tr>
    <tr height="160px">
        <td align="center">
            <img height="120px" weight="120px" src="https://avatars.githubusercontent.com/u/71438046?v=4"/>
        </td>
        <td align="center">
            <img height="120px" weight="120px" src="https://avatars.githubusercontent.com/u/79351899?v=4"/>
        <td align="center">
            <img height="120px" weight="120px" src="https://avatars.githubusercontent.com/u/98138782?v=4"/>
        </td>
        </td>
        <td align="center">
            <img height="120px" weight="120px" src="https://avatars.githubusercontent.com/u/90559493?v=4"/>
        </td>
        <td align="center">
            <img height="120px" weight="120px" src="https://avatars.githubusercontent.com/u/57648890?v=4"/>
        </td>
    </tr>
    <tr>
    </tr>
    <tr>
        <td align="center"><a href="https://github.com/uijoon"><img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white"/></a></td>
        <td align="center"><a href="https://github.com/chaeminsoo"><img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white"/></a></td>
        <td align="center"><a href="https://github.com/jeonhaelee"><img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white"/></a></td>
        <td align="center"><a href="https://github.com/lnh31"><img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white"/></a></td>
        <td align="center"><a href="https://github.com/netsus"><img src="https://img.shields.io/badge/GitHub-181717?style=flat&logo=github&logoColor=white"/></a></td>
    </tr>

</table>

<div align="center">


| 전체 | 문제 정의, 계획 수립, 목표 설정, EDA, Feature Engineering, 모델 실험 |
| --- | --- |
| 정의준 | TabNet, wandb, Feature Selection, Stacking Ensemble |
| 채민수 | SAKT 구현, CatBoost 구현, Feature Selection |
| 전해리 | lightGCN+lqTransfomer 구현, lightGCN+Seq model 구현, Feature Selection |
| 이나현 | LastQuery Transformer 구현, Feature Selection |
| 조원준 | CV Strategy, LGBM, XGBoost, MLflow, Feature Selection,  Feature Importance, Ensemble |


</div>

<br>



<br>

# 🔔 프로젝트 수행 결과

- [EDA](https://github.com/boostcampaitech4lv23recsys2/level2_dkt_recsys-level2-recsys-08/tree/master/EDA)

- [Model]()

- [MLFlow](https://github.com/boostcampaitech4lv23recsys2/level2_dkt_recsys-level2-recsys-08/tree/master/MLflow)

<br>

## A. **모델별 시연 결과**

| Model | Details | AUROC (제출) | Accuracy (제출) |
| --- | --- | --- | --- |
| SAKT | userID, assessmentItemID, answerCode 만 사용, Output에 sigmoid 적용 | 0.6425 | 0.6075 |
| CatBoost | 21개 피쳐 사용, learning rate = 0.01, iteration = 10000 | 0.8456 | 0.7769 |
| TabNet | Sweep, Feature Selection | 0.8160 | 0.6989 |
| LSTM | 16개 피처 사용, 연속형변수 batchnorm, data augmentation | 0.7828 | 0.7124 |
| LSTMATTN | 16개 피처 사용, 연속형변수 batchnorm, data augmentation | 0.7747 | 0.7177 |
| LastQuery Transformer | 16개 피처 사용, 연속형변수 batchnorm, data augmentation | 0.7785 | 0.7151 |
| LGCN+LSTM | LGCN 임베딩(assessmentItemID, testId) 사용, Sweep | 0.7915 | 0.7339 |
| LGCN+lqTransfomer | LGCN 임베딩(assessmentItemID, testId) 사용, Sweep, Feature Selection | 0.8032 | 0.7285 |
| LGBM | 27개 피처 사용, 주요 피처 - Time Lag 관련 피처, learning rate = 0.023 | 0.8507 | 0.7715 |
| XGBoost | 32개 피처 사용, 주요 피처 - elo Rating, learning rate = 0.023 | 0.8481 | 0.7661 |

<br>

## B. 최종 결과

<br>

**앙상블**

각 모델의 예측값에 가중치를 부여하는 Weighted Voting을 사용하였습니다.

LGBM+CatBoost, LGBM, XGBM, TabNet, LSTM, LGCN (가중치 : 40 - 30 - 15 - 5 - 5 - 5)

**최종 LB AUC : 0.8486  LB Accuracy : 0.7823**

<br>


<br><br>

# 📔 프로젝트 결과물

* [Wrap-Up Report](https://github.com/boostcampaitech4lv23recsys2/level2_dkt_recsys-level2-recsys-08/blob/master/RecSys_DKT_Wrap_UP_%E1%84%85%E1%85%B5%E1%84%91%E1%85%A9%E1%84%90%E1%85%B3.pdf)<br><br>

<br>
