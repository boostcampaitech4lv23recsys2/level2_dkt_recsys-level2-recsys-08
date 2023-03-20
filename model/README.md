# Model

## A. 모델 개요

<img width="734" alt="image" src="https://user-images.githubusercontent.com/98138782/226339599-49f8e5b3-69ba-4d2f-af1b-3cfc710f8254.png">

<br>

## B. 모델 선정 및 분석


- SAKT
    
    해당 대회의 목적은 학생들의 과거 학습 기록을 바탕으로 미래에 접하게될 문제의 정오답을 예측하는 것이다. 이때 이전에 풀었던 문제들 과의 연관성을 Transformer의 attention mechanism으로 학습에 반영한다면 좋은 성능을 모일것으로 판단했다. 따라서 동일한 아이디어에서 출발한 SAKT 모델을 적용해보기로 경정했다. 적용 결과 기본적인 구조에서 아쉬운 성능을 모여주어 고도화하여 사용하기로 하였으나 반복되는 에러 발생으로 최종적으로 체택하지 않았다.
    
- CatBoost
    
    Feature Engineering을 하는 과정에서 다양한 categorical feature를 추가로 생성하였고 범주형 변수가 많아짐에 따라 CatBoost 모델을 사용할 경우 높은 예측 성능을 가질 것으로 기대하여 채택하였다. 적용 결과 다른 Boosting 모델과 같이 auroc 0.8점대의 높은 성능을 모여주었다.
    
- TabNet
    
    Tabular형 데이터에 특화되어 있는 딥러닝 모델. Hyper Parameter보다는 Feature Engineering의 여부에 따라 성능이 크게 좌우되었다. 학습 시간이나 성능은 Boosting 계열보다 낮게나오는 경향이 있었고, 다른 딥러닝 모델과는 다르게 사용한 Feature를 시각화 할 수 있었고 그를 이용해 Feature Selection을 할 수 있었음.
    
- LGBM, XGBoost
    
    Time Series에 대해 Lag 기법을 적용한 피처들을 추가하고, 문제 유형별, 항목별 클러스터링 진행 및 elo Rating에 대한 피처를 추가하여 성능을 개선했습니다.
    
- Seq / Transfomer 계열 (LSTM, LSTM-ATTN, BERT, LastQuery Transfomer)
    
    : Baseline에 주어진 LSTM, LSTM-ATTN, BERT 모델들에 더하여 과제로 주어진 LastQuery Transfomer 모델까지 구현하여, 각 유저가 푼 문제들을 이용하여 Sequence 형태로 Seq 계열 모델에 적용하여 학습시켰습니다. 사용한 피처로 가장 성능이 잘 나온 모델은 LSTM이었습니다. 피처엔지니어링보다 모델링 위주로 개선을 이어나가 높은 성능을 내지 못했습니다.
    
- lightGCN
    
    : 그래프 구조를 이용하여 특징을 추출한다는 점이 흥미로웠고, 이를 이용하여 유저와 아이템의 관계와 상호작용을 잘 추출할 수 있을 것으로 판단되어 lightGCN 모델을 사용했습니다. 하지만 단독으로 모델을 사용할 시 데이터 양이 충분하지 않다면 분산이 낮아 오히려 유저와 아이템의 관계와 상호작용을 잘 표현하지 못하는 현상을 발견했습니다.
    
- Graph + Transfomer (lightGCN + lqtransfomer)
    
    : lightGCN을 이용하여 뽑은 임베딩 벡터를 사용하여 LastQuery Transfomer에 input으로 사용한다면 더 높은 성능을 보일 것으로 예상되어 두 모델을 함께 사용했습니다. 실제로 lightGCN 모델을 단순 사용한 것의 성능(0.7371)보다 lightGCN 임베딩 결과를 LastQuery Transfomer 모델에 넣어 함께 사용한 것의 성능(0.7632)이 더 높았습니다. (0.7371→ 0.7632)

<br>

## C. 모델 성능

<img width="876" alt="image" src="https://user-images.githubusercontent.com/98138782/226339405-7b4dd5ec-251b-43db-a996-66f747721e72.png">

<br>

  
