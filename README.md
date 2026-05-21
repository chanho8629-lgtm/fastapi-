# FastAPI AI 핵심 코드 발표 정리 (BIDEO 서비스 맞춤)

## 발표 순서

| 순서 | 섹션 | 핵심 내용 |
|---|---|---|
| 1 | 기획 배경 | 시장성, 이용행태, 창작자 문제 정의 |
| 2 | 데이터셋/피처 설계 | 학습 데이터 구성, 분포/상관 분석 |
| 3 | 분류 모델 | 저조회수/고조회수 분류, 임계값 조정 |
| 4 | 회귀 모델 | 조회수 예측, 오차/중요 피처 분석 |
| 5 | 유사도 추천 | Kiwi + TF-IDF 기반 콘텐츠 매칭 |
| 6 | LLM/RAG 분석 | Fast 모드 vs Hybrid RAG 모드 |
| 7 | 모델 비교/운영 | 기능별 모델 선택 기준과 운영 지표 |

## 발표용 한눈에 요약

### 어떤 피처를 사용했는가

- 반응 지표: `likes`, `comments`, `shares`, `engagement_score`
- 비율 지표: `like_ratio`, `comment_ratio`, `share_ratio`, `watch_completion_rate`
- 품질 지표: `ai_quality_score`, `quality_completion_score`
- 시간/신선도 지표: `age_days`, `upload_month`, `upload_dayofweek`, `upload_year`
- 경매 지표: `starting_bid`, `bid_count`, `bidder_count`, `final_bid_price`
- 안정화 피처: `log_likes`, `log_comments`, `log_shares`, `log_engagement_score`

### 분석 결과 핵심

- EDA에서 조회수는 `likes/comments/shares`와 강한 양의 상관을 보였습니다.
- 회귀 모델은 실측 추세를 전반적으로 따라가며, 큰 오차는 일부 이상치 구간에서 발생했습니다.
- 피처 중요도는 `like_ratio`, `likes`, `log_likes` 계열이 상위로 나타났습니다.
- 분류 모델은 threshold 조정으로 precision/recall 균형을 목적에 맞게 제어할 수 있었습니다.
- 추천 모델은 Kiwi + TF-IDF 기반으로 콜드스타트에서도 동작 가능했습니다.
- LLM/RAG는 `fast_mode(속도)`와 `hybrid RAG(정밀도)`를 분리해 운영 목적에 맞춰 선택 가능합니다.

### 서비스 적용 결론

- 단기: 조회수 예측 + 분류로 노출/추천 우선순위를 자동화
- 중기: 추천 CTR, 분류 F1, 회귀 RMSE를 함께 모니터링해 모델 개선 루프 운영
- 장기: 경매 RAG 리포트를 의사결정 보조 기능으로 고도화

## 1. 기획 배경: 데이터로 본 문제

### 분석 질문

| 질문 | 확인하려는 내용 |
|---|---|
| 숏폼/모바일 영상 소비 비중은 충분히 큰가? | BIDEO 핵심 사용자층 규모 확인 |
| 사용자들이 영상 콘텐츠에 비용을 지불하는 흐름이 있는가? | 경매/결제 기능 확장 가능성 확인 |
| 창작자 수익 구조가 상위 집중인지, 개선 여지가 있는가? | 수익 다변화 기능 필요성 확인 |

### 사용 데이터

| 데이터 | 출처 | 사용 목적 |
|---|---|---|
| 국내 콘텐츠산업 매출 추이(2020~2024) | 콘텐츠산업 통계/정리 데이터 | 시장 성장성 확인 |
| OTT 이용률 및 유료 이용 비율 추이 | 미디어 이용 통계/정리 데이터 | 결제 기반 서비스 수용도 확인 |
| 모바일·숏폼 콘텐츠 이용률 | 디지털 미디어 이용 통계/정리 데이터 | 피드형 UX/추천 기능 근거 확보 |
| 1인 창작자 수익 분포(상위 집중도) | 창작자 경제 통계/정리 데이터 | 창작자 수익화 문제 검증 |

### 분석 과정 1: 시장성/수익화 가능성 비교

![콘텐츠 시장 성장](images/data-bideo-market-growth.png)
콘텐츠 시장이 꾸준히 성장해 서비스 진입 근거가 됩니다.

![OTT 이용률/유료 이용](images/data-bideo-usage-rate.png)
OTT 이용과 유료 전환이 증가해 결제 기능 수요를 뒷받침합니다.

![모바일/숏폼 소비 구조](images/data-bideo-mobile-content-core.png)
모바일·숏폼 중심 소비가 피드형 UX 전략과 맞습니다.

![창작자 수익 집중 문제](images/data-bideo-creator-problem-summary.png)
수익 집중 구조를 완화할 수익 다변화 기능이 필요합니다.

## 2. 데이터셋과 피처 설계

![회귀 EDA: 산점도](images/regression_eda_scatter.png)
반응 지표가 커질수록 조회수도 함께 커지는 경향이 확인됩니다.

![회귀 EDA: 상관행렬](images/regression_eda_corr.png)
조회수는 likes/comments/shares와 높은 양의 상관을 보입니다.

![회귀 EDA: 분포](images/regression_eda_dist.png)
주요 변수 분포가 치우쳐 있어 로그/비율 피처가 필요합니다.

```python
feature_dict = features.model_dump()
values = [feature_dict[name] for name in self.work_regressor_features]
prediction = self.work_regressor.predict([values])
```

- 회귀/분류 공통으로 참여/품질/경매 피처를 함께 사용합니다.
- 로그 변환과 비율 피처를 적용해 heavy-tail 분포를 안정화했습니다.

## 3. 분류 모델 (고조회수 여부)

![분류 데이터 정제 후 분포](images/classification_dist_cleaned.png)
이상치 제거 후 분포가 안정화되어 분류 학습 품질이 개선됩니다.

![분류 클래스 평균 비교](images/classification_class_mean.png)
고조회수 클래스가 반응 지표 평균에서 뚜렷한 차이를 보입니다.

![분류 임계값별 P/R/F1](images/classification_prf_threshold.png)
threshold 조정으로 precision-recall 균형을 목적에 맞게 선택할 수 있습니다.

![분류 모델 비교](images/classification_model_compare.png)
튜닝+임계값 조정 모델이 전체 균형 지표에서 안정적입니다.

```python
prob = self.work_classifier.predict_proba([values])[0][1]
label = int(prob >= self.classification_threshold)
```

## 4. 회귀 모델 (조회수 예측)

![실측 vs 예측](images/regression_actual_vs_pred.png)
모델 예측이 실측 추세를 전반적으로 따라가며 실사용 가능 수준을 보입니다.

![잔차 분포](images/regression_residuals.png)
대부분 오차가 0 근처에 모여 평균적인 예측 편향은 크지 않습니다.

![피처 중요도](images/regression_feature_importance.png)
like_ratio와 likes 계열이 조회수 설명력에 크게 기여합니다.

```python
self.load_work_regressor()
prediction = self.work_regressor.predict([values])
return {"predicted_views": int(prediction[0])}
```

## 5. 유사도 추천 모델

![작품/갤러리 유사도 추천](images/fastapi-slide-12.png)
텍스트 유사도 기반으로 사용자 취향과 가까운 콘텐츠를 추천합니다.

```python
tfidf_v = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=6000)
tfidf_mat = tfidf_v.fit_transform(work_texts)
query_mat = tfidf_v.transform([request.content])
sim_scores = cosine_similarity(query_mat, tfidf_mat)[0]
```

## 6. LLM/RAG 분석 모델

![경매 LLM/RAG 분석](images/fastapi-slide-13.png)
fast 모드와 hybrid RAG를 분리해 속도/정밀도 요구를 동시에 대응합니다.

```python
if request.fast_mode:
    return await self._analyze_fast(request)

auction_report = await rag.aquery(rag_query, mode="hybrid")
```

## 7. 모델 비교 및 운영 적용

| 기능 | 적용 모델 | 선택 이유 | 핵심 운영 지표 |
|---|---|---|---|
| 조회수 예측 | LGBM Regressor | 비선형 관계/스케일 차이를 안정적으로 학습 | MAE, RMSE, 잔차 왜도 |
| 고조회수 분류 | RF 계열 + threshold 조정 | precision/recall 트레이드오프 제어 용이 | Precision, Recall, F1 |
| 콘텐츠 추천 | Kiwi + TF-IDF + Cosine | 실시간성/해석성/콜드스타트 대응 | CTR, 저장률, 재방문율 |
| 경매 분석 | Vision + LLM + Hybrid RAG | 근거 기반 설명 가능 | 응답시간, 근거 일치율, 사용자 채택률 |

### 운영 대시보드 포인트

- 예측 결과는 일별 추이로 집계해 모니터링
- 분류 결과는 임계값별 정밀도/재현율로 관리
- 추천 결과는 CTR/전환율 중심으로 관리
- RAG 결과는 응답시간/피드백 점수로 관리
