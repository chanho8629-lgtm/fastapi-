# FastAPI AI 발표 요약 (BIDEO)

## 발표 흐름

1. 기획 배경: 왜 이 문제를 풀어야 하는가
2. 데이터셋/피처: 어떤 데이터로 학습했는가
3. 분류/회귀: 어떤 모델이 어떤 결정을 하는가
4. 추천/LLM-RAG: 사용자 경험을 어떻게 높였는가
5. 운영 지표: 서비스에 어떻게 적용하고 개선하는가

## 1) 기획 배경: 데이터로 본 문제

### 분석 질문

| 질문 | 확인 포인트 |
|---|---|
| 숏폼/모바일 영상 소비 비중은 충분히 큰가? | 타깃 사용자 규모 확인 |
| 사용자는 영상 콘텐츠에 비용을 지불하는가? | 경매/결제 기능 확장 가능성 확인 |
| 창작자 수익은 상위 집중 구조인가? | 수익 다변화 기능 필요성 확인 |

### 사용 데이터

| 데이터 | 목적 |
|---|---|
| 콘텐츠 산업 매출 추이 | 시장 성장성 검증 |
| OTT 이용률/유료 이용 비율 | 결제 수용도 검증 |
| 모바일·숏폼 이용률 | 피드형 UX 필요성 검증 |
| 창작자 수익 분포 | 수익화 문제 검증 |

![콘텐츠 시장 성장](images/data-bideo-market-growth.png)
콘텐츠 시장은 성장 중이며, 서비스 진입 타당성이 있습니다.

![OTT 이용률/유료 이용](images/data-bideo-usage-rate.png)
OTT 이용과 유료 전환 증가로 결제 기능 수요가 확인됩니다.

![모바일/숏폼 소비 구조](images/data-bideo-mobile-content-core.png)
모바일·숏폼 중심 소비는 피드형 탐색 UX와 일치합니다.

![창작자 수익 집중 문제](images/data-bideo-creator-problem-summary.png)
창작자 수익 집중 완화를 위한 추가 수익화 수단이 필요합니다.

## 2) 데이터셋/피처 설계

### 사용 피처(핵심)

- 반응: `likes`, `comments`, `shares`, `engagement_score`
- 비율: `like_ratio`, `comment_ratio`, `share_ratio`, `watch_completion_rate`
- 품질: `ai_quality_score`, `quality_completion_score`
- 시간: `age_days`, `upload_month`, `upload_dayofweek`
- 경매: `starting_bid`, `bid_count`, `bidder_count`, `final_bid_price`
- 안정화: `log_likes`, `log_comments`, `log_shares`

![회귀 EDA 상관행렬](images/regression_eda_corr.png)
조회수는 반응 지표와 높은 양의 상관을 보입니다.

![회귀 EDA 분포](images/regression_eda_dist.png)
분포 치우침이 커 로그/비율 피처가 유효합니다.

## 3) 모델 결과

### 분류 모델 (고조회수 여부)

![분류 임계값별 P/R/F1](images/classification_prf_threshold.png)
threshold 조정으로 precision/recall을 운영 목적에 맞게 설정할 수 있습니다.

![분류 모델 비교](images/classification_model_compare.png)
튜닝 + 임계값 조정 모델이 가장 안정적인 균형을 보였습니다.

### 회귀 모델 (조회수 예측)

![실측 vs 예측](images/regression_actual_vs_pred.png)
예측값이 실측 추세를 전반적으로 따라가 실사용 가능성이 확인됩니다.

![피처 중요도](images/regression_feature_importance.png)
`like_ratio`, `likes` 계열이 예측에 가장 큰 기여를 보였습니다.

## 4) 추천 / LLM-RAG 적용

![유사도 추천](images/fastapi-slide-12.png)
Kiwi + TF-IDF 기반 추천으로 콜드스타트 상황에서도 추천이 가능합니다.

![경매 LLM/RAG 분석](images/fastapi-slide-13.png)
`fast`(속도)와 `hybrid RAG`(정밀도)를 분리해 상황별 대응이 가능합니다.

## 5) FastAPI 기능 전체 구성

### AI API 구성

- 이미지 생성/분석: 프롬프트 기반 이미지 생성, 결과 분석, 저장소 업로드
- 조회수 회귀 예측: 작품 메타/반응 피처로 예상 조회수 추정
- 고조회수 분류: 확률 + threshold 기준으로 노출 우선순위 분류
- 유사도 추천: Kiwi + TF-IDF + Cosine 기반 작품/갤러리 추천
- 경매 분석: 비전 분석 + LLM 리포트 생성
- 문서 기반 분석: `RAGAnything` + `Hybrid RAG` 조합으로 근거 문서 기반 분석

### RAGAnything 적용 포인트

- 경매/시장 리포트 PDF를 인덱싱해 질의 시 근거 문맥을 먼저 검색
- 검색 결과를 LLM 생성 단계에 결합해 설명 가능한 분석 결과 제공
- `fast mode`는 검색 단계 단축, `hybrid`는 정확도 우선 전략으로 운용

### FastAPI 운영 관점 요약

- API를 기능별로 분리해 장애 격리와 확장성을 확보
- 모델별 지표를 분리 모니터링해 성능 저하 지점을 빠르게 파악
- 추천/예측/분석 결과를 대시보드에 연결해 의사결정 루프를 자동화

## 6) 운영 적용 포인트

| 기능 | 모델 | 운영 지표 |
|---|---|---|
| 조회수 예측 | LGBM Regressor | MAE, RMSE |
| 고조회수 분류 | RF + threshold | Precision, Recall, F1 |
| 추천 | Kiwi + TF-IDF + Cosine | CTR, 저장률 |
| 경매 분석 | Vision + LLM + Hybrid RAG | 응답시간, 채택률 |

## 발표 결론

- 데이터로 문제를 검증하고
- 피처/모델을 기능별로 분리 적용했으며
- 운영 지표 기반으로 개선 루프를 돌릴 수 있는 구조를 만들었습니다.
