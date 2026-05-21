# FastAPI AI 핵심 코드 발표 정리



## 1. FastAPI AI 서버 역할

![FastAPI AI 서버 역할](assets/fastapi-slide-08.png)

원본: `main.py`

```python
from router import product, ai, work, gallery, auction_rag

app = FastAPI(lifespan=lifespan)
app.include_router(product.router)   #  기본 AI 기능을 연결합니다 /  legacy product domain과의 하위 호환 라우팅입니다.
app.include_router(ai.router)        #  이미지 생성과 분석 엔드포인트를 묶었습니다 /  image generation, vision analysis, orchestration router입니다.
app.include_router(work.router)      #  작품 성과 예측을 연결했습니다 /  regression, classification inference endpoint입니다.
app.include_router(gallery.router)   #  예술관 추천을 연결했습니다 /  content-based similarity recommendation router입니다.
app.include_router(auction_rag.router) # 경매 분석을 연결했습니다 /  auction domain RAG pipeline router입니다.
```

 Spring Boot가 무거운 AI 작업을 직접 처리하지 않고, FastAPI가 전담하는 구조입니다.

## 2. 이미지 생성 및 분석

![이미지 생성 및 분석 API](assets/fastapi-slide-09.png)

원본: `service/ai_service.py`

```python
async def run_image_pipeline(self, request: ImagePipelineRequest) -> ImagePipelineResponse:
    return await run_in_threadpool(self._run_image_pipeline, request)

def _run_image_pipeline(self, request: ImagePipelineRequest) -> ImagePipelineResponse:
    image_path = self._generate_image_file(  #  이미지를 생성합니다 / OpenAI image generation model을 호출해서 local artifact를 만들었습니다.
        prompt=request.prompt,
        size=request.size
    )
    description, cache_hit = self._analyze_image_with_cache(str(image_path))  # 생성된 이미지를 분석합니다 /  vision inference 결과를 cache-aware하게 재사용했습니다.
    uploaded_image = self._upload_image_to_s3(image_path)  #  이미지를 S3에 저장합니다 /  object storage에 업로드하고 key와 presigned URL 메타데이터를 확보했습니다.

    return ImagePipelineResponse(
        image_path=str(image_path),
        description=description,
        image_key=uploaded_image["key"],
        image_url=uploaded_image["url"],
        file_type=uploaded_image["content_type"],
        file_size=uploaded_image["size"]
    )
```

 프롬프트를 받아 이미지를 만들고, 바로 분석한 뒤, S3 key와 presigned URL까지 반환합니다.

## 3. 작품 성과 예측

![작품 성과 예측 API](assets/fastapi-slide-11.png)

원본: `service/work_service.py`

```python
async def predict_views(self, features: WorkRegressionFeatures) -> WorkRegressionResponse:
    self.load_work_regressor()  #  예측 모델을 불러옵니다 / joblib/pkl serialized estimator를 lazy-load해서 재사용했습니다.

    feature_dict = features.model_dump()
    values = [feature_dict[name] for name in self.work_regressor_features]  # 입력값 순서를 맞춥니다 /  training schema order에 맞춰 feature vector를 정렬했습니다.
    prediction = self.work_regressor.predict([values])

    return WorkRegressionResponse(
        predicted_views=int(prediction[0]),
        created_datetime=datetime.now(),
        updated_datetime=datetime.now()
    )
```

 회귀 모델은 저장된 pkl과 feature 순서를 맞춰 예상 조회수를 계산하고, 첫 호출 이후에는 메모리 재사용으로 성능을 확보합니다.

## 4. 유사도 추천

![작품 및 갤러리 추천](assets/fastapi-slide-12.png)

원본: `service/work_recommend_service.py`

```python
work_texts = [
    f"{r['title']} {r['title']} {r['category']} {r['description']} {r['tags']}"
    for r in rows
]

tfidf_v = TfidfVectorizer(
    analyzer="char_wb",     # 한국어 유사도를 계산합니다 /  형태소 분해 대신 character n-gram 기반 vectorizer를 사용했습니다.
    ngram_range=(2, 4),
    max_features=6000,
)
tfidf_mat = tfidf_v.fit_transform(work_texts)
query_mat = tfidf_v.transform([request.content])
sim_scores = cosine_similarity(query_mat, tfidf_mat)[0]  #  추천 점수를 계산합니다 /  query-document similarity를 dense score로 산출했습니다.
```

 제목, 설명, 태그를 하나의 텍스트로 합치고, Kiwi 기반 전처리와 TF-IDF 유사도 계산으로 추천 점수를 만듭니다.

## 5. 경매 RAG 분석

![경매 RAG 분석](assets/fastapi-slide-13.png)

원본: `service/auction_rag_service.py`

```python
async def analyze(self, request: AuctionRagAnalyzeRequest) -> AuctionRagAnalyzeResponse:
    if request.fast_mode:
        return await self._analyze_fast(request)  #  빠른 분석 경로를 사용합니다 /  retrieval step을 생략한 low-latency path입니다.

    rag = await self.get_rag()                  #  RAG 분석기를 준비합니다 /  RAG runtime을 lazy-init하고 connection pool처럼 재사용했습니다.
    image_data = self._resolve_image_base64(request)
    image_analysis = await rag.vision_model_func(
        self._build_image_prompt(request),
        image_data=image_data,
    )

    rag_query = self._build_rag_query(request, image_analysis)
    auction_report = await rag.aquery(rag_query, mode="hybrid")  # 문서 기반 정밀 분석을 수행합니다 /  dense/sparse retrieval과 generation을 결합한 hybrid RAG를 사용했습니다.

    return AuctionRagAnalyzeResponse(
        image_analysis=str(image_analysis),
        auction_report=str(auction_report),
        used_rag=True,
        indexed_document_hint=str(DEFAULT_REPORT_PATH),
    )
```

 빠른 모드와 RAG 모드를 분리해서, 즉시 응답과 문서 기반 정밀 분석을 모두 지원합니다.

## 발표 흐름 요약

1. FastAPI는 Spring이 호출하는 AI 전용 백엔드입니다.
2. 이미지 생성은 생성, 분석, S3 저장을 한 번에 묶어 처리합니다.
3. 작품 예측은 저장된 pkl 모델과 feature 순서를 그대로 사용합니다.
4. 추천은 텍스트 유사도 기반으로 후보를 정렬합니다.
5. 경매 RAG는 빠른 분석과 문서 기반 정밀 분석을 분리합니다.
