# 🧠 Sophia AI (MARIA-Mammo)

**유방영상의학 전문 AI 어시스턴트**

RAG(Retrieval-Augmented Generation) 기반 논문 검색 & 질의응답 시스템

## ✨ 주요 기능

- 🔍 **BI-RADS 하이브리드 검색**: 구조화 매칭 + 시맨틱 검색
- 🌐 **다국어 쿼리 번역**: 한글 질문 → 영문 의학 키워드 자동 변환
- 📚 **17,870개 논문**: PubMed + BI-RADS 가이드라인
- 🤖 **LLM 답변 생성**: Ollama qwen2.5:14b 기반 RAG
- 💬 **채팅 인터페이스**: Streamlit 기반 직관적인 UI
- 📊 **자동 품질 평가**: 5-metric 답변 품질 측정

## 🚀 빠른 시작 (macOS)

### 방법 1: 원클릭 설치 (권장)

1. **`install.command`** 파일을 더블클릭
   - 모든 의존성 자동 설치
   - Ollama, Python, uv 자동 설정
   - LLM 모델(qwen2.5:14b) 다운로드

2. **`start-sophiaai.command`** 파일을 더블클릭
   - 자동으로 서버 시작 및 브라우저 오픈
   - http://localhost:8501 접속

### 방법 2: 수동 설치

#### 1. 필수 프로그램 설치

```bash
# Homebrew 설치 (macOS 패키지 매니저)
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Python 설치
brew install python@3.12

# uv 설치 (빠른 Python 패키지 매니저)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Ollama 설치 (https://ollama.com)
# 웹사이트에서 macOS용 다운로드
```

#### 2. LLM 모델 다운로드

```bash
# Ollama 서버 시작
ollama serve

# qwen2.5:14b 모델 다운로드 (약 8GB)
ollama pull qwen2.5:14b
```

#### 3. 프로젝트 설정

```bash
# 프로젝트 디렉토리 이동
cd maria-mammo

# Python 패키지 설치
uv sync

# 데이터베이스 인덱싱 (기존 DB가 있으면 생략)
uv run python scripts/index.py
```

#### 4. 실행

```bash
# Streamlit UI 실행
uv run streamlit run src/ui/app.py
```

## 📖 사용 방법

### 질문 예시

#### BI-RADS 카테고리
- "BI-RADS 5는 무엇인가요?"
- "BI-RADS Category 4A 설명해줘"

#### 기술적 질문
- "mammography의 기본 exposure procedure를 설명해줘"
- "유방촬영 노출 기법은?"
- "맘모그래피 포지셔닝 방법"

#### 비교 질문
- "DBT와 mammography 차이는?"
- "유방 밀도는 무엇인가요?"

#### 임상 가이드라인
- "맘모그래피 스크리닝은 몇 살부터 하나요?"

### 검색 키워드 자동 변환

한글 질문을 입력하면 자동으로 영문 의학 키워드로 변환됩니다:

```
입력: "유방촬영 노출 기법을 설명해줘"
   ↓ (자동 번역)
검색: "mammography exposure technique kVp mAs radiation dose"
   ↓
결과: 정확한 기술 논문 검색
```

## 🛠️ 고급 사용법

### CLI 검색

```bash
# 직접 검색
uv run python scripts/search.py

# 테스트 질문 실행
uv run python scripts/test_questions.py --category birads
```

### 답변 품질 평가

```bash
# 자동 품질 평가
python /tmp/auto_evaluate_quality.py
```

## Project Structure

```
maria-mammo/
├── src/
│   ├── collection/      # PubMed 데이터 수집
│   ├── processing/      # 데이터 정제 및 분류
│   ├── indexing/        # 임베딩 및 인덱싱
│   ├── search/          # 검색 엔진
│   ├── api/             # FastAPI 서버
│   └── ui/              # Streamlit UI
├── data/
│   ├── raw/             # 수집된 원본 데이터
│   ├── processed/       # 처리된 데이터
│   └── index/           # 검색 인덱스
├── scripts/             # 실행 스크립트
├── tests/               # 테스트
├── evaluation/          # 평가 데이터 및 스크립트
└── docs/                # 문서
```

## Configuration

`.env` 파일에서 설정:

| 변수 | 설명 | 기본값 |
|------|------|--------|
| `NCBI_API_KEY` | PubMed API 키 (선택) | - |
| `CROSSREF_EMAIL` | CrossRef API 이메일 | - |
| `EMBEDDING_MODEL` | 임베딩 모델 | S-PubMedBert-MS-MARCO |
| `RERANKER_MODEL` | 리랭커 모델 | ms-marco-MiniLM-L-6-v2 |

## API Reference

### POST /search

```json
{
  "query": "DBT screening microcalcification detection",
  "top_k": 10,
  "use_rerank": true
}
```

### GET /paper/{pmid}

논문 상세 정보 조회

### GET /stats

인덱스 통계 조회

## Development

```bash
# 개발 의존성 설치
uv sync --extra dev

# 테스트 실행
uv run pytest

# 린트
uv run ruff check src/
```

## License

MIT License
