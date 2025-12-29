# MARIA-Mammo

**M**ammography **A**dvanced **R**etrieval and **I**ntelligent **A**nalysis

PubMed 기반 유방영상의학 논문 검색 엔진

## Overview

MARIA-Mammo는 맘모그래피 관련 의학 논문을 효율적으로 검색하고 분석하기 위한 특화된 문헌 검색 시스템입니다.

### 주요 기능

- **PubMed 논문 수집**: 2005-2025년 맘모그래피 관련 논문 자동 수집
- **자동 분류**: Modality(DBT, FFDM, CEM), Pathology, Study Type 자동 분류
- **하이브리드 검색**: BM25 + Vector 검색 + Cross-encoder 리랭킹
- **웹 UI**: Streamlit 기반 직관적인 검색 인터페이스
- **REST API**: FastAPI 기반 API 서버

## Installation

### 요구사항

- Python 3.11+
- UV 패키지 매니저

### 설치

```bash
# UV 설치 (없는 경우)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 프로젝트 클론 및 의존성 설치
cd maria-mammo
uv sync

# 환경 변수 설정
cp .env.example .env
# .env 파일 편집하여 API 키 설정
```

## Quick Start

### 1. 데이터 수집

```bash
uv run python scripts/collect.py --start-year 2020 --end-year 2025
```

### 2. 데이터 처리

```bash
uv run python scripts/process.py
```

### 3. 인덱싱

```bash
uv run python scripts/index.py --full
```

### 4. 검색 실행

```bash
# CLI 검색
uv run python scripts/search.py

# API 서버
uv run uvicorn src.api.server:app --reload

# Web UI
uv run streamlit run src/ui/app.py
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
