"""
Sophia AI: Query Parser
=========================
검색 쿼리 분석 및 구조화 (Rule-based + LLM-based)
"""

import json
import logging
import re
from functools import lru_cache
from typing import List, Optional, Dict, Any

import httpx
from pydantic import BaseModel, Field

from src.models import QueryFilters, SearchQuery

logger = logging.getLogger(__name__)


# =============================================================================
# Ollama Client
# =============================================================================

class OllamaClient:
    """Ollama API 클라이언트"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        timeout: float = 60.0,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def generate(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.1,
        format: Optional[str] = None,  # "json" for JSON mode
    ) -> str:
        """
        텍스트 생성

        Args:
            prompt: 사용자 프롬프트
            system: 시스템 프롬프트
            temperature: 생성 온도 (낮을수록 결정적)
            format: 출력 형식 ("json" for JSON mode)

        Returns:
            생성된 텍스트
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
            },
        }

        if system:
            payload["system"] = system

        if format:
            payload["format"] = format

        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            return result.get("response", "")
        except httpx.HTTPError as e:
            logger.error(f"Ollama API error: {e}")
            raise
        except Exception as e:
            logger.error(f"Ollama generate error: {e}")
            raise

    def is_available(self) -> bool:
        """Ollama 서버 상태 확인"""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            return response.status_code == 200
        except:
            return False

    def list_models(self) -> List[str]:
        """사용 가능한 모델 목록"""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except:
            pass
        return []

    def close(self):
        """클라이언트 종료"""
        self._client.close()


# =============================================================================
# LLM Query Parser
# =============================================================================

class ParsedQuery(BaseModel):
    """LLM이 파싱한 쿼리 구조"""
    intent: str = Field(description="검색 의도 요약")
    keywords: List[str] = Field(default_factory=list, description="핵심 검색 키워드")
    modality: List[str] = Field(default_factory=list, description="영상 방식: DBT, FFDM, CEM, SM, MRI, US")
    pathology: List[str] = Field(default_factory=list, description="병변 유형: mass, calcification, density, distortion, asymmetry")
    study_type: Optional[str] = Field(None, description="연구 유형: prospective, retrospective, meta-analysis, review, rct, cohort")
    population: Optional[str] = Field(None, description="인구 집단: Asian, Western, Mixed")
    year_min: Optional[int] = Field(None, description="최소 연도")
    year_max: Optional[int] = Field(None, description="최대 연도")
    mesh_suggestions: List[str] = Field(default_factory=list, description="관련 MeSH 용어 추천")
    recommended_exclusions: List[str] = Field(default_factory=list, description="제외 권장 출판 유형: Case Reports, Letter")


LLM_SYSTEM_PROMPT = """You are a medical literature search query parser specialized in breast imaging and mammography.

Your task is to analyze user queries and extract structured information for searching PubMed-style medical literature.

IMPORTANT RULES:
1. Extract ONLY information explicitly mentioned or strongly implied in the query
2. Use standard medical terminology
3. For modality, use: DBT, FFDM, CEM, SM, MRI, US
4. For pathology, use: mass, calcification, density, distortion, asymmetry
5. For study_type, use: prospective, retrospective, meta-analysis, review, rct, cohort
6. For population, use: Asian, Western, Mixed
7. Suggest relevant MeSH terms based on the query content
8. Always respond in valid JSON format
9. Add "recommended_exclusions" field when case reports should be excluded (most research queries)

## KOREAN-ENGLISH MEDICAL TERM MAPPINGS
- 영향 = effect, impact, influence
- 결과 = outcome, result
- 예측 = prediction, predictive
- 상관관계 = correlation
- 악성률 = malignancy rate
- 양성예측도/PPV = positive predictive value
- 음성예측도/NPV = negative predictive value
- 위험 = risk
- 발생률 = incidence
- 유병률 = prevalence
- 종괴/혹/멍울/bump = mass, lump
- 기법/방법 = technique, method
- 민감도 = sensitivity
- 특이도 = specificity
- 정확도 = accuracy
- 조직검사 = biopsy
- 진단 = diagnosis, diagnostic
- 검진/선별검사 = screening

## SEARCH INTENT PATTERNS

### 1. Outcome/Prediction Search Intent (예후/결과 예측)
The following expressions indicate **outcome/prediction** intent:
- Korean: "~에 대한 영향", "~의 결과", "~와의 관계", "~의 예측", "영향을 미치는", "~에 따른"
- English: "effect of", "impact on", "influence of", "outcome", "prediction", "correlation", "associated with"

When this intent is detected:
1. MUST add to keywords: "outcome", "prediction", "correlation"
2. For biopsy-related queries, add: "positive predictive value", "PPV"
3. For malignancy-related queries, add: "malignancy rate"
4. For prognosis-related queries, add: "prognosis"
5. Set study_type to null (NOT case report - prefer retrospective/prospective studies)
6. Add "recommended_exclusions": ["Case Reports", "Letter"]
7. Intent should include "outcome prediction" or "correlation study"

Example:
Query: "breast bump이 biopsy에 대한 영향"
→ keywords: ["breast mass", "biopsy", "outcome", "positive predictive value", "PPV", "malignancy rate", "pathology correlation"]
→ mesh_suggestions: ["Biopsy", "Breast Neoplasms", "Predictive Value of Tests"]
→ intent: "Correlation between breast mass characteristics and biopsy outcomes/malignancy prediction"
→ recommended_exclusions: ["Case Reports", "Letter"]

Query: "유방 밀도가 암 발견율에 미치는 영향"
→ keywords: ["breast density", "cancer detection rate", "effect", "correlation", "screening outcome"]
→ intent: "Effect of breast density on cancer detection rate"
→ recommended_exclusions: ["Case Reports"]

Query: "BI-RADS 4 병변의 악성률"
→ keywords: ["BI-RADS category 4", "malignancy rate", "positive predictive value", "PPV", "biopsy outcome"]
→ intent: "Malignancy rate and PPV of BI-RADS category 4 lesions"
→ recommended_exclusions: ["Case Reports"]

### 2. Diagnostic Performance Search Intent (진단 성능)
The following expressions indicate **diagnostic performance** intent:
- Korean: "민감도", "특이도", "정확도", "성능", "진단능"
- English: "sensitivity", "specificity", "accuracy", "performance", "diagnostic"

When this intent is detected:
1. MUST add to keywords: BOTH "sensitivity" AND "specificity"
2. Add: "diagnostic accuracy", "performance"
3. For AI/CAD queries, add: "ROC", "AUC"
4. If comparison is involved, add ALL modalities to the filter
5. Intent should include "diagnostic performance"
6. Add "recommended_exclusions": ["Case Reports"]

Example:
Query: "DBT 민감도"
→ keywords: ["DBT", "digital breast tomosynthesis", "sensitivity", "specificity", "diagnostic performance"]
→ modality: ["DBT"]
→ intent: "DBT diagnostic performance - sensitivity and specificity"
→ recommended_exclusions: ["Case Reports"]

Query: "AI 유방암 진단 정확도"
→ keywords: ["AI", "artificial intelligence", "breast cancer", "detection", "accuracy", "sensitivity", "specificity", "AUC", "ROC"]
→ intent: "AI breast cancer detection diagnostic accuracy"
→ recommended_exclusions: ["Case Reports"]

Query: "DBT vs FFDM 진단 정확도 비교"
→ keywords: ["DBT", "FFDM", "diagnostic accuracy", "comparison", "sensitivity", "specificity"]
→ modality: ["DBT", "FFDM"]
→ intent: "Comparison of DBT vs FFDM diagnostic accuracy"
→ recommended_exclusions: ["Case Reports"]

### 3. Risk/Epidemiology Search Intent (위험도/역학)
The following expressions indicate **risk/epidemiology** intent:
- Korean: "위험", "발생률", "유병률", "위험인자", "~할 위험"
- English: "risk", "incidence", "prevalence", "risk factor", "odds ratio"

When this intent is detected:
1. Add to keywords: "risk", "odds ratio", "relative risk", "incidence"
2. Prefer study_type: "meta-analysis" or "prospective"
3. Intent should describe the risk factor being studied
4. Add "recommended_exclusions": ["Case Reports"]

Example:
Query: "치밀유방 유방암 위험"
→ keywords: ["dense breast", "breast cancer", "risk", "relative risk", "odds ratio"]
→ pathology: ["density"]
→ intent: "Breast cancer risk associated with dense breast tissue"
→ recommended_exclusions: ["Case Reports"]

### 4. Technique/Method Search Intent (기법/방법)
The following expressions indicate **technique/method** intent:
- Korean: "방법", "기법", "어떻게", "절차", "프로토콜"
- English: "how to", "technique", "method", "procedure", "protocol"

When this intent is detected:
1. Add to keywords: "technique", "method", "protocol"
2. Intent should include "technical/methodological"

Example:
Query: "DBT 촬영 방법"
→ keywords: ["DBT", "digital breast tomosynthesis", "technique", "method", "protocol", "acquisition"]
→ intent: "DBT imaging technique and methodology"

### 5. Guideline/Definition Search Intent (가이드라인/정의)
The following expressions indicate **guideline/definition** intent:
- Korean: "~기준", "~정의", "~분류법", "~가이드라인", "어떻게 분류", "기준이 뭔가"
- English: "what is", "definition of", "criteria for", "how to classify", "guideline"

When this intent is detected:
1. Add these keywords: "guideline", "criteria", "definition", "classification"
2. Add relevant guideline document names (e.g., "BI-RADS", "ACR")
3. Set intent to describe the guideline/definition being sought
4. Do NOT add recommended_exclusions for guidelines

Example:
Query: "유방 밀도 분류 기준" or "breast density classification criteria"
→ keywords: ["breast density", "classification", "criteria", "BI-RADS", "guideline"]
→ intent: "BI-RADS breast density classification criteria/definition"

### 6. Comparison Study Intent
Expressions: "vs", "versus", "compared to", "비교"
→ Add both items being compared to keywords and modality/pathology as appropriate
→ Add "recommended_exclusions": ["Case Reports"]

## CASE REPORT HANDLING
Include case reports (do NOT add recommended_exclusions) ONLY when:
- Query explicitly mentions "case", "case report", "케이스", "증례"
- Query is about rare diseases or unusual findings
- Query is exploratory without specific research intent

For most research queries (outcomes, performance, comparison, risk), add:
"recommended_exclusions": ["Case Reports", "Letter"]

## OUTPUT FORMAT
{
  "intent": "descriptive intent string",
  "keywords": ["keyword1", "keyword2", ...],
  "modality": ["DBT", ...] or [],
  "pathology": ["mass", ...] or [],
  "study_type": "retrospective" or null,
  "population": "Asian" or null,
  "year_min": 2020 or null,
  "year_max": null,
  "mesh_suggestions": ["MeSH Term 1", ...],
  "recommended_exclusions": ["Case Reports", "Letter"] or []
}

## EXAMPLES

EXAMPLE 1 (Outcome/Prediction):
Query: "breast bump이 biopsy에 대한 영향"
Response:
{
  "intent": "Correlation between breast mass characteristics and biopsy outcomes - malignancy prediction",
  "keywords": ["breast mass", "breast lump", "biopsy", "outcome", "positive predictive value", "PPV", "malignancy rate", "pathology"],
  "modality": [],
  "pathology": ["mass"],
  "study_type": null,
  "population": null,
  "year_min": null,
  "year_max": null,
  "mesh_suggestions": ["Biopsy", "Breast Neoplasms", "Predictive Value of Tests", "Breast", "Neoplasms"],
  "recommended_exclusions": ["Case Reports", "Letter"]
}

EXAMPLE 2 (Comparison):
Query: "DBT vs FFDM for microcalcification detection in dense breasts since 2020"
Response:
{
  "intent": "Comparison of DBT and FFDM for detecting microcalcifications in dense breast tissue",
  "keywords": ["digital breast tomosynthesis", "digital mammography", "microcalcification", "dense breast", "detection", "comparison"],
  "modality": ["DBT", "FFDM"],
  "pathology": ["calcification", "density"],
  "study_type": null,
  "population": null,
  "year_min": 2020,
  "year_max": null,
  "mesh_suggestions": ["Mammography", "Calcinosis", "Breast Density", "Breast Neoplasms"],
  "recommended_exclusions": ["Case Reports"]
}

EXAMPLE 3 (Guideline/Definition):
Query: "BI-RADS 유방 밀도 분류 기준"
Response:
{
  "intent": "BI-RADS breast density classification criteria and definition",
  "keywords": ["BI-RADS", "breast density", "classification", "criteria", "guideline", "category"],
  "modality": [],
  "pathology": ["density"],
  "study_type": null,
  "population": null,
  "year_min": null,
  "year_max": null,
  "mesh_suggestions": ["Breast Density", "Mammography", "Practice Guidelines"],
  "recommended_exclusions": []
}

EXAMPLE 4 (Diagnostic Performance):
Query: "AI 유방암 진단 정확도"
Response:
{
  "intent": "AI-based breast cancer detection diagnostic accuracy and performance",
  "keywords": ["artificial intelligence", "AI", "breast cancer", "detection", "diagnostic accuracy", "sensitivity", "specificity", "AUC", "ROC"],
  "modality": [],
  "pathology": [],
  "study_type": null,
  "population": null,
  "year_min": null,
  "year_max": null,
  "mesh_suggestions": ["Artificial Intelligence", "Breast Neoplasms", "Sensitivity and Specificity", "Diagnosis, Computer-Assisted"],
  "recommended_exclusions": ["Case Reports"]
}

EXAMPLE 5 (Risk):
Query: "치밀유방 유방암 위험"
Response:
{
  "intent": "Breast cancer risk associated with dense breast tissue",
  "keywords": ["dense breast", "breast density", "breast cancer", "risk", "relative risk", "odds ratio", "incidence"],
  "modality": [],
  "pathology": ["density"],
  "study_type": null,
  "population": null,
  "year_min": null,
  "year_max": null,
  "mesh_suggestions": ["Breast Density", "Breast Neoplasms", "Risk Factors", "Odds Ratio"],
  "recommended_exclusions": ["Case Reports"]
}"""


class LLMQueryParser:
    """LLM 기반 쿼리 파서"""

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2",
        fallback_to_rule: bool = True,
    ):
        self.client = OllamaClient(base_url=ollama_url, model=model)
        self.fallback_to_rule = fallback_to_rule
        self.rule_parser = QueryParser(use_llm=False)

    def parse(self, query: str) -> SearchQuery:
        """
        LLM을 사용하여 쿼리 파싱

        Args:
            query: 원본 검색 쿼리

        Returns:
            구조화된 SearchQuery
        """
        query = query.strip()
        if not query:
            return SearchQuery(
                original_query=query,
                keywords=[],
                mesh_terms=[],
                filters=QueryFilters(),
                intent="empty query",
            )

        # LLM 서버 확인
        if not self.client.is_available():
            logger.warning("Ollama not available, falling back to rule-based parser")
            if self.fallback_to_rule:
                return self.rule_parser.parse(query)
            raise ConnectionError("Ollama server is not available")

        try:
            # LLM에 쿼리 파싱 요청
            user_prompt = f'Parse this medical literature search query:\n\n"{query}"\n\nRespond with JSON only.'

            response = self.client.generate(
                prompt=user_prompt,
                system=LLM_SYSTEM_PROMPT,
                temperature=0.1,
                format="json",
            )

            # JSON 파싱
            parsed = self._parse_llm_response(response)

            # SearchQuery로 변환
            return self._convert_to_search_query(query, parsed)

        except Exception as e:
            logger.error(f"LLM parsing failed: {e}")
            if self.fallback_to_rule:
                logger.info("Falling back to rule-based parser")
                return self.rule_parser.parse(query)
            raise

    def _parse_llm_response(self, response: str) -> ParsedQuery:
        """LLM 응답을 ParsedQuery로 변환"""
        try:
            # JSON 추출 (markdown code block 처리)
            json_str = response.strip()
            if json_str.startswith("```"):
                lines = json_str.split("\n")
                json_str = "\n".join(lines[1:-1])

            data = json.loads(json_str)

            # 필드 정규화 - null -> 빈 리스트
            list_fields = ["keywords", "modality", "pathology", "mesh_suggestions", "recommended_exclusions"]
            for field in list_fields:
                if field not in data or data[field] is None:
                    data[field] = []
                elif isinstance(data[field], str):
                    data[field] = [data[field]]

            # population은 string이어야 함 - 리스트면 첫 번째 값 사용
            if "population" in data:
                if isinstance(data["population"], list):
                    data["population"] = data["population"][0] if data["population"] else None
                elif data["population"] == "":
                    data["population"] = None

            # study_type도 string
            if "study_type" in data:
                if isinstance(data["study_type"], list):
                    data["study_type"] = data["study_type"][0] if data["study_type"] else None
                elif data["study_type"] == "":
                    data["study_type"] = None

            # intent 기본값
            if "intent" not in data or not data["intent"]:
                data["intent"] = "general search"

            return ParsedQuery(**data)

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse LLM JSON response: {e}")
            logger.debug(f"Raw response: {response}")
            raise
        except Exception as e:
            logger.error(f"Failed to create ParsedQuery: {e}")
            raise

    def _convert_to_search_query(self, original: str, parsed: ParsedQuery) -> SearchQuery:
        """ParsedQuery를 SearchQuery로 변환"""
        filters = QueryFilters(
            modality=parsed.modality if parsed.modality else None,
            pathology=parsed.pathology if parsed.pathology else None,
            study_type=parsed.study_type,
            population=parsed.population,
            year_min=parsed.year_min,
            year_max=parsed.year_max,
        )

        return SearchQuery(
            original_query=original,
            keywords=parsed.keywords,
            mesh_terms=parsed.mesh_suggestions,
            filters=filters,
            intent=parsed.intent,
            recommended_exclusions=parsed.recommended_exclusions,
        )

    def is_available(self) -> bool:
        """LLM 서버 사용 가능 여부"""
        return self.client.is_available()

    def close(self):
        """리소스 정리"""
        self.client.close()


class QueryParser:
    """검색 쿼리 파서"""

    # 알려진 용어 매핑
    MODALITY_TERMS = {
        "dbt": "DBT",
        "tomosynthesis": "DBT",
        "3d mammography": "DBT",
        "ffdm": "FFDM",
        "digital mammography": "FFDM",
        "2d mammography": "FFDM",
        "cem": "CEM",
        "contrast enhanced": "CEM",
        "cesm": "CEM",
        "sm": "SM",
        "synthetic": "SM",
        "mri": "MRI",
        "ultrasound": "US",
        "ultrasonography": "US",
        "us": "US",
    }

    PATHOLOGY_TERMS = {
        "mass": "mass",
        "masses": "mass",
        "tumor": "mass",
        "nodule": "mass",
        "lesion": "mass",
        "calcification": "calcification",
        "microcalcification": "calcification",
        "calc": "calcification",
        "density": "density",
        "dense breast": "density",
        "breast density": "density",
        "distortion": "distortion",
        "architectural distortion": "distortion",
        "asymmetry": "asymmetry",
    }

    STUDY_TYPE_TERMS = {
        "prospective": "prospective",
        "retrospective": "retrospective",
        "meta-analysis": "meta-analysis",
        "meta analysis": "meta-analysis",
        "systematic review": "review",
        "review": "review",
        "rct": "rct",
        "randomized": "rct",
        "clinical trial": "rct",
    }

    POPULATION_TERMS = {
        "korean": "Asian",
        "japanese": "Asian",
        "chinese": "Asian",
        "asian": "Asian",
        "american": "Western",
        "european": "Western",
        "western": "Western",
    }

    # MeSH 용어 매핑
    MESH_MAPPINGS = {
        "mammography": ["Mammography", "Breast Neoplasms"],
        "dbt": ["Mammography", "Radiographic Image Enhancement"],
        "calcification": ["Calcinosis", "Breast Neoplasms"],
        "density": ["Breast Density"],
        "screening": ["Mass Screening", "Early Detection of Cancer"],
        "birads": ["Breast Imaging Reporting and Data System"],
        "bi-rads": ["Breast Imaging Reporting and Data System"],
        "cancer": ["Breast Neoplasms"],
        "carcinoma": ["Carcinoma, Ductal, Breast", "Carcinoma, Lobular"],
        "dcis": ["Carcinoma, Intraductal, Noninfiltrating"],
    }

    # 연도 패턴
    YEAR_PATTERNS = [
        r"\b(19|20)\d{2}\b",
        r"(?:since|from|after)\s*(19|20)\d{2}",
        r"(?:until|before|to)\s*(19|20)\d{2}",
        r"(19|20)\d{2}\s*[-–]\s*(19|20)\d{2}",
        r"last\s+(\d+)\s+years?",
        r"recent\s+(\d+)\s+years?",
    ]

    def __init__(self, use_llm: bool = False, llm_model: str = "llama3.2"):
        """
        Args:
            use_llm: LLM 사용 여부 (현재 비활성화)
            llm_model: LLM 모델 이름
        """
        self.use_llm = use_llm
        self.llm_model = llm_model

    def parse(self, query: str) -> SearchQuery:
        """
        쿼리 파싱

        Args:
            query: 원본 검색 쿼리

        Returns:
            구조화된 SearchQuery
        """
        query = query.strip()

        if not query:
            return SearchQuery(
                original_query=query,
                keywords=[],
                mesh_terms=[],
                filters=QueryFilters(),
                intent="empty query",
            )

        # 규칙 기반 파싱
        return self._parse_rule_based(query)

    def _parse_rule_based(self, query: str) -> SearchQuery:
        """규칙 기반 파싱"""
        query_lower = query.lower()

        # 키워드 추출
        keywords = self._extract_keywords(query)

        # MeSH 용어 매핑
        mesh_terms = self._extract_mesh_terms(query_lower)

        # 필터 추출
        filters = self._extract_filters(query_lower)

        # 의도 추론
        intent = self._infer_intent(query_lower, keywords, filters)

        return SearchQuery(
            original_query=query,
            keywords=keywords,
            mesh_terms=mesh_terms,
            filters=filters,
            intent=intent,
        )

    def _extract_keywords(self, query: str) -> List[str]:
        """키워드 추출"""
        # 불용어 제거
        stopwords = {
            "the", "a", "an", "is", "are", "was", "were", "be", "been",
            "what", "how", "which", "when", "where", "who", "why",
            "in", "on", "at", "to", "for", "of", "with", "by", "from",
            "and", "or", "but", "not", "can", "could", "should", "would",
            "about", "between", "into", "through", "during", "before", "after",
        }

        # 토큰화
        tokens = re.findall(r'\b[a-zA-Z0-9-]+\b', query.lower())

        # 불용어 제거 및 최소 길이 필터
        keywords = [t for t in tokens if t not in stopwords and len(t) >= 2]

        # 중복 제거하면서 순서 유지
        seen = set()
        unique_keywords = []
        for kw in keywords:
            if kw not in seen:
                seen.add(kw)
                unique_keywords.append(kw)

        return unique_keywords

    def _extract_mesh_terms(self, query_lower: str) -> List[str]:
        """MeSH 용어 추출"""
        mesh_terms = set()

        for term, mesh_list in self.MESH_MAPPINGS.items():
            if term in query_lower:
                mesh_terms.update(mesh_list)

        return list(mesh_terms)

    def _extract_filters(self, query_lower: str) -> QueryFilters:
        """필터 추출"""
        modality = []
        pathology = []
        study_type = None
        population = None
        year_min = None
        year_max = None

        # Modality
        for term, mod in self.MODALITY_TERMS.items():
            if term in query_lower and mod not in modality:
                modality.append(mod)

        # Pathology
        for term, path in self.PATHOLOGY_TERMS.items():
            if term in query_lower and path not in pathology:
                pathology.append(path)

        # Study Type
        for term, st in self.STUDY_TYPE_TERMS.items():
            if term in query_lower:
                study_type = st
                break

        # Population
        for term, pop in self.POPULATION_TERMS.items():
            if term in query_lower:
                population = pop
                break

        # 연도 추출
        year_match = re.search(r"\b(20[0-2]\d)\b", query_lower)
        if year_match:
            year = int(year_match.group(1))
            # "since 2020", "after 2020" 패턴
            if re.search(r"(?:since|from|after)\s*" + str(year), query_lower):
                year_min = year
            # "until 2020", "before 2020" 패턴
            elif re.search(r"(?:until|before|to)\s*" + str(year), query_lower):
                year_max = year

        # "last N years" 패턴
        last_years_match = re.search(r"(?:last|recent)\s+(\d+)\s+years?", query_lower)
        if last_years_match:
            years = int(last_years_match.group(1))
            from datetime import datetime
            year_min = datetime.now().year - years

        return QueryFilters(
            modality=modality if modality else None,
            pathology=pathology if pathology else None,
            study_type=study_type,
            population=population,
            year_min=year_min,
            year_max=year_max,
        )

    def _infer_intent(
        self,
        query_lower: str,
        keywords: List[str],
        filters: QueryFilters,
    ) -> str:
        """검색 의도 추론"""
        intent_parts = []

        # 비교 연구
        if "vs" in query_lower or "versus" in query_lower or "compare" in query_lower:
            intent_parts.append("comparison study")

        # 성능 평가
        if any(term in query_lower for term in ["performance", "accuracy", "sensitivity", "specificity"]):
            intent_parts.append("diagnostic performance evaluation")

        # 스크리닝
        if "screening" in query_lower:
            intent_parts.append("screening effectiveness")

        # 기술적
        if any(term in query_lower for term in ["technique", "method", "protocol", "optimization"]):
            intent_parts.append("technical methodology")

        # Modality 언급
        if filters.modality:
            intent_parts.append(f"{', '.join(filters.modality)} related")

        # Pathology 언급
        if filters.pathology:
            intent_parts.append(f"{', '.join(filters.pathology)} detection")

        if not intent_parts:
            intent_parts.append("general mammography research")

        return "; ".join(intent_parts)


# =============================================================================
# Smart Query Parser (Auto-select)
# =============================================================================

class SmartQueryParser:
    """자동으로 LLM/Rule-based 선택하는 파서"""

    def __init__(
        self,
        prefer_llm: bool = True,
        ollama_url: str = "http://localhost:11434",
        model: str = "llama3.2",
    ):
        self.prefer_llm = prefer_llm
        self.rule_parser = QueryParser(use_llm=False)
        self._llm_parser: Optional[LLMQueryParser] = None

        if prefer_llm:
            try:
                self._llm_parser = LLMQueryParser(
                    ollama_url=ollama_url,
                    model=model,
                    fallback_to_rule=True,
                )
                if not self._llm_parser.is_available():
                    logger.info("LLM not available, will use rule-based parser")
                    self._llm_parser = None
            except Exception as e:
                logger.warning(f"Failed to initialize LLM parser: {e}")
                self._llm_parser = None

    def parse(self, query: str) -> SearchQuery:
        """쿼리 파싱 (LLM 우선, 실패시 Rule-based)"""
        if self._llm_parser and self._llm_parser.is_available():
            return self._llm_parser.parse(query)
        return self.rule_parser.parse(query)

    @property
    def using_llm(self) -> bool:
        """LLM 사용 중인지"""
        return self._llm_parser is not None and self._llm_parser.is_available()

    def close(self):
        """리소스 정리"""
        if self._llm_parser:
            self._llm_parser.close()


# 테스트
if __name__ == "__main__":
    import sys

    test_queries = [
        "DBT vs FFDM for microcalcification detection",
        "breast density screening in Korean women since 2020",
        "meta-analysis of tomosynthesis performance",
        "BI-RADS 4 lesion management retrospective study",
        "What is the sensitivity of MRI for detecting breast cancer?",
    ]

    print("=" * 60)
    print("Rule-based Parser Test")
    print("=" * 60)

    rule_parser = QueryParser()
    for query in test_queries:
        result = rule_parser.parse(query)
        print(f"\nQuery: {query}")
        print(f"  Keywords: {result.keywords}")
        print(f"  MeSH: {result.mesh_terms}")
        print(f"  Filters: {result.filters}")
        print(f"  Intent: {result.intent}")

    # LLM 테스트 (--llm 인자 사용 시)
    if "--llm" in sys.argv:
        print("\n" + "=" * 60)
        print("LLM Parser Test")
        print("=" * 60)

        llm_parser = LLMQueryParser()

        if not llm_parser.is_available():
            print("Ollama is not running. Start with: ollama serve")
        else:
            for query in test_queries:
                print(f"\nQuery: {query}")
                try:
                    result = llm_parser.parse(query)
                    print(f"  Intent: {result.intent}")
                    print(f"  Keywords: {result.keywords}")
                    print(f"  MeSH: {result.mesh_terms}")
                    print(f"  Filters: {result.filters}")
                except Exception as e:
                    print(f"  Error: {e}")

            llm_parser.close()
