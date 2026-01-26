"""
Sophia AI: Medical-Logic-CoT Query Expander
=============================================
DeepSeek-R1 기반 Chain-of-Thought 쿼리 확장

Phase 1 고도화: 의학 키워드 확장 및 추론을 통해 검색 성공률 향상
"""

import json
import logging
import re
from typing import Dict, List, Optional, Tuple

import httpx
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


# =============================================================================
# Medical-Logic-CoT System Prompt
# =============================================================================

MEDICAL_COT_SYSTEM_PROMPT = """You are an expert medical radiologist specialized in breast imaging and mammography literature search.

Your task is to EXPAND the user's search query by reasoning through:
1. Hidden medical concepts implied but not explicitly stated
2. Related technical terminology and synonyms
3. Clinical context and typical research questions in this domain

## REASONING PROCESS (use <think> tags)

Before outputting the final expansion, you MUST think step-by-step inside <think></think> tags:

1. **Understand Intent**: What is the user really asking? What clinical/research question underlies this query?
2. **Identify Implicit Concepts**: What medical concepts are implied but not stated?
3. **Find Semantic Variations**: What synonyms, abbreviations, or alternative terms exist?
4. **Consider Related Topics**: What closely related topics might contain relevant information?
5. **Map to MeSH Terms**: What standard MeSH (Medical Subject Headings) terms apply?

## EXAMPLE REASONING

Query: "BI-RADS 4A"

<think>
1. Understanding Intent:
   - User is asking about BI-RADS category 4A lesions
   - This is a breast imaging classification for lesions with low suspicion for malignancy
   - Likely interested in: malignancy rates, biopsy outcomes, management guidelines

2. Implicit Concepts:
   - "Low suspicion for malignancy" (official definition of 4A)
   - Biopsy recommendation (standard management)
   - Positive Predictive Value (PPV) - key metric for 4A
   - Cancer probability range (>2% to ≤10%)

3. Semantic Variations:
   - "BI-RADS 4a" / "BIRADS 4A" / "BI-RADS category 4A"
   - "probably benign" is 3, "suspicious" is 4
   - "intermediate concern" / "low suspicion"

4. Related Topics:
   - BI-RADS 4B, 4C for comparison
   - Biopsy outcomes
   - False positive rates in screening

5. MeSH Terms:
   - "Breast Imaging Reporting and Data System"
   - "Biopsy"
   - "Predictive Value of Tests"
   - "Breast Neoplasms"
</think>

## OUTPUT FORMAT (JSON)

After reasoning, output a JSON object with:

```json
{
  "expanded_keywords": [
    "list of additional search keywords derived from reasoning"
  ],
  "semantic_variations": [
    "synonyms and alternative spellings"
  ],
  "implicit_concepts": [
    "concepts implied but not stated in original query"
  ],
  "mesh_expansions": [
    "relevant MeSH terms not in original"
  ],
  "search_strategy": "brief description of recommended search approach"
}
```

## MEDICAL DOMAIN KNOWLEDGE

### BI-RADS Categories
- 0: Incomplete - Need additional imaging
- 1: Negative - Normal
- 2: Benign
- 3: Probably Benign - <2% malignancy
- 4A: Low suspicion - 2-10% malignancy, biopsy recommended
- 4B: Moderate suspicion - 10-50% malignancy
- 4C: High suspicion - 50-95% malignancy
- 5: Highly suggestive of malignancy - >95%
- 6: Known biopsy-proven malignancy

### Imaging Modalities
- DBT (Digital Breast Tomosynthesis) = 3D mammography
- FFDM (Full-Field Digital Mammography) = 2D digital mammography
- CEM (Contrast-Enhanced Mammography) = CESM
- SM (Synthetic Mammography) = 2D images from DBT
- MRI = Breast MRI
- US = Ultrasound = Ultrasonography

### Pathology Types
- Mass = tumor, nodule, lump, lesion
- Calcification = microcalcification, macrocalcification
- Architectural distortion
- Asymmetry = focal asymmetry, global asymmetry
- Density = breast density (a, b, c, d categories)

### Key Metrics
- PPV (Positive Predictive Value) = malignancy rate among positive findings
- NPV (Negative Predictive Value)
- Sensitivity = true positive rate
- Specificity = true negative rate
- AUC/ROC = diagnostic accuracy measures
- CDR (Cancer Detection Rate)
- Recall rate = callback rate

### Korean-English Mappings
- 유방 촬영/맘모그래피 = mammography
- 치밀유방/고밀도유방 = dense breast
- 악성률 = malignancy rate
- 양성예측도 = positive predictive value (PPV)
- 조직검사 = biopsy
- 병변/종괴 = lesion, mass
- 석회화 = calcification
- 비대칭 = asymmetry

IMPORTANT: Always output valid JSON after your <think> reasoning."""


# =============================================================================
# DeepSeek-R1 Client
# =============================================================================

class DeepSeekR1Client:
    """DeepSeek-R1 API 클라이언트 (Chain-of-Thought 지원)"""

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "glm4:9b",  # Phase 7.7: 경량 모델
        timeout: float = 60.0,  # 경량 모델 타임아웃
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.timeout = timeout
        self._client = httpx.Client(timeout=timeout)

    def generate_with_thinking(
        self,
        prompt: str,
        system: Optional[str] = None,
        temperature: float = 0.3,  # 약간의 창의성 허용
    ) -> Tuple[str, str]:
        """
        DeepSeek-R1으로 추론 생성

        Returns:
            Tuple of (thinking_content, final_output)
        """
        url = f"{self.base_url}/api/generate"

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": 4096,  # 긴 추론 허용
            },
        }

        if system:
            payload["system"] = system

        try:
            response = self._client.post(url, json=payload)
            response.raise_for_status()
            result = response.json()
            full_response = result.get("response", "")

            # <think>...</think> 태그 파싱
            thinking, output = self._parse_thinking_response(full_response)

            return thinking, output

        except httpx.HTTPError as e:
            logger.error(f"DeepSeek-R1 API error: {e}")
            raise
        except Exception as e:
            logger.error(f"DeepSeek-R1 generate error: {e}")
            raise

    def _parse_thinking_response(self, response: str) -> Tuple[str, str]:
        """
        <think>...</think> 태그 파싱

        Returns:
            Tuple of (thinking_content, final_output)
        """
        # <think> 태그 추출
        think_pattern = r"<think>(.*?)</think>"
        think_match = re.search(think_pattern, response, re.DOTALL)

        if think_match:
            thinking = think_match.group(1).strip()
            # <think> 태그 이후의 내용이 final output
            output = response[think_match.end():].strip()
        else:
            # <think> 태그가 없으면 전체가 output
            thinking = ""
            output = response.strip()

        return thinking, output

    def is_available(self) -> bool:
        """서버 상태 확인"""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                # Phase 7.7: 설정된 모델이 있는지 확인
                data = response.json()
                models = [m["name"] for m in data.get("models", [])]
                return self.model in models or any(self.model.split(":")[0] in m for m in models)
            return False
        except Exception:
            return False

    def list_models(self) -> List[str]:
        """사용 가능한 모델 목록"""
        try:
            response = self._client.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                data = response.json()
                return [m["name"] for m in data.get("models", [])]
        except Exception:
            pass
        return []

    def close(self):
        """클라이언트 종료"""
        self._client.close()


# =============================================================================
# Query Expansion Result
# =============================================================================

class QueryExpansion(BaseModel):
    """CoT 쿼리 확장 결과"""
    expanded_keywords: List[str] = Field(default_factory=list)
    semantic_variations: List[str] = Field(default_factory=list)
    implicit_concepts: List[str] = Field(default_factory=list)
    mesh_expansions: List[str] = Field(default_factory=list)
    search_strategy: str = ""
    reasoning_trace: str = ""  # <think> 내용


# =============================================================================
# Medical-Logic-CoT Query Expander
# =============================================================================

class MedicalCoTQueryExpander:
    """
    Medical-Logic-CoT 쿼리 확장기

    LLM의 Chain-of-Thought 능력을 활용하여:
    1. 질문에 숨겨진 의학 키워드를 추론
    2. 동의어/유사어 확장
    3. 관련 MeSH 용어 제안
    4. 검색 전략 추천
    """

    def __init__(
        self,
        ollama_url: str = "http://localhost:11434",
        model: str = "glm4:9b",  # Phase 7.7: 경량 CoT 모델
        fallback_model: str = "llama3.2",  # fallback
        enable_cot: bool = True,
    ):
        self.ollama_url = ollama_url
        self.model = model
        self.fallback_model = fallback_model
        self.enable_cot = enable_cot

        self._deepseek_client: Optional[DeepSeekR1Client] = None
        self._fallback_client: Optional[httpx.Client] = None

        # DeepSeek-R1 클라이언트 초기화 시도
        if enable_cot:
            try:
                self._deepseek_client = DeepSeekR1Client(
                    base_url=ollama_url,
                    model=model,
                )
                if not self._deepseek_client.is_available():
                    logger.warning(f"DeepSeek-R1 not available, will use fallback")
                    self._deepseek_client = None
            except Exception as e:
                logger.warning(f"Failed to initialize DeepSeek-R1: {e}")

    def expand(self, query: str, existing_keywords: List[str] = None) -> QueryExpansion:
        """
        쿼리 확장 실행

        Args:
            query: 원본 쿼리
            existing_keywords: 이미 추출된 키워드 (중복 방지)

        Returns:
            QueryExpansion 객체
        """
        if not query.strip():
            return QueryExpansion()

        existing = set(existing_keywords or [])

        # DeepSeek-R1 사용 가능하면 CoT 확장
        if self._deepseek_client and self._deepseek_client.is_available():
            return self._expand_with_cot(query, existing)

        # Fallback: 규칙 기반 확장
        logger.info("Using rule-based expansion (DeepSeek-R1 not available)")
        return self._expand_rule_based(query, existing)

    def _expand_with_cot(self, query: str, existing: set) -> QueryExpansion:
        """DeepSeek-R1 CoT로 확장"""
        user_prompt = f"""Expand this breast imaging search query:

Query: "{query}"

Think step-by-step about what medical concepts, synonyms, and related terms should be added to improve search recall. Then output JSON with expanded keywords."""

        try:
            thinking, output = self._deepseek_client.generate_with_thinking(
                prompt=user_prompt,
                system=MEDICAL_COT_SYSTEM_PROMPT,
            )

            logger.info(f"CoT reasoning length: {len(thinking)} chars")
            logger.debug(f"CoT thinking: {thinking[:500]}...")

            # JSON 파싱
            expansion = self._parse_expansion_json(output)
            expansion.reasoning_trace = thinking

            # 기존 키워드와 중복 제거
            expansion.expanded_keywords = [
                kw for kw in expansion.expanded_keywords
                if kw.lower() not in {e.lower() for e in existing}
            ]

            return expansion

        except Exception as e:
            logger.error(f"CoT expansion failed: {e}")
            return self._expand_rule_based(query, existing)

    def _parse_expansion_json(self, output: str) -> QueryExpansion:
        """JSON 출력 파싱"""
        try:
            # JSON 블록 추출 (```json ... ``` 또는 순수 JSON)
            json_str = output.strip()

            if "```json" in json_str:
                start = json_str.index("```json") + 7
                end = json_str.index("```", start)
                json_str = json_str[start:end].strip()
            elif "```" in json_str:
                start = json_str.index("```") + 3
                end = json_str.index("```", start)
                json_str = json_str[start:end].strip()

            # JSON 파싱
            data = json.loads(json_str)

            return QueryExpansion(
                expanded_keywords=data.get("expanded_keywords", []),
                semantic_variations=data.get("semantic_variations", []),
                implicit_concepts=data.get("implicit_concepts", []),
                mesh_expansions=data.get("mesh_expansions", []),
                search_strategy=data.get("search_strategy", ""),
            )

        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse expansion JSON: {e}")
            logger.debug(f"Raw output: {output}")
            return QueryExpansion()

    def _expand_rule_based(self, query: str, existing: set) -> QueryExpansion:
        """규칙 기반 확장 (fallback)"""
        query_lower = query.lower()
        expanded = []
        variations = []
        mesh = []

        # BI-RADS 확장
        birads_expansions = {
            "bi-rads 4a": ["low suspicion malignancy", "PPV 2-10%", "biopsy recommended"],
            "bi-rads 4b": ["moderate suspicion", "PPV 10-50%"],
            "bi-rads 4c": ["high suspicion", "PPV 50-95%"],
            "bi-rads 4": ["suspicious abnormality", "biopsy", "positive predictive value"],
            "bi-rads 3": ["probably benign", "short-term follow-up", "6 month"],
            "bi-rads 5": ["highly suggestive malignancy", "PPV >95%"],
        }

        for pattern, terms in birads_expansions.items():
            if pattern in query_lower or pattern.replace("-", "") in query_lower:
                expanded.extend(terms)
                mesh.extend(["Breast Imaging Reporting and Data System", "Biopsy"])
                variations.append(pattern.replace("-", ""))
                variations.append(pattern.upper())

        # 모달리티 확장
        modality_synonyms = {
            "dbt": ["digital breast tomosynthesis", "3D mammography", "tomosynthesis"],
            "ffdm": ["full-field digital mammography", "2D mammography", "digital mammography"],
            "cem": ["contrast-enhanced mammography", "CESM", "contrast mammography"],
            "mri": ["breast MRI", "magnetic resonance imaging"],
            "ultrasound": ["ultrasonography", "US", "sonography"],
        }

        for mod, synonyms in modality_synonyms.items():
            if mod in query_lower:
                variations.extend(synonyms)
                mesh.append("Mammography")

        # 병변 유형 확장
        pathology_synonyms = {
            "mass": ["tumor", "nodule", "lump", "lesion"],
            "calcification": ["microcalcification", "macrocalcification", "calcium"],
            "density": ["breast density", "dense breast", "fibroglandular tissue"],
            "distortion": ["architectural distortion", "AD"],
            "asymmetry": ["focal asymmetry", "global asymmetry", "developing asymmetry"],
        }

        for path, synonyms in pathology_synonyms.items():
            if path in query_lower:
                variations.extend(synonyms)

        # 한국어 키워드 영어 확장
        korean_mappings = {
            "악성률": ["malignancy rate", "cancer rate", "positive predictive value"],
            "양성예측도": ["positive predictive value", "PPV"],
            "민감도": ["sensitivity", "true positive rate"],
            "특이도": ["specificity", "true negative rate"],
            "치밀유방": ["dense breast", "breast density category c d"],
            "조직검사": ["biopsy", "tissue sampling", "histopathology"],
            "유방암": ["breast cancer", "breast carcinoma", "breast neoplasm"],
        }

        for korean, english in korean_mappings.items():
            if korean in query:
                expanded.extend(english)

        # 중복 제거
        expanded = list(set(expanded) - existing)
        variations = list(set(variations) - existing)
        mesh = list(set(mesh))

        return QueryExpansion(
            expanded_keywords=expanded,
            semantic_variations=variations,
            mesh_expansions=mesh,
            search_strategy="Rule-based expansion applied",
        )

    @property
    def using_cot(self) -> bool:
        """CoT 사용 중인지"""
        return (
            self._deepseek_client is not None
            and self._deepseek_client.is_available()
        )

    def close(self):
        """리소스 정리"""
        if self._deepseek_client:
            self._deepseek_client.close()
        if self._fallback_client:
            self._fallback_client.close()


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    test_queries = [
        "BI-RADS 4A",
        "DBT vs FFDM 진단 정확도",
        "유방 밀도가 암 발견율에 미치는 영향",
        "breast mass biopsy outcome",
        "치밀유방 악성률",
    ]

    expander = MedicalCoTQueryExpander()

    print("=" * 60)
    print(f"Using CoT: {expander.using_cot}")
    print("=" * 60)

    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 40)

        expansion = expander.expand(query)

        print(f"Expanded Keywords: {expansion.expanded_keywords[:5]}")
        print(f"Semantic Variations: {expansion.semantic_variations[:5]}")
        print(f"MeSH Expansions: {expansion.mesh_expansions[:3]}")
        print(f"Strategy: {expansion.search_strategy}")

        if expansion.reasoning_trace:
            print(f"Reasoning (first 200 chars): {expansion.reasoning_trace[:200]}...")

    expander.close()
