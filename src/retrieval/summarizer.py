"""
Medical Text Summarizer: SLM 기반 전문 요약 레이어 (v2.9 - Phase 7.2)
====================================================================
Phase 7.1의 핵심: PMC 전문(20,000자+)을 Llama-3.1-8B로 핵심만 추출
Phase 7.2 강화: 임상적 영향(FN/FP 기전, CNR, Rose Criterion) 우선 추출

Workflow:
    1. [Input] PMC 전문 텍스트 (Results, Methods, Discussion 등)
    2. [Filter] SLM이 질문 관련 핵심 수치/결론 + 임상 메트릭 추출
    3. [Output] 1,000자 이내의 정제된 근거 텍스트

"전문을 그대로 R1에게 주면 타임아웃, 요약해서 주면 정확도 상승"
"""

import logging
import re
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
import requests

logger = logging.getLogger(__name__)


@dataclass
class SummaryResult:
    """요약 결과 (v2.9 - Phase 7.2)"""
    original_length: int        # 원본 길이
    summary_length: int         # 요약 길이
    summary: str                # 요약 텍스트
    key_findings: List[str]     # 핵심 발견 목록
    key_numbers: List[str]      # 핵심 수치 목록
    compression_ratio: float    # 압축률
    clinical_outcomes: List[str] = None  # Phase 7.2: 임상적 결과 (FN/FP, CNR, sensitivity 등)

    def __post_init__(self):
        if self.clinical_outcomes is None:
            self.clinical_outcomes = []


@dataclass
class SummarizerConfig:
    """요약기 설정"""
    ollama_url: str = "http://localhost:11434"
    model: str = "glm4:9b"          # SLM for fast summarization
    max_output_chars: int = 1200        # 최대 출력 길이
    timeout: int = 45                    # SLM은 빠르므로 45초
    temperature: float = 0.1            # 낮은 온도로 일관성 유지
    chunk_size: int = 6000              # 섹션별 청크 크기


class MedicalTextSummarizer:
    """
    의학 전문 요약기 (SLM 기반)

    Usage:
        summarizer = MedicalTextSummarizer()

        result = summarizer.summarize(
            full_text="...(20,000자 PMC 전문)...",
            question="DBT에서 MGD 계산 방법은?",
            sections=["results", "methods", "discussion"]
        )

        print(result.summary)  # 1,000자 내외의 핵심 요약
    """

    def __init__(self, config: Optional[SummarizerConfig] = None):
        self.config = config or SummarizerConfig()

    def summarize(
        self,
        full_text: str,
        question: str,
        sections: Optional[List[str]] = None,
        pmc_id: str = ""
    ) -> SummaryResult:
        """
        전문을 질문 관련 핵심만 추출하여 요약

        Args:
            full_text: PMC 전문 텍스트
            question: 사용자 질문 (요약 방향 결정)
            sections: 우선 추출할 섹션 ["results", "methods", "discussion"]
            pmc_id: PMC ID (로깅용)

        Returns:
            SummaryResult: 요약 결과
        """
        original_length = len(full_text)

        # 이미 짧으면 그대로 반환
        if original_length <= self.config.max_output_chars:
            return SummaryResult(
                original_length=original_length,
                summary_length=original_length,
                summary=full_text,
                key_findings=[],
                key_numbers=self._extract_numbers(full_text),
                compression_ratio=1.0,
                clinical_outcomes=self._extract_clinical_outcomes(full_text)  # Phase 7.2
            )

        logger.info(f"Summarizing {pmc_id}: {original_length:,} chars → target {self.config.max_output_chars} chars")

        # 섹션별 추출
        sections = sections or ["results", "methods", "discussion", "conclusions"]
        extracted_sections = self._extract_sections(full_text, sections)

        # SLM 요약 호출
        summary = self._call_slm_summarize(extracted_sections, question)

        # 핵심 수치 추출
        key_numbers = self._extract_numbers(summary)
        key_findings = self._extract_findings(summary)

        # Phase 7.2: 임상적 결과 추출 (원본 + 요약 모두에서)
        clinical_outcomes = self._extract_clinical_outcomes(summary)
        # 원본에서 추가 추출 (요약에서 누락된 경우)
        original_outcomes = self._extract_clinical_outcomes(extracted_sections)
        for outcome in original_outcomes:
            if outcome not in clinical_outcomes:
                clinical_outcomes.append(outcome)

        summary_length = len(summary)
        compression_ratio = summary_length / original_length if original_length > 0 else 1.0

        logger.info(f"Summarized {pmc_id}: {original_length:,} → {summary_length:,} chars ({compression_ratio:.1%})")
        if clinical_outcomes:
            logger.info(f"  Clinical outcomes extracted: {clinical_outcomes[:3]}...")

        return SummaryResult(
            original_length=original_length,
            summary_length=summary_length,
            summary=summary,
            key_findings=key_findings,
            key_numbers=key_numbers,
            compression_ratio=compression_ratio,
            clinical_outcomes=clinical_outcomes  # Phase 7.2
        )

    def summarize_multiple(
        self,
        texts: List[Dict[str, str]],
        question: str
    ) -> List[SummaryResult]:
        """
        여러 전문을 일괄 요약

        Args:
            texts: [{"pmc_id": "PMC123", "text": "..."}, ...]
            question: 사용자 질문

        Returns:
            List[SummaryResult]
        """
        results = []
        for item in texts:
            result = self.summarize(
                full_text=item.get("text", ""),
                question=question,
                pmc_id=item.get("pmc_id", "")
            )
            results.append(result)
        return results

    # =========================================================================
    # Section Extraction
    # =========================================================================

    def _extract_sections(self, full_text: str, sections: List[str]) -> str:
        """섹션별 텍스트 추출"""
        extracted_parts = []
        text_lower = full_text.lower()

        # 섹션 패턴 정의
        section_patterns = {
            "results": [
                r"(?:^|\n)(?:results?|findings?)[\s:]*\n(.*?)(?=\n(?:discussion|conclusion|method|reference)|\Z)",
                r"(?:^|\n)3\.?\s*results?(.*?)(?=\n4\.|\Z)",
            ],
            "methods": [
                r"(?:^|\n)(?:methods?|materials?\s*and\s*methods?)[\s:]*\n(.*?)(?=\n(?:results?|discussion)|\Z)",
                r"(?:^|\n)2\.?\s*(?:methods?|materials?)(.*?)(?=\n3\.|\Z)",
            ],
            "discussion": [
                r"(?:^|\n)(?:discussion)[\s:]*\n(.*?)(?=\n(?:conclusion|reference|acknowledgment)|\Z)",
                r"(?:^|\n)4\.?\s*discussion(.*?)(?=\n5\.|\Z)",
            ],
            "conclusions": [
                r"(?:^|\n)(?:conclusions?|summary)[\s:]*\n(.*?)(?=\n(?:reference|acknowledgment)|\Z)",
            ],
            "abstract": [
                r"(?:^|\n)(?:abstract)[\s:]*\n(.*?)(?=\n(?:introduction|background|keyword)|\Z)",
            ],
        }

        for section in sections:
            patterns = section_patterns.get(section, [])
            for pattern in patterns:
                match = re.search(pattern, full_text, re.IGNORECASE | re.DOTALL)
                if match:
                    section_text = match.group(1).strip()
                    # 청크 크기 제한
                    if len(section_text) > self.config.chunk_size:
                        section_text = section_text[:self.config.chunk_size] + "..."
                    extracted_parts.append(f"[{section.upper()}]\n{section_text}")
                    break

        # 섹션 추출 실패 시 전체 텍스트의 앞부분 사용
        if not extracted_parts:
            truncated = full_text[:self.config.chunk_size * 2]
            extracted_parts.append(f"[CONTENT]\n{truncated}")

        return "\n\n".join(extracted_parts)

    # =========================================================================
    # SLM Summarization
    # =========================================================================

    def _call_slm_summarize(self, section_text: str, question: str) -> str:
        """SLM(Llama-3.1-8B)으로 핵심 요약 (v2.9 - Phase 7.2)"""

        system_prompt = """You are a medical literature summarizer specialized in radiology and mammography.

## Task
Extract ONLY the key findings relevant to the user's question from the provided text.

## Output Requirements
1. **Numerical Data First**: Extract specific numbers, percentages, measurements.
2. **Key Conclusions**: Main findings that directly answer the question.
3. **Clinical Outcomes** (Phase 7.2): Extract diagnostic performance metrics:
   - Sensitivity, Specificity, FN rate, FP rate
   - CNR, SNR, d' (detectability index)
   - Rose Criterion (CNR ≥ 5) mentions
   - Detection rates for microcalcifications, masses
4. **Maximum 800 characters** in Korean.

## Output Format
### 핵심 수치
- [수치 1]
- [수치 2]

### 임상적 결과 (Clinical Outcomes)
- [민감도/특이도/검출율 등]
- [CNR/SNR 변화]

### 주요 발견
- [발견 1]
- [발견 2]

### 결론
[한 문장 요약]

## Rules
- DO NOT add information not in the source text.
- DO NOT include generic background information.
- Focus ONLY on what answers the question.
- PRIORITIZE clinical outcome metrics (sensitivity, CNR, detection rate).
- Use Korean for output."""

        user_prompt = f"""## 사용자 질문
{question}

## 논문 내용 (섹션별)
{section_text[:8000]}

위 내용에서 질문과 관련된 핵심만 추출하세요. 800자 이내로 요약."""

        try:
            response = requests.post(
                f"{self.config.ollama_url}/api/chat",
                json={
                    "model": self.config.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    "stream": False,
                    "options": {
                        "temperature": self.config.temperature,
                        "num_predict": 600,  # 토큰 제한
                    }
                },
                timeout=self.config.timeout
            )
            response.raise_for_status()
            summary = response.json().get("message", {}).get("content", "")

            # 길이 제한 강제
            if len(summary) > self.config.max_output_chars:
                summary = summary[:self.config.max_output_chars] + "..."

            return summary

        except requests.exceptions.Timeout:
            logger.warning(f"SLM timeout, using fallback extraction")
            return self._fallback_extract(section_text, question)

        except Exception as e:
            logger.error(f"SLM summarization error: {e}")
            return self._fallback_extract(section_text, question)

    def _fallback_extract(self, text: str, question: str) -> str:
        """SLM 실패 시 규칙 기반 추출 (Fallback)"""
        # 질문 키워드 추출
        keywords = re.findall(r'\b[a-zA-Z가-힣]{3,}\b', question.lower())

        # 키워드 포함 문장 추출
        sentences = re.split(r'(?<=[.!?])\s+', text)
        relevant = []

        for sent in sentences:
            sent_lower = sent.lower()
            # 수치 포함 또는 키워드 2개 이상 매칭
            has_number = bool(re.search(r'\d+\.?\d*\s*(%|mm|cm|mGy|kV)', sent))
            keyword_match = sum(1 for kw in keywords if kw in sent_lower)

            if has_number or keyword_match >= 2:
                relevant.append(sent)

            if len(" ".join(relevant)) > self.config.max_output_chars:
                break

        if relevant:
            return "[규칙 기반 추출]\n" + " ".join(relevant[:10])
        else:
            # 그래도 없으면 앞부분 반환
            return text[:self.config.max_output_chars]

    # =========================================================================
    # Extraction Helpers
    # =========================================================================

    def _extract_numbers(self, text: str) -> List[str]:
        """텍스트에서 핵심 수치 추출"""
        patterns = [
            r'(\d+\.?\d*)\s*%',                    # 퍼센트
            r'(\d+\.?\d*)\s*(?:mm|cm|mGy|kV|keV)', # 단위 포함
            r'(\d+\.?\d*)\s*(?:배|times|fold)',    # 배수
            r'[<>≤≥]\s*(\d+\.?\d*)',               # 비교
            r'(\d+\.?\d*)\s*[-–]\s*(\d+\.?\d*)',   # 범위
        ]

        numbers = []
        for pattern in patterns:
            matches = re.findall(pattern, text, re.IGNORECASE)
            for match in matches:
                if isinstance(match, tuple):
                    numbers.extend([m for m in match if m])
                else:
                    numbers.append(match)

        # 중복 제거 및 상위 10개
        unique_numbers = list(dict.fromkeys(numbers))
        return unique_numbers[:10]

    def _extract_findings(self, text: str) -> List[str]:
        """핵심 발견 문장 추출"""
        # 결론 지시어가 있는 문장
        conclusion_markers = [
            r'(?:결론|conclusion|결과|result|발견|found|showed|demonstrated|indicated)',
            r'(?:유의|significant|증가|increase|감소|decrease|향상|improve)',
        ]

        sentences = re.split(r'(?<=[.!?])\s+', text)
        findings = []

        for sent in sentences:
            for marker in conclusion_markers:
                if re.search(marker, sent, re.IGNORECASE):
                    # 너무 긴 문장은 자르기
                    if len(sent) > 150:
                        sent = sent[:150] + "..."
                    findings.append(sent)
                    break

            if len(findings) >= 5:
                break

        return findings

    def _extract_clinical_outcomes(self, text: str) -> List[str]:
        """
        Phase 7.2: 임상적 결과 메트릭 추출

        Extracts:
        - Sensitivity, Specificity, FN/FP rates
        - CNR, SNR values and changes
        - Detection rates
        - Rose Criterion mentions
        - d' (detectability index)
        """
        clinical_patterns = [
            # Sensitivity/Specificity
            (r'(?:sensitivity|민감도)[:\s]*(\d+\.?\d*)\s*%?', 'sensitivity'),
            (r'(?:specificity|특이도)[:\s]*(\d+\.?\d*)\s*%?', 'specificity'),

            # False Negative/Positive
            (r'(?:false\s*negative|FN|위음성)[:\s]*(\d+\.?\d*)\s*%?', 'FN rate'),
            (r'(?:false\s*positive|FP|위양성)[:\s]*(\d+\.?\d*)\s*%?', 'FP rate'),

            # CNR/SNR
            (r'(?:CNR|contrast.to.noise)[:\s]*(\d+\.?\d*)', 'CNR'),
            (r'(?:SNR|signal.to.noise)[:\s]*(\d+\.?\d*)', 'SNR'),

            # Detection rate
            (r'(?:detection\s*rate|검출율)[:\s]*(\d+\.?\d*)\s*%?', 'detection rate'),
            (r'(?:detection|검출)[^\d]*(\d+\.?\d*)\s*%', 'detection'),

            # Rose Criterion
            (r'(?:rose\s*criterion|CNR\s*[≥>=]\s*5)', 'Rose Criterion'),

            # d' (detectability index)
            (r"(?:d'|d-prime|detectability)[:\s]*(\d+\.?\d*)", "d'"),

            # AUC/ROC
            (r'(?:AUC|area\s*under)[:\s]*(\d+\.?\d*)', 'AUC'),
        ]

        outcomes = []
        text_lower = text.lower()

        for pattern, metric_name in clinical_patterns:
            matches = re.findall(pattern, text_lower, re.IGNORECASE)
            if matches:
                for match in matches[:2]:  # 최대 2개씩
                    if isinstance(match, str) and match:
                        if metric_name == 'Rose Criterion':
                            outcomes.append(f"{metric_name} 언급됨")
                        else:
                            outcomes.append(f"{metric_name}: {match}")

        # 미세석회화 관련 문장 추출
        calc_patterns = [
            r'microcalcification[s]?.*?(\d+\.?\d*)\s*%',
            r'미세석회화.*?(\d+\.?\d*)\s*%',
        ]
        for pattern in calc_patterns:
            match = re.search(pattern, text_lower)
            if match:
                outcomes.append(f"microcalcification: {match.group(1)}%")

        # 중복 제거
        unique_outcomes = list(dict.fromkeys(outcomes))
        return unique_outcomes[:8]  # 최대 8개


# =============================================================================
# Singleton
# =============================================================================

_summarizer_instance: Optional[MedicalTextSummarizer] = None


def get_medical_summarizer(config: Optional[SummarizerConfig] = None) -> MedicalTextSummarizer:
    """MedicalTextSummarizer 싱글톤"""
    global _summarizer_instance
    if _summarizer_instance is None:
        _summarizer_instance = MedicalTextSummarizer(config)
    return _summarizer_instance


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    import logging
    logging.basicConfig(level=logging.INFO)

    summarizer = MedicalTextSummarizer()

    # 테스트 텍스트 (긴 전문 시뮬레이션)
    test_fulltext = """
    ABSTRACT
    This study evaluates the dose distribution in digital breast tomosynthesis (DBT)
    compared to full-field digital mammography (FFDM).

    METHODS
    We analyzed 500 patients who underwent both DBT and FFDM examinations.
    Mean glandular dose (MGD) was calculated using the Dance method with
    appropriate t-factors for each projection angle. The study used a Hologic
    Selenia Dimensions system operating at 28-32 kVp with W/Rh target/filter.

    RESULTS
    The mean MGD for DBT was 2.1 mGy (range: 1.4-3.2 mGy), compared to
    1.8 mGy for FFDM. The dose increase was 16.7% for DBT versus FFDM.
    For compressed breast thickness of 50mm, the t-factor was 0.919 at 0°
    and decreased to 0.892 at 25° projection angle.

    SNR was 45.2 for DBT and 42.1 for FFDM, showing 7.4% improvement.
    Detection rate for microcalcifications improved by 23% with DBT.

    DISCUSSION
    The slightly higher dose in DBT is offset by improved diagnostic performance.
    The t-factor variation with angle must be considered for accurate MGD calculation.

    CONCLUSIONS
    DBT provides superior image quality with acceptable dose increase.
    The Dance method with angle-dependent t-factors is essential for DBT dosimetry.
    """ * 10  # 반복하여 긴 텍스트 시뮬레이션

    print("=" * 60)
    print("Medical Text Summarizer Test")
    print("=" * 60)
    print(f"원본 길이: {len(test_fulltext):,} chars")

    result = summarizer.summarize(
        full_text=test_fulltext,
        question="DBT와 FFDM의 선량 차이는?",
        pmc_id="PMC_TEST"
    )

    print(f"요약 길이: {result.summary_length:,} chars")
    print(f"압축률: {result.compression_ratio:.1%}")
    print(f"핵심 수치: {result.key_numbers}")
    print(f"임상적 결과 (Phase 7.2): {result.clinical_outcomes}")  # Phase 7.2
    print()
    print("--- 요약 결과 ---")
    print(result.summary)
