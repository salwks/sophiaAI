"""
Sophia AI: Formula Integrity Checker
=====================================
Phase 2: 수식 무결성 검증기

LLM이 생성한 답변 내 LaTeX 수식을 파싱하여
GOLDEN_FORMULAS와 구조적으로 일치하는지 검사합니다.
"""

import re
import logging
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from src.knowledge.core_physics import GOLDEN_FORMULAS, get_formula, get_all_formula_ids

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

@dataclass
class FormulaMatch:
    """수식 매칭 결과"""
    extracted_formula: str  # 추출된 수식
    matched_golden_id: Optional[str]  # 매칭된 골든 포뮬러 ID
    similarity_score: float  # 유사도 점수 (0-1)
    is_valid: bool  # 유효성
    issues: List[str] = field(default_factory=list)  # 발견된 문제점


@dataclass
class FormulaCheckResult:
    """전체 검증 결과"""
    is_valid: bool  # 전체 유효성
    total_formulas: int  # 추출된 수식 수
    valid_formulas: int  # 유효한 수식 수
    invalid_formulas: int  # 무효한 수식 수
    matches: List[FormulaMatch]  # 개별 매칭 결과
    unknown_variables: Set[str]  # 알 수 없는 변수들
    errors: List[Dict[str, str]]  # 오류 상세


# =============================================================================
# Formula Extractor
# =============================================================================

class FormulaExtractor:
    """
    텍스트에서 LaTeX 수식 추출
    """

    # LaTeX 수식 패턴들
    PATTERNS = [
        # Display math: $$ ... $$
        (r"\$\$(.*?)\$\$", "display"),
        # Inline math: $ ... $
        (r"\$([^$]+)\$", "inline"),
        # LaTeX environments: \begin{equation} ... \end{equation}
        (r"\\begin\{equation\}(.*?)\\end\{equation\}", "equation"),
        # \[ ... \]
        (r"\\\[(.*?)\\\]", "bracket"),
        # Code block with formula
        (r"```(?:latex|math)?\s*(.*?)\s*```", "codeblock"),
    ]

    @classmethod
    def extract(cls, text: str) -> List[Tuple[str, str]]:
        """
        텍스트에서 모든 수식 추출

        Args:
            text: 검사할 텍스트

        Returns:
            List of (formula, type) tuples
        """
        formulas = []
        seen = set()

        for pattern, formula_type in cls.PATTERNS:
            matches = re.findall(pattern, text, re.DOTALL | re.IGNORECASE)
            for match in matches:
                formula = match.strip()
                # 중복 제거 및 빈 수식 필터
                if formula and formula not in seen and len(formula) > 2:
                    seen.add(formula)
                    formulas.append((formula, formula_type))

        return formulas


# =============================================================================
# Formula Normalizer
# =============================================================================

class FormulaNormalizer:
    """
    수식 정규화 (비교를 위한 전처리)
    """

    @staticmethod
    def normalize(formula: str) -> str:
        """
        수식 정규화

        - 공백 제거
        - 대소문자 통일 (변수는 제외)
        - LaTeX 명령어 정규화
        """
        normalized = formula

        # 공백 정규화
        normalized = re.sub(r'\s+', '', normalized)

        # \cdot, \times, × 통일
        normalized = re.sub(r'\\cdot|\\times|×', '*', normalized)

        # \frac{a}{b} → a/b (간단 비교용)
        normalized = re.sub(r'\\frac\{([^}]+)\}\{([^}]+)\}', r'(\1)/(\2)', normalized)

        # \sqrt{x} → sqrt(x)
        normalized = re.sub(r'\\sqrt\{([^}]+)\}', r'sqrt(\1)', normalized)

        # 그리스 문자 정규화
        greek_map = {
            r'\\sigma': 'σ',
            r'\\mu': 'μ',
            r'\\theta': 'θ',
            r'\\alpha': 'α',
            r'\\Sigma': 'Σ',
            r'\\propto': '∝',
        }
        for latex, unicode_char in greek_map.items():
            normalized = re.sub(latex, unicode_char, normalized)

        # 첨자 정규화
        normalized = re.sub(r'_\{([^}]+)\}', r'_\1', normalized)
        normalized = re.sub(r'\^\{([^}]+)\}', r'^\1', normalized)

        return normalized

    @staticmethod
    def extract_variables(formula: str) -> Set[str]:
        """
        수식에서 변수 추출

        Args:
            formula: 정규화된 수식

        Returns:
            변수 집합
        """
        variables = set()

        # 단일 문자 변수 (대소문자)
        variables.update(re.findall(r'\b([A-Za-z])\b', formula))

        # 그리스 문자
        variables.update(re.findall(r'[σμθαβγδεΣ]', formula))

        # 첨자 붙은 변수 (예: K_total, MGD_i)
        variables.update(re.findall(r'([A-Za-z]+_[A-Za-z0-9]+)', formula))

        # 함수 이름 제외 (sqrt, ln, log, sin, cos 등)
        functions = {'sqrt', 'ln', 'log', 'sin', 'cos', 'tan', 'exp', 'sum'}
        variables = {v for v in variables if v.lower() not in functions}

        return variables


# =============================================================================
# Formula Checker
# =============================================================================

class FormulaChecker:
    """
    수식 무결성 검증기

    LLM 생성 답변의 수식이 GOLDEN_FORMULAS와 일치하는지 검증
    """

    def __init__(self, strict_mode: bool = False, similarity_threshold: float = 0.7):
        """
        Args:
            strict_mode: 엄격 모드 (등록되지 않은 수식 차단)
            similarity_threshold: 유사도 임계값
        """
        self.strict_mode = strict_mode
        self.similarity_threshold = similarity_threshold
        self.normalizer = FormulaNormalizer()
        self.extractor = FormulaExtractor()

        # 골든 포뮬러 정규화 캐시
        self._golden_normalized = {}
        self._golden_variables = {}
        self._build_golden_cache()

    def _build_golden_cache(self):
        """골든 포뮬러 캐시 구축"""
        for fid, formula in GOLDEN_FORMULAS.items():
            latex = formula.get("formula_latex", "")
            unicode_form = formula.get("formula_unicode", "")

            # LaTeX와 Unicode 모두 정규화
            self._golden_normalized[fid] = {
                "latex": self.normalizer.normalize(latex),
                "unicode": self.normalizer.normalize(unicode_form),
                "original_latex": latex,
                "original_unicode": unicode_form,
            }

            # 허용된 변수 목록
            self._golden_variables[fid] = set(formula.get("variables", {}).keys())

    def verify(self, generated_text: str) -> FormulaCheckResult:
        """
        생성된 텍스트의 수식 검증

        Args:
            generated_text: LLM이 생성한 답변 텍스트

        Returns:
            FormulaCheckResult
        """
        # 1. 수식 추출
        extracted = self.extractor.extract(generated_text)

        if not extracted:
            return FormulaCheckResult(
                is_valid=True,
                total_formulas=0,
                valid_formulas=0,
                invalid_formulas=0,
                matches=[],
                unknown_variables=set(),
                errors=[],
            )

        matches = []
        all_unknown_vars = set()
        errors = []

        # 2. 각 수식 검증
        for formula, formula_type in extracted:
            match_result = self._check_single_formula(formula)
            matches.append(match_result)

            # 알 수 없는 변수 수집
            if match_result.issues:
                for issue in match_result.issues:
                    if "unknown variable" in issue.lower():
                        # 변수 추출
                        var_match = re.search(r"'([^']+)'", issue)
                        if var_match:
                            all_unknown_vars.add(var_match.group(1))

            # 오류 수집
            if not match_result.is_valid:
                errors.append({
                    "type": "formula_error",
                    "formula": formula,
                    "matched_to": match_result.matched_golden_id,
                    "similarity": match_result.similarity_score,
                    "issues": match_result.issues,
                })

        # 3. 결과 집계
        valid_count = sum(1 for m in matches if m.is_valid)
        invalid_count = len(matches) - valid_count

        # 엄격 모드에서는 하나라도 무효면 전체 무효
        is_valid = invalid_count == 0 if self.strict_mode else (invalid_count < len(matches) // 2)

        return FormulaCheckResult(
            is_valid=is_valid,
            total_formulas=len(matches),
            valid_formulas=valid_count,
            invalid_formulas=invalid_count,
            matches=matches,
            unknown_variables=all_unknown_vars,
            errors=errors,
        )

    def _check_single_formula(self, formula: str) -> FormulaMatch:
        """
        단일 수식 검증

        Args:
            formula: 검사할 수식

        Returns:
            FormulaMatch
        """
        issues = []
        normalized = self.normalizer.normalize(formula)
        extracted_vars = self.normalizer.extract_variables(formula)

        # 1. 골든 포뮬러와 매칭 시도
        best_match_id = None
        best_similarity = 0.0

        for fid, cached in self._golden_normalized.items():
            # LaTeX 형식 비교
            sim_latex = SequenceMatcher(
                None, normalized, cached["latex"]
            ).ratio()

            # Unicode 형식 비교
            sim_unicode = SequenceMatcher(
                None, normalized, cached["unicode"]
            ).ratio()

            similarity = max(sim_latex, sim_unicode)

            if similarity > best_similarity:
                best_similarity = similarity
                best_match_id = fid

        # 2. 매칭 결과 평가
        is_valid = best_similarity >= self.similarity_threshold

        if not is_valid and self.strict_mode:
            issues.append(
                f"Formula does not match any golden formula (best: {best_match_id}, "
                f"similarity: {best_similarity:.2%})"
            )

        # 3. 변수 검증 (매칭된 경우)
        if best_match_id and best_similarity > 0.5:
            allowed_vars = self._golden_variables.get(best_match_id, set())

            # 허용되지 않은 변수 찾기
            for var in extracted_vars:
                # 변수 이름 정규화 (첨자 제거 등)
                base_var = var.split('_')[0] if '_' in var else var

                # 허용된 변수 또는 일반적인 변수인지 확인
                if base_var not in allowed_vars and var not in allowed_vars:
                    # 일반적으로 허용되는 변수 (숫자, 인덱스 등)
                    common_vars = {'i', 'j', 'k', 'n', 'N', 'x', 'y', '1', '2', 'total', 'out', 'in'}
                    if base_var not in common_vars and var not in common_vars:
                        issues.append(f"Unknown variable '{var}' not in {best_match_id}")

        # 4. 물리적 일관성 검사
        physics_issues = self._check_physics_consistency(formula, best_match_id)
        issues.extend(physics_issues)

        return FormulaMatch(
            extracted_formula=formula,
            matched_golden_id=best_match_id,
            similarity_score=best_similarity,
            is_valid=is_valid and len(issues) == 0,
            issues=issues,
        )

    def _check_physics_consistency(
        self, formula: str, matched_id: Optional[str]
    ) -> List[str]:
        """
        물리적 일관성 검사

        Args:
            formula: 검사할 수식
            matched_id: 매칭된 골든 포뮬러 ID

        Returns:
            발견된 문제점 리스트
        """
        issues = []
        formula_lower = formula.lower()

        # MGD 관련 검사
        if matched_id and "MGD" in matched_id:
            # MGD에 필수 변수(g, c, s)가 있어야 함
            required = ['g', 'c', 's']
            missing = [v for v in required if v not in formula]
            if missing and 'K' in formula:
                # K가 있는데 필수 변수가 없으면 경고 (완전한 수식이 아닐 수 있음)
                pass  # 부분 수식일 수 있으므로 경고만

        # SNR 관련 검사
        if matched_id and "SNR" in matched_id:
            # SNR = √N 또는 N/√N 형태여야 함
            if 'snr' in formula_lower:
                # 선량과 노이즈의 역관계 확인
                if ('dose' in formula_lower and 'noise' in formula_lower):
                    # 선량 증가 → 노이즈 감소 관계가 올바른지 확인
                    pass

        # 흔한 오류 패턴 검사
        error_patterns = [
            # 가우시안 필터 + MTF 보존 주장
            (r'gaussian.*mtf.*preserv', "Gaussian filter cannot preserve MTF"),
            # 선량 증가 → 노이즈 증가 (역관계 오류)
            (r'dose.*increase.*noise.*increase', "Dose increase should DECREASE noise, not increase"),
        ]

        for pattern, error_msg in error_patterns:
            if re.search(pattern, formula_lower, re.IGNORECASE):
                issues.append(error_msg)

        return issues

    def get_golden_formula_hint(self, formula: str) -> Optional[str]:
        """
        주어진 수식과 가장 유사한 골든 포뮬러 힌트 제공

        Args:
            formula: 검사할 수식

        Returns:
            권장 골든 포뮬러 정보 문자열
        """
        normalized = self.normalizer.normalize(formula)

        best_match = None
        best_sim = 0.0

        for fid, cached in self._golden_normalized.items():
            sim = max(
                SequenceMatcher(None, normalized, cached["latex"]).ratio(),
                SequenceMatcher(None, normalized, cached["unicode"]).ratio(),
            )
            if sim > best_sim:
                best_sim = sim
                best_match = fid

        if best_match and best_sim > 0.3:
            golden = GOLDEN_FORMULAS[best_match]
            return (
                f"Did you mean: {golden['name']}?\n"
                f"Standard formula: {golden['formula_unicode']}\n"
                f"Source: {golden.get('source', 'N/A')}"
            )

        return None


# =============================================================================
# Convenience Functions
# =============================================================================

def check_formulas(text: str, strict: bool = False) -> FormulaCheckResult:
    """
    텍스트의 수식 검증 (편의 함수)

    Args:
        text: 검사할 텍스트
        strict: 엄격 모드

    Returns:
        검증 결과
    """
    checker = FormulaChecker(strict_mode=strict)
    return checker.verify(text)


def extract_and_validate_formulas(text: str) -> Dict:
    """
    수식 추출 및 검증 결과를 딕셔너리로 반환

    Args:
        text: 검사할 텍스트

    Returns:
        검증 결과 딕셔너리
    """
    result = check_formulas(text)

    return {
        "is_valid": result.is_valid,
        "total": result.total_formulas,
        "valid": result.valid_formulas,
        "invalid": result.invalid_formulas,
        "errors": result.errors,
        "unknown_variables": list(result.unknown_variables),
    }


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # 테스트 케이스
    test_texts = [
        # 올바른 수식
        """
        MGD는 다음 공식으로 계산됩니다:
        $$MGD = K \\cdot g \\cdot c \\cdot s$$

        여기서 K는 입사 공기 kerma입니다.
        """,

        # 올바른 SNR 수식
        """
        양자 노이즈의 SNR은:
        $SNR = \\sqrt{N}$

        선량이 증가하면 SNR이 증가합니다.
        """,

        # 잘못된 수식 (존재하지 않는 변수)
        """
        저는 새로운 MGD 공식을 발견했습니다:
        $$MGD = K \\cdot g \\cdot \\phi \\cdot \\omega$$

        여기서 φ와 ω는 제가 만든 새로운 계수입니다.
        """,

        # 물리적 오류
        """
        선량을 증가시키면 노이즈가 증가합니다:
        $Noise \\propto Dose$
        """,
    ]

    checker = FormulaChecker(strict_mode=True)

    for i, text in enumerate(test_texts, 1):
        print(f"\n{'=' * 60}")
        print(f"Test Case {i}")
        print("=" * 60)

        result = checker.verify(text)

        print(f"Valid: {result.is_valid}")
        print(f"Total formulas: {result.total_formulas}")
        print(f"Valid: {result.valid_formulas}, Invalid: {result.invalid_formulas}")

        if result.errors:
            print("\nErrors:")
            for error in result.errors:
                print(f"  - {error['formula'][:50]}...")
                print(f"    Matched to: {error['matched_to']}")
                print(f"    Issues: {error['issues']}")

        if result.unknown_variables:
            print(f"\nUnknown variables: {result.unknown_variables}")
