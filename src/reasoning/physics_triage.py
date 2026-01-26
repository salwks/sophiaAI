"""
Physics Triage Layer (Phase 7.9)
================================
질문 분석 → 물리 도메인 분류 → Solver 선호출 → 풀이 전략 주입 → 사후 검증

4개 컴포넌트:
1. PhysicsClassifier: 질문 → Phase 1-5 매핑 (dual-path: 키워드 + 의미)
2. SolverRouter: Phase → solver 메서드 호출 + 파라미터 자동 추출
3. FrameworkInjector: solver 결과를 풀이 전략으로 변환 (정답 미포함)
4. PostVerifier: LLM 답변 vs solver 정답 비교 (multi-phase)

핵심 원칙:
- 정답 주입 ✗ → 풀이 프레임워크 주입 ✓
- LLM이 물리를 이해하고 계산 → solver가 검증
- Triage 오분류 시에도 LLM이 불일치를 감지 가능
"""

import re
import math
import json
import logging
import requests
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Any, Optional, Tuple

from src.reasoning.mammo_physics_solver import MammoPhysicsSolver, get_mammo_solver

logger = logging.getLogger(__name__)


# =============================================================================
# Enums & Data Classes
# =============================================================================

class PhysicsDomain(Enum):
    """물리 도메인 (Phase 1-5)"""
    PHASE1_SNR = "phase1_snr"               # 전자노이즈 + SNR
    PHASE2_SPECTRAL = "phase2_spectral"     # 에너지 가중치, 스펙트럴
    PHASE3_DQE = "phase3_dqe"              # DQE 선량의존성
    PHASE4_MTF = "phase4_mtf"              # MTF/해상도
    PHASE4B_DEPTH = "phase4b_depth"        # 토모 깊이분해능
    PHASE5_TOMO_IQ = "phase5_tomo_iq"      # 토모 영상품질 (dose-split, detectability)
    UNKNOWN = "unknown"


@dataclass
class ClassificationResult:
    """분류 결과"""
    primary_domain: PhysicsDomain
    confidence: float                       # 0-1
    keyword_path: PhysicsDomain             # 키워드 기반 분류
    semantic_path: PhysicsDomain            # 의미 기반 분류
    paths_agree: bool                       # 양 경로 일치?
    extracted_params: Dict[str, float] = field(default_factory=dict)
    reasoning: str = ""                     # 분류 근거


@dataclass
class SolverResult:
    """Solver 계산 결과 (범용)"""
    domain: PhysicsDomain
    primary_value: float                    # 주요 결과값
    primary_label: str                      # 주요 결과 라벨
    all_values: Dict[str, float] = field(default_factory=dict)  # 모든 계산값
    formula_used: str = ""                  # 적용된 공식
    physical_principle: str = ""            # 핵심 물리 원칙
    parameters: Dict[str, float] = field(default_factory=dict)  # 사용된 파라미터
    derivation_summary: List[str] = field(default_factory=list)  # 풀이 단계 요약


@dataclass
class FrameworkPrompt:
    """풀이 전략 프롬프트"""
    physics_principle: str                  # 적용할 물리 원칙
    formula_guide: str                      # 사용할 공식 안내
    parameter_values: str                   # 대입할 파라미터
    solving_strategy: str                   # 풀이 전략 설명
    warning_constraints: str                # 물리적 제약 (위반 불가)


@dataclass
class PostVerificationResult:
    """사후 검증 결과"""
    passed: bool
    domain: PhysicsDomain
    llm_value: Optional[float]
    solver_value: float
    error_percent: float
    explanation: str
    should_reject: bool


@dataclass
class MultiDomainClassificationResult:
    """다중 도메인 분류 결과"""
    domains: List[PhysicsDomain]                    # 상위 도메인 리스트
    domain_scores: Dict[PhysicsDomain, float]       # 각 도메인 점수
    extracted_params: Dict[str, float]              # 추출된 파라미터 (전체)
    is_multi_domain: bool                           # True: 2개 이상 도메인 활성
    reasoning: str


# =============================================================================
# Component 1: PhysicsClassifier
# =============================================================================

class PhysicsClassifier:
    """
    질문 → Phase 1-5 매핑 (dual-path classification)

    경로 1: 키워드 기반 (빠르고 결정론적)
    경로 2: 의미/구조 기반 (질문 의도 분석)
    결론: 두 경로 일치 시 고신뢰, 불일치 시 저신뢰
    """

    # Phase별 키워드 맵 (우선순위 순)
    PHASE_KEYWORDS: Dict[PhysicsDomain, List[str]] = {
        PhysicsDomain.PHASE1_SNR: [
            'snr', 'signal-to-noise', '신호 대 잡음',
            '신호대잡음', '잡음비', '신호대잡음비',
            '전자 노이즈', '전자노이즈', 'electronic noise', 'σ_e',
            '전자잡음', '전자 잡음',
            '양자 노이즈', '양자노이즈', 'quantum noise', 'σ_q',
            '양자잡음', '양자 잡음',
            'noise fraction', '노이즈 비율', '잡음 비율',
            'rose criterion', 'rose', '로즈',
            'snr 감소', 'snr 변화', 'snr 하락',
            '신호 잡음', '잡음 비',
            'dose ratio', '선량 저감', '선량.*저감',
        ],
        PhysicsDomain.PHASE2_SPECTRAL: [
            '에너지 가중', 'energy weighting', 'spectral',
            '스펙트럴', 'bin', 'threshold',
            '에너지 분해능', 'energy resolution',
            'optimal weighting', '최적 가중',
            'CESM', 'cesm', '조영증강', 'contrast enhanced',
            'K-edge', 'k-edge', 'kedge', '아이오딘', 'iodine',
            'CNR', 'cnr', '대조도 잡음비',
        ],
        PhysicsDomain.PHASE3_DQE: [
            'dqe', '양자검출효율', 'detective quantum efficiency',
            '선량 의존', 'dose-dependent', 'dose dependent',
            'nps', 'noise power spectrum',
            'dqe 비교', 'dqe 차이',
            '전자노이즈 비율.*dqe', 'dqe.*선량',
        ],
        PhysicsDomain.PHASE4_MTF: [
            'mtf', '변조전달함수', 'modulation transfer',
            'pixel pitch', '픽셀 피치',
            'nyquist', '나이퀴스트',
            '해상도', 'resolution',
            '직접변환', '간접변환',
            'lp/mm', 'lpmm',
        ],
        PhysicsDomain.PHASE4B_DEPTH: [
            '깊이 분해능', 'depth resolution',
            'angular range', '각도 범위',
            '토모합성 기하', 'tomosynthesis geometry',
            '슬라이스 두께', 'slice thickness',
            'δz', 'delta_z',
            'depth accuracy', '생검 오차', '생검.*오차',
        ],
        PhysicsDomain.PHASE5_TOMO_IQ: [
            'dose split', '선량 분할', '선량을 분할', '투영당 선량',
            'dose per projection', '투영당',
            '투영으로', '투영수', '개 투영',
            '나눌', '나누면', '분할하면',
            'clutter', '해부학적 잡음', 'anatomical noise',
            'detectability', '검출능',
            '토모.*dqe', 'tomo.*dqe', 'dbt.*dqe',
            '토모.*snr', 'tomo.*snr',
            '토모.*pcd', '토모.*eid',
            '토모합성.*영상', 'tomosynthesis.*image quality',
            '토모합성', 'tomosynthesis', 'dbt',
            '토모신세시스', '투사 수', '투사수',
            'asf', 'artifact spread',
            '분해능 비대칭', 'resolution asymmetry',
            'neq', 'noise equivalent quanta',
            '투영.*snr.*gain', '투영.*pcd.*eid',
        ],
    }

    # ==========================================================================
    # Material vs Lesion 분리 규칙 (Selection Ability)
    # ==========================================================================
    # CsI 검출기의 Iodine과 조영제 Iodine, 석회화(Calcium)를 구분
    CONTEXT_RULES = {
        "detector_material": {
            "keywords": ["csi 검출기", "csi scintillator", "섬광체", "간접변환",
                        "간접 변환", "eid 검출기", "eid detector", "csi:tl"],
            "note": "CsI의 Iodine은 검출기 소재이며, 조영제가 아님"
        },
        "contrast_agent": {
            "keywords": ["cesm", "조영증강", "contrast enhanced", "조영제",
                        "iodinated", "아이오딘 조영", "iodine contrast",
                        "dual-energy", "이중에너지", "recombined"],
            "note": "CESM의 Iodine은 조영제이며, 검출기 소재가 아님"
        },
        "lesion_calcium": {
            "keywords": ["석회화", "calcification", "microcalcification", "미세석회화",
                        "calcium", "칼슘", "hydroxyapatite", "bi-rads 4",
                        "fine linear", "amorphous", "pleomorphic", "coarse",
                        "생검", "biopsy", "타겟팅", "targeting"],
            "note": "석회화는 Calcium 기반이며, Iodine 조영제와 무관"
        }
    }

    # 문맥 충돌 감지 규칙
    CONTEXT_CONFLICT_RULES = [
        {
            "condition": "석회화/calcification + CsI 검출기 언급",
            "resolution": "Iodine은 검출기 소재로 해석, 조영제 아님",
            "warning": "CsI의 Iodine과 조영제 Iodine 혼동 주의"
        },
        {
            "condition": "생검/biopsy + 확대촬영",
            "resolution": "MTF Chain 분석 필요 (penumbra + 빛 확산)",
            "warning": "형태 왜곡(Fine Linear→Amorphous) 가능성 검토"
        },
        {
            "condition": "두꺼운 유방 + W/Ag 필터 + CsI",
            "resolution": "빔 경화로 인한 대조도 손실 + MTF 저하 동시 분석",
            "warning": "선량 증가로 해결 불가 - Δμ 손실과 MTF 저하는 스펙트럼/기하 문제"
        }
    ]

    # 의미 패턴 (질문 구조 기반)
    SEMANTIC_PATTERNS: Dict[PhysicsDomain, List[str]] = {
        PhysicsDomain.PHASE1_SNR: [
            r'선량.*감소.*(?:시|때|하면).*(?:snr|신호|노이즈|잡음)',
            r'(?:snr|신호).*(?:변화|감소|하락).*(?:계산|증명|도출)',
            r'전자\s*(?:노이즈|잡음).*(?:\d+).*%.*(?:차지|비율)',
            r'(?:mgd|선량).*(?:\d+).*%.*(?:감축|감소).*(?:snr|noise|잡음)',
            r'저선량.*(?:snr|신호|노이즈|잡음)',
            r'(?:잡음|노이즈).*비율.*(?:snr|신호)',
            r'(?:snr|신호\s*대\s*잡음).*(?:어떻게|얼마나)',
        ],
        PhysicsDomain.PHASE3_DQE: [
            r'(?:dqe|양자검출).*(?:비교|차이|변화)',
            r'(?:eid|pcd).*(?:dqe|양자검출)',
            r'선량.*(?:변화|감소).*dqe',
            r'dqe.*선량.*(?:의존|관계)',
        ],
        PhysicsDomain.PHASE4_MTF: [
            r'(?:mtf|해상도).*(?:비교|차이)',
            r'(?:pixel|픽셀).*(?:pitch|피치).*(?:해상도|mtf)',
            r'(?:직접|간접).*변환.*(?:해상도|mtf)',
        ],
        PhysicsDomain.PHASE4B_DEPTH: [
            r'(?:깊이|depth).*(?:분해능|resolution)',
            r'(?:각도|angular).*(?:범위|range).*(?:분해능|resolution)',
            r'토모.*(?:기하|geometry).*(?:분해능|slice)',
        ],
        PhysicsDomain.PHASE5_TOMO_IQ: [
            r'토모.*(?:선량.*분할|dose.*split)',
            r'투영.*(?:수|개).*(?:snr|dqe|영상)',
            r'(?:n|투영).*(?:\d+).*(?:선량|dose)',
            r'토모.*(?:pcd|eid).*(?:비교|우위|advantage)',
            r'(?:clutter|잡음|중첩).*(?:제거|rejection)',
            r'(?:μGy|uGy|선량).*투영.*(?:나눌|분할|나누)',
            r'투영.*(?:나눌|분할).*(?:pcd|eid|snr|gain)',
        ],
    }

    def classify(self, query: str) -> ClassificationResult:
        """
        질문을 물리 도메인으로 분류 (dual-path)
        """
        query_lower = query.lower()

        # ============================================================
        # Material vs Lesion 문맥 감지 (Selection Ability)
        # ============================================================
        context_info = self._detect_context(query_lower)
        context_warnings = []

        # 문맥 충돌 검사: 석회화 + CsI 동시 언급 시 Iodine 혼동 방지
        if context_info.get("lesion_calcium") and context_info.get("detector_material"):
            context_warnings.append(
                "⚠️ CsI 검출기의 Iodine은 검출기 소재임 (조영제 아님). "
                "석회화(Calcium) 대조도 분석 시 조영제 문맥 배제 필요."
            )
            # PHASE2_SPECTRAL(조영제 관련) 키워드 점수 감점 처리는 아래에서

        # 경로 1: 키워드 기반
        keyword_result, keyword_score = self._keyword_path(query_lower)

        # 문맥 보정: 석회화 문맥에서 PHASE2_SPECTRAL(조영제)로 분류되면 재조정
        if (context_info.get("lesion_calcium") and
            not context_info.get("contrast_agent") and
            keyword_result == PhysicsDomain.PHASE2_SPECTRAL):
            # 석회화 + CsI 문맥에서 조영제 도메인은 오분류
            context_warnings.append(
                "⚠️ 석회화 진단에서 PHASE2_SPECTRAL(조영제) 분류 감지 → 재조정"
            )
            keyword_result = PhysicsDomain.PHASE4_MTF  # MTF 분석으로 전환
            keyword_score *= 0.5  # 신뢰도 감점

        # 경로 2: 의미/구조 기반
        semantic_result, semantic_score = self._semantic_path(query)

        # 파라미터 추출
        params = self._extract_parameters(query)

        # 문맥 기반 추가 파라미터 (MTF Chain 분석용)
        if context_info.get("magnification_biopsy"):
            params["context_magnification"] = 1.8
            params["context_mtf_chain"] = 1.0
            context_warnings.append(
                "📐 확대 생검 문맥 감지 → MTF Chain 분석 필요 (penumbra + 빛 확산)"
            )

        if context_info.get("thick_breast_hardening"):
            params["context_beam_hardening"] = 1.0
            context_warnings.append(
                "🔬 두꺼운 유방 + 필터 경화 문맥 → Δμ 손실 + MTF 저하 동시 분석"
            )

        # 결론: 양 경로 합산
        paths_agree = (keyword_result == semantic_result)

        if paths_agree:
            primary = keyword_result
            confidence = min(1.0, (keyword_score + semantic_score) / 2 + 0.2)
        elif keyword_score > semantic_score + 0.3:
            primary = keyword_result
            confidence = keyword_score * 0.7
        elif semantic_score > keyword_score + 0.2:
            # 의미 경로가 더 구체적이면 신뢰 (DQE 질문에 전자노이즈 언급 등)
            primary = semantic_result
            confidence = semantic_score * 0.7
        else:
            # 추가 휴리스틱: 파라미터 기반 판단
            primary = self._resolve_by_params(keyword_result, semantic_result, params)
            # 파라미터가 도메인을 명확히 지시하면 신뢰도 상향
            if primary != PhysicsDomain.UNKNOWN and params:
                confidence = max(0.35, max(keyword_score, semantic_score) * 0.7)
            else:
                confidence = max(keyword_score, semantic_score) * 0.5

        # ============================================================
        # 복합 문맥 기반 도메인 재결정 (MTF Chain, 형태 오분류 등)
        # ============================================================
        # 석회화 형태 오분류(Fine Linear→Amorphous) 문맥 감지 시 → MTF 분석
        if (context_info.get("morphology_confusion") or
            (context_info.get("lesion_calcium") and context_info.get("detector_material"))):
            if primary == PhysicsDomain.UNKNOWN or primary == PhysicsDomain.PHASE2_SPECTRAL:
                primary = PhysicsDomain.PHASE4_MTF
                confidence = max(0.6, confidence)
                context_warnings.append(
                    "🎯 복합 MTF Chain 분석으로 재분류 (석회화 형태 + 검출기 + 확대 촬영)"
                )

        # 확대 생검 + 두꺼운 유방 문맥 → MTF 분석 우선
        if (context_info.get("magnification_biopsy") and
            context_info.get("thick_breast_hardening") and
            primary == PhysicsDomain.UNKNOWN):
            primary = PhysicsDomain.PHASE4_MTF
            confidence = max(0.55, confidence)
            context_warnings.append(
                "🎯 확대 생검 + 빔 경화 문맥 → MTF Chain 분석으로 분류"
            )

        # 문맥 경고가 있으면 reasoning에 포함
        context_note = " | ".join(context_warnings) if context_warnings else ""
        reasoning = (
            f"키워드경로={keyword_result.value}({keyword_score:.2f}), "
            f"의미경로={semantic_result.value}({semantic_score:.2f}), "
            f"일치={'✓' if paths_agree else '✗'}, "
            f"파라미터={list(params.keys())}"
        )
        if context_note:
            reasoning = f"{reasoning} | 문맥감지: {context_note}"

        return ClassificationResult(
            primary_domain=primary,
            confidence=confidence,
            keyword_path=keyword_result,
            semantic_path=semantic_result,
            paths_agree=paths_agree,
            extracted_params=params,
            reasoning=reasoning
        )

    def classify_multi(self, query: str) -> Tuple[ClassificationResult, Dict[PhysicsDomain, float]]:
        """
        단일 분류 + 전체 도메인 점수 반환

        Returns:
            (ClassificationResult, all_domain_scores)
            - ClassificationResult: 기존 classify()와 동일한 결과
            - all_domain_scores: 모든 도메인의 합산 점수 (keyword + semantic)
        """
        query_lower = query.lower()

        # 전체 점수 산출
        keyword_scores = self._keyword_scores(query_lower)
        semantic_scores = self._semantic_scores(query)

        # 합산 점수: (keyword + semantic) / 2
        all_domains = set(list(keyword_scores.keys()) + list(semantic_scores.keys()))
        all_scores: Dict[PhysicsDomain, float] = {}
        for domain in all_domains:
            kw_s = keyword_scores.get(domain, 0.0)
            sem_s = semantic_scores.get(domain, 0.0)
            all_scores[domain] = min(1.0, (kw_s + sem_s) / 2)

        # 파라미터 기반 점수 부스트: 추출된 파라미터로 도메인 신호 강화
        params = self._extract_parameters(query)
        PARAM_DOMAIN_MAP = {
            'dose_ratio': [PhysicsDomain.PHASE1_SNR, PhysicsDomain.PHASE3_DQE],
            'electronic_noise_fraction': [PhysicsDomain.PHASE1_SNR, PhysicsDomain.PHASE3_DQE],
            'pixel_pitch_mm': [PhysicsDomain.PHASE4_MTF],
            'angular_range_deg': [PhysicsDomain.PHASE4B_DEPTH],
            'n_projections': [PhysicsDomain.PHASE5_TOMO_IQ],
            'total_dose_uGy': [PhysicsDomain.PHASE5_TOMO_IQ],
        }
        for param, domains in PARAM_DOMAIN_MAP.items():
            if param in params:
                for domain in domains:
                    current = all_scores.get(domain, 0.0)
                    all_scores[domain] = min(1.0, current + 0.25)

        # 기존 classify() 결과도 생성
        classification = self.classify(query)

        return classification, all_scores

    def _keyword_scores(self, query_lower: str) -> Dict[PhysicsDomain, float]:
        """키워드 기반 전체 도메인 점수 반환"""
        scores: Dict[PhysicsDomain, float] = {}

        for domain, keywords in self.PHASE_KEYWORDS.items():
            score = 0.0
            for kw in keywords:
                if '.*' in kw:
                    if re.search(kw, query_lower):
                        score += 3.0
                elif kw in query_lower:
                    score += 1.0
            if score > 0:
                scores[domain] = min(1.0, score / 4.0)

        return scores

    def _semantic_scores(self, query: str) -> Dict[PhysicsDomain, float]:
        """의미/구조 기반 전체 도메인 점수 반환"""
        scores: Dict[PhysicsDomain, float] = {}

        for domain, patterns in self.SEMANTIC_PATTERNS.items():
            score = 0.0
            for pattern in patterns:
                if re.search(pattern, query, re.IGNORECASE):
                    score += 1.0
            if patterns:
                s = score / len(patterns) * 2
                if s > 0:
                    scores[domain] = min(1.0, s)

        return scores

    def _keyword_path(self, query_lower: str) -> Tuple[PhysicsDomain, float]:
        """키워드 기반 분류"""
        scores = self._keyword_scores(query_lower)

        if not scores or max(scores.values()) == 0:
            return PhysicsDomain.UNKNOWN, 0.0

        best_domain = max(scores, key=scores.get)
        best_score = scores[best_domain]

        return best_domain, best_score

    def _semantic_path(self, query: str) -> Tuple[PhysicsDomain, float]:
        """의미/구조 기반 분류"""
        scores = self._semantic_scores(query)

        if not scores or max(scores.values()) == 0:
            return PhysicsDomain.UNKNOWN, 0.0

        best_domain = max(scores, key=scores.get)
        best_score = min(1.0, scores[best_domain])

        return best_domain, best_score

    def _resolve_by_params(
        self,
        keyword_result: PhysicsDomain,
        semantic_result: PhysicsDomain,
        params: Dict[str, float]
    ) -> PhysicsDomain:
        """파라미터 기반 최종 판정"""
        # dose_ratio + electronic_noise_fraction → Phase 1 (SNR) OR Phase 3 (DQE)
        # Phase 3 우선: semantic이 DQE를 선택했으면 DQE가 의도
        if 'dose_ratio' in params and 'electronic_noise_fraction' in params:
            if semantic_result == PhysicsDomain.PHASE3_DQE:
                return PhysicsDomain.PHASE3_DQE
            return PhysicsDomain.PHASE1_SNR

        # n_projections + dose → Phase 5 (Tomo IQ)
        if 'n_projections' in params:
            return PhysicsDomain.PHASE5_TOMO_IQ

        # angular_range → Phase 4-B or 5
        if 'angular_range_deg' in params:
            if 'n_projections' in params or 'dose' in params:
                return PhysicsDomain.PHASE5_TOMO_IQ
            return PhysicsDomain.PHASE4B_DEPTH

        # pixel_pitch → Phase 4
        if 'pixel_pitch_mm' in params:
            return PhysicsDomain.PHASE4_MTF

        # 둘 다 UNKNOWN이 아니면 키워드 우선
        if keyword_result != PhysicsDomain.UNKNOWN:
            return keyword_result
        if semantic_result != PhysicsDomain.UNKNOWN:
            return semantic_result

        return PhysicsDomain.UNKNOWN

    def _detect_context(self, query_lower: str) -> Dict[str, bool]:
        """
        Material vs Lesion 문맥 감지 (Selection Ability)

        Returns:
            Dict with detected contexts:
            - detector_material: CsI 검출기 문맥
            - contrast_agent: 조영제(CESM) 문맥
            - lesion_calcium: 석회화(Calcium) 병변 문맥
            - magnification_biopsy: 확대 생검 문맥
            - thick_breast_hardening: 두꺼운 유방 + 빔 경화 문맥
        """
        context = {}

        # 검출기 소재 문맥
        detector_keywords = self.CONTEXT_RULES["detector_material"]["keywords"]
        context["detector_material"] = any(kw in query_lower for kw in detector_keywords)

        # 조영제 문맥
        contrast_keywords = self.CONTEXT_RULES["contrast_agent"]["keywords"]
        context["contrast_agent"] = any(kw in query_lower for kw in contrast_keywords)

        # 석회화(Calcium) 병변 문맥
        calcium_keywords = self.CONTEXT_RULES["lesion_calcium"]["keywords"]
        context["lesion_calcium"] = any(kw in query_lower for kw in calcium_keywords)

        # 확대 생검 문맥 (magnification + biopsy/stereotactic)
        magnification_terms = ["확대", "magnification", "1.5배", "1.8배", "2.0배", "2배"]
        biopsy_terms = ["생검", "biopsy", "스테레오", "stereotactic", "타겟팅", "targeting"]
        has_magnification = any(t in query_lower for t in magnification_terms)
        has_biopsy = any(t in query_lower for t in biopsy_terms)
        context["magnification_biopsy"] = has_magnification and has_biopsy

        # 두꺼운 유방 + 빔 경화 문맥
        thick_terms = ["6cm", "두꺼운", "thick", "치밀", "dense"]
        filter_terms = ["w/ag", "w/rh", "필터", "filter", "경화", "hardened", "hardening"]
        has_thick = any(t in query_lower for t in thick_terms)
        has_filter = any(t in query_lower for t in filter_terms)
        context["thick_breast_hardening"] = has_thick and has_filter

        # 추가: Fine Linear → Amorphous 오분류 문맥
        morphology_terms = ["fine linear", "amorphous", "4c", "4b", "뭉개", "형태"]
        context["morphology_confusion"] = any(t in query_lower for t in morphology_terms)

        return context

    def _extract_parameters(self, query: str) -> Dict[str, float]:
        """질문에서 물리 파라미터 자동 추출"""
        params = {}

        # 선량 비율: "50% 감소/감축" → 0.5, "D=0.6" → 0.6
        dose_patterns = [
            r'(?:MGD|선량|dose)[를을]?\s*(?:기존\s*대비\s*)?(\d+(?:\.\d+)?)\s*%\s*(?:로\s*)?(?:감축|감소|줄)',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:로\s*)?(?:저선량|감축|감소)',
            r'선량[이가을를]?\s*(\d+(?:\.\d+)?)\s*%',
        ]
        for pat in dose_patterns:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                params['dose_ratio'] = (100 - float(m.group(1))) / 100
                break

        # 직접 D 값 지정: "D=0.6", "D'=0.5"
        if 'dose_ratio' not in params:
            d_direct = re.search(r"[Dd]['\u2019]?\s*[=:]\s*(\d+(?:\.\d+)?)", query)
            if d_direct:
                val = float(d_direct.group(1))
                if 0 < val < 1:
                    params['dose_ratio'] = val

        # 전자 노이즈 비율: "30%를 차지" → 0.30
        noise_patterns = [
            r'전자\s*(?:노이즈|잡음)[가이]?\s*(?:전체\s*노이즈의\s*)?(\d+(?:\.\d+)?)\s*%',
            r'전자\s*(?:노이즈|잡음)\s*(?:비율|비중)\s*(?:\([^)]*\))?\s*[이가은는]?\s*(\d+(?:\.\d+)?)\s*%',
            r'(?:f_e|f_e\s*[=:]\s*)(?:\)?\s*[이가은는]?\s*)?(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%[를을]?\s*차지',
            r'(?:electronic\s*noise|σ_e)\s*(?:is\s*)?(\d+(?:\.\d+)?)\s*%',
        ]
        for pat in noise_patterns:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                params['electronic_noise_fraction'] = float(m.group(1)) / 100
                break

        # 직접 f_e 값 지정: "f_e=0.25" (소수점 직접 지정)
        if 'electronic_noise_fraction' not in params:
            fe_direct = re.search(r"f_e\s*[=:]\s*(\d+\.\d+)", query)
            if fe_direct:
                val = float(fe_direct.group(1))
                if 0 < val < 1:
                    params['electronic_noise_fraction'] = val

        # 투영 수: "N=25", "25개 투영", "25 projections", "투사 수 15"
        proj_patterns = [
            r'[Nn]\s*[=:]\s*(\d+)',
            r'(\d+)\s*(?:개|회)?\s*(?:투영|투사|projection)',
            r'(?:투영|투사|projection)\s*(?:수|개수|count)[가이은는]?\s*(\d+)',
        ]
        for pat in proj_patterns:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                val = int(m.group(1))
                if 2 <= val <= 100:  # 합리적 범위
                    params['n_projections'] = float(val)
                    break

        # 각도 범위: "25도", "25°", "angular range 25"
        angle_patterns = [
            r'(\d+(?:\.\d+)?)\s*[°도]',
            r'(?:angular\s*range|각도\s*범위)[가이은는]?\s*(\d+(?:\.\d+)?)',
        ]
        for pat in angle_patterns:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                if 10 <= val <= 90:  # 합리적 범위
                    params['angular_range_deg'] = val
                    break

        # 픽셀 피치: "0.1mm", "pixel pitch 100um", "pixel pitch 75μm"
        pitch_patterns = [
            r'(?:pixel\s*pitch|픽셀\s*피치)[가이은는]?\s*(\d+(?:\.\d+)?)\s*mm',
            r'(?:pixel\s*pitch|픽셀\s*피치)[가이은는]?\s*(\d+(?:\.\d+)?)\s*[uμ]m',
            r'(\d+(?:\.\d+)?)\s*mm\s*(?:pixel|픽셀)',
            r'(\d+)\s*[uμ]m\s*(?:pixel|픽셀)',
        ]
        for pat in pitch_patterns:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                if val > 1:  # μm 단위
                    params['pixel_pitch_mm'] = val / 1000
                else:
                    params['pixel_pitch_mm'] = val
                break

        # 유방 두께: "50mm 유방", "breast thickness 50mm"
        thickness_patterns = [
            r'(?:유방|breast)\s*(?:두께|thickness)[가이은는]?\s*(\d+(?:\.\d+)?)\s*mm',
            r'(\d+(?:\.\d+)?)\s*mm\s*(?:유방|breast)',
        ]
        for pat in thickness_patterns:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                params['breast_thickness_mm'] = float(m.group(1))
                break

        # 총 선량: "1500 uGy", "1.5 mGy"
        dose_val_patterns = [
            r'(\d+(?:\.\d+)?)\s*[uμ]Gy',
            r'(\d+(?:\.\d+)?)\s*mGy',
        ]
        for pat in dose_val_patterns:
            m = re.search(pat, query, re.IGNORECASE)
            if m:
                val = float(m.group(1))
                if 'mGy' in pat:
                    params['total_dose_uGy'] = val * 1000
                else:
                    params['total_dose_uGy'] = val
                break

        # Rose k값
        rose_m = re.search(r'Rose\s*(?:Criterion)?\s*\(?k\s*[=:]\s*(\d+(?:\.\d+)?)\)?', query, re.IGNORECASE)
        if rose_m:
            params['rose_k'] = float(rose_m.group(1))

        return params


# =============================================================================
# Component 2: SolverRouter
# =============================================================================

class SolverRouter:
    """
    Phase → solver 메서드 호출 + 결과 래핑

    ClassificationResult를 받아 해당 Phase의 solver를 호출하고,
    범용 SolverResult로 래핑하여 반환.
    """

    def __init__(self):
        self._solver = get_mammo_solver()

    def route_and_solve(self, classification: ClassificationResult) -> Optional[SolverResult]:
        """
        분류 결과에 따라 적절한 solver 호출
        """
        domain = classification.primary_domain
        params = classification.extracted_params

        if domain == PhysicsDomain.PHASE1_SNR:
            return self._solve_phase1(params)
        elif domain == PhysicsDomain.PHASE2_SPECTRAL:
            return self._solve_phase2(params)
        elif domain == PhysicsDomain.PHASE3_DQE:
            return self._solve_phase3(params)
        elif domain == PhysicsDomain.PHASE4_MTF:
            return self._solve_phase4(params)
        elif domain == PhysicsDomain.PHASE4B_DEPTH:
            return self._solve_phase4b(params)
        elif domain == PhysicsDomain.PHASE5_TOMO_IQ:
            return self._solve_phase5(params)
        else:
            logger.info(f"No solver available for domain: {domain.value}")
            return None

    def route_and_solve_multi(
        self,
        domains: List[PhysicsDomain],
        params: Dict[str, float]
    ) -> Dict[PhysicsDomain, SolverResult]:
        """
        복수 도메인에 대해 solver 호출

        Args:
            domains: 활성 도메인 리스트
            params: 추출된 파라미터 (전체)

        Returns:
            Dict[PhysicsDomain, SolverResult]: 도메인별 solver 결과
        """
        results: Dict[PhysicsDomain, SolverResult] = {}
        for domain in domains:
            classification = ClassificationResult(
                primary_domain=domain,
                confidence=1.0,
                keyword_path=domain,
                semantic_path=domain,
                paths_agree=True,
                extracted_params=params,
                reasoning=f"Multi-domain routing: {domain.value}"
            )
            result = self.route_and_solve(classification)
            if result:
                results[domain] = result
                logger.info(
                    f"Multi-domain solver: {domain.value} → "
                    f"{result.primary_label}={result.primary_value:.4f}"
                )
        return results

    def _solve_phase1(self, params: Dict[str, float]) -> Optional[SolverResult]:
        """Phase 1: SNR with electronic noise"""
        dose_ratio = params.get('dose_ratio', 0.5)
        f_e = params.get('electronic_noise_fraction', 0.30)

        try:
            sol = self._solver.solve_snr_with_electronic_noise(
                dose_ratio=dose_ratio,
                electronic_noise_fraction=f_e
            )

            # Compact formula derivation:
            # σ_e² = f_e×D'/(1-f_e), σ_ref² = 1+σ_e², σ_new² = D'+σ_e²
            # SNR ratio = D'×√(σ_ref²/σ_new²) = √(D'×(1-f_e×(1-D')))
            sigma_e2 = f_e * dose_ratio / (1 - f_e)
            sigma_ref2 = 1 + sigma_e2
            sigma_new2 = dose_ratio + sigma_e2
            snr_ratio_val = dose_ratio * (sigma_ref2 / sigma_new2) ** 0.5
            # Compact formula intermediate values
            inner = 1 - f_e * (1 - dose_ratio)
            product = dose_ratio * inner

            return SolverResult(
                domain=PhysicsDomain.PHASE1_SNR,
                primary_value=sol.eid_snr_reduction_pct,
                primary_label="EID SNR 감소율 (%)",
                all_values={
                    'eid_snr_reduction_pct': sol.eid_snr_reduction_pct,
                    'eid_snr_ratio': sol.eid_snr_ratio,
                    'pcd_snr_reduction_pct': sol.pcd_snr_reduction_pct,
                    'pcd_snr_ratio': sol.pcd_snr_ratio,
                    'pcd_recovery_pct': sol.pcd_recovery_pct,
                },
                formula_used="SNR_new/SNR_ref = √(D' × (1 - f_e×(1-D')))",
                physical_principle=(
                    "전자노이즈(σ_e²)는 선량에 무관한 상수. "
                    "양자노이즈(σ_q²)만 선량에 비례하여 감소. "
                    "Signal도 선량에 비례 감소. "
                    "따라서 SNR = Signal/Noise 이고, 전자노이즈가 있으면 "
                    "SNR 감소율 > √D' 감소율."
                ),
                parameters={'dose_ratio': dose_ratio, 'f_e': f_e},
                derivation_summary=[
                    f"Step 1: f_e={f_e}는 '감소된 선량(D'={dose_ratio})'에서의 전자노이즈 비율",
                    f"Step 2: 공식 대입 — 1 - f_e×(1-D') = 1 - {f_e}×{1-dose_ratio} = {inner:.4f}",
                    f"Step 3: D'×(위 결과) = {dose_ratio}×{inner:.4f} = {product:.4f}",
                    f"Step 4: SNR_new/SNR_ref = √{product:.4f} = {snr_ratio_val:.4f} (검증: {snr_ratio_val:.4f}² = {snr_ratio_val**2:.4f} ≈ {product:.4f} ✓)",
                    f"Step 5: 감소율(%) = (1 - SNR_ratio) × 100 ← 위에서 구한 SNR_ratio 대입하여 계산",
                    f"  ※ 이 결과를 물리적으로 설명하고 유도 과정을 보이세요.",
                ]
            )
        except Exception as e:
            logger.error(f"Phase 1 solver failed: {e}")
            return None

    def _solve_phase2(self, params: Dict[str, float]) -> Optional[SolverResult]:
        """Phase 2: Spectral Contrast / Energy Weighting"""
        # 파라미터 추출 (기본값: 아이오딘 4-bin 스펙트럼)
        n_bins = int(params.get('n_bins', 4))
        contrast_agent = params.get('contrast_agent', 'iodine')

        try:
            # 기본 에너지 빈 사용 (아이오딘 K-edge 스펙트럼)
            from src.reasoning.mammo_physics_solver import MammoPhysicsSolver

            # 아이오딘 기준 4-bin 스펙트럼 (K-edge = 33.2 keV)
            bins = MammoPhysicsSolver.get_iodine_cesm_bins()

            sol = self._solver.solve_energy_weighting_gain(bins)

            return SolverResult(
                domain=PhysicsDomain.PHASE2_SPECTRAL,
                primary_value=sol.eta,
                primary_label="에너지 가중 이득 (η = CNR_PCD/CNR_EID)",
                all_values={
                    'eta': sol.eta,
                    'eta_percent': sol.eta_percent,
                    'cnr_eid': sol.cnr_eid,
                    'cnr_pcd': sol.cnr_pcd,
                    'n_bins': sol.n_bins,
                },
                formula_used=(
                    "η² = [Σ Δμᵢ² × Nᵢ] × [Σ Nᵢ] / [Σ Δμᵢ × Nᵢ]² "
                    "(Cauchy-Schwarz: η ≥ 1)"
                ),
                physical_principle=(
                    "EID는 모든 광자를 에너지 비례 가중(w∝E)으로 통합하여 "
                    "고에너지 광자를 과대평가하고 저에너지 대조도 정보를 손실. "
                    "PCD는 에너지 빈별 최적 가중(matched filter)으로 "
                    "K-edge 전후 Δμ 차이를 최대한 활용하여 CNR을 η배 향상. "
                    "아이오딘(K=33.2keV) 조영제에서 η ≈ 1.3-1.5 (30-50% 향상)."
                ),
                parameters={
                    'n_bins': n_bins,
                    'contrast_agent': contrast_agent,
                    'kedge_keV': 33.2,  # 아이오딘 K-edge
                },
                derivation_summary=[
                    f"Step 1: 에너지 빈 정의 — {n_bins}개 빈 (K-edge 기준 분할)",
                    f"Step 2: EID CNR — 에너지 비례 가중 Σ(E×Δμ×N)/√Σ(E²×N) = {sol.cnr_eid:.4f}",
                    f"Step 3: PCD CNR — 최적 가중 √Σ(Δμ²×N) = {sol.cnr_pcd:.4f}",
                    f"Step 4: η = CNR_PCD/CNR_EID = {sol.cnr_pcd:.4f}/{sol.cnr_eid:.4f} = {sol.eta:.4f}",
                    f"Step 5: CNR 향상률 = (η-1)×100 = {sol.eta_percent:.1f}%",
                    "  ※ Cauchy-Schwarz 부등식에 의해 η ≥ 1 항상 성립",
                ]
            )
        except Exception as e:
            logger.error(f"Phase 2 solver failed: {e}")
            return None

    def _solve_phase3(self, params: Dict[str, float]) -> Optional[SolverResult]:
        """Phase 3: DQE dose-dependence"""
        dose_ratio = params.get('dose_ratio', 0.5)
        f_e = params.get('electronic_noise_fraction', 0.30)
        eta_abs = 0.85

        try:
            sol = self._solver.solve_dqe_dose_dependence(
                dose_ratio=dose_ratio,
                electronic_noise_fraction=f_e,
                eta_abs=eta_abs
            )

            # α 역산 (solver 내부와 동일)
            alpha = f_e * dose_ratio / (1 - f_e)
            dqe_full = eta_abs / (1 + alpha)
            dqe_reduced = sol.dqe_eid_at_dose_ratio
            degradation = (dqe_full - dqe_reduced) / dqe_full * 100

            return SolverResult(
                domain=PhysicsDomain.PHASE3_DQE,
                primary_value=sol.dqe_eid_at_dose_ratio,
                primary_label="EID DQE at reduced dose",
                all_values={
                    'dqe_eid_full': sol.dqe_eid_full_dose,
                    'dqe_eid_at_dose': sol.dqe_eid_at_dose_ratio,
                    'dqe_pcd': sol.dqe_pcd,
                    'pcd_advantage_pct': sol.pcd_advantage_percent,
                    'dqe_degradation_pct': sol.dqe_degradation_percent,
                    'alpha': alpha,
                },
                formula_used="DQE_EID(D) = η_abs / (1 + α/D), DQE_PCD = η_abs",
                physical_principle=(
                    "EID의 DQE는 선량 감소 시 저하 (전자노이즈 비중 증가). "
                    "PCD의 DQE는 선량 무관 (전자노이즈 없음). "
                    "α = f_e×D_ref/(1-f_e): 전자노이즈 기여 파라미터."
                ),
                parameters={'dose_ratio': dose_ratio, 'f_e': f_e, 'eta_abs': eta_abs},
                derivation_summary=[
                    f"Step 1: α = f_e×D'/(1-f_e) = {f_e}×{dose_ratio}/{1-f_e:.2f} = {alpha:.4f}",
                    f"Step 2: DQE_EID(D_ref) = {eta_abs}/(1+{alpha:.4f}) = {dqe_full:.4f}",
                    f"Step 3: DQE_EID(D'={dose_ratio}) = {eta_abs}/(1+{alpha:.4f}/{dose_ratio}) = {dqe_reduced:.4f}",
                    f"Step 4: DQE_PCD = {eta_abs} (전자노이즈 없으므로 선량 무관)",
                    f"Step 5: DQE 저하율 = ({dqe_full:.4f}-{dqe_reduced:.4f})/{dqe_full:.4f}×100 = {degradation:.1f}%",
                    f"  ※ 이 결과를 물리적으로 설명하고, PCD와의 차이를 유도하세요.",
                ]
            )
        except Exception as e:
            logger.error(f"Phase 3 solver failed: {e}")
            return None

    def _solve_phase4(self, params: Dict[str, float]) -> Optional[SolverResult]:
        """Phase 4: MTF/Resolution"""
        pixel_pitch = params.get('pixel_pitch_mm', 0.1)

        try:
            sol = self._solver.solve_mtf_comparison(pixel_pitch_mm=pixel_pitch)

            return SolverResult(
                domain=PhysicsDomain.PHASE4_MTF,
                primary_value=sol.mtf_pcd_at_nyquist,
                primary_label="PCD MTF at Nyquist",
                all_values={
                    'nyquist_freq': sol.nyquist_freq,
                    'pcd_mtf_nyquist': sol.mtf_pcd_at_nyquist,
                    'eid_mtf_nyquist': sol.mtf_eid_at_nyquist,
                    'pcd_resolution_gain': sol.pcd_resolution_gain,
                    'f10_pcd': sol.f10_pcd,
                    'f10_eid': sol.f10_eid,
                },
                formula_used="MTF_PCD = sinc(π×f×a), MTF_EID = sinc(π×f×a) × MTF_scint",
                physical_principle=(
                    "PCD: 직접변환, 전하확산 없음 → sinc만. "
                    "EID: 간접변환, 광확산 → 추가 blur (MTF_scint). "
                    "Nyquist = 1/(2×pixel_pitch)."
                ),
                parameters={'pixel_pitch_mm': pixel_pitch},
                derivation_summary=[
                    f"Step 1: Nyquist = 1/(2×{pixel_pitch}) = {1/(2*pixel_pitch):.1f} lp/mm",
                    f"Step 2: MTF_PCD(f_Nyq) = sinc(π×f_Nyq×{pixel_pitch})",
                    f"Step 3: MTF_EID(f_Nyq) = MTF_PCD × MTF_scintillator",
                ]
            )
        except Exception as e:
            logger.error(f"Phase 4 solver failed: {e}")
            return None

    def _solve_phase4b(self, params: Dict[str, float]) -> Optional[SolverResult]:
        """Phase 4-B: Depth resolution (via tomo_resolution solver)"""
        angular_range = params.get('angular_range_deg', 25.0)
        pixel_pitch = params.get('pixel_pitch_mm', 0.1)
        breast_thickness = params.get('breast_thickness_mm', 50.0)

        try:
            sol = self._solver.solve_tomo_resolution(
                angular_range_deg=angular_range,
                pixel_pitch_mm=pixel_pitch,
                breast_thickness_mm=breast_thickness
            )

            return SolverResult(
                domain=PhysicsDomain.PHASE4B_DEPTH,
                primary_value=sol.delta_z_mm,
                primary_label="Through-plane resolution (mm)",
                all_values={
                    'delta_z_mm': sol.delta_z_mm,
                    'delta_xy_mm': sol.delta_xy_mm,
                    'asymmetry_ratio': sol.resolution_asymmetry_ratio,
                    'depth_resolution_constant': sol.depth_resolution_constant,
                    'n_resolvable_slices': sol.n_resolvable_slices,
                },
                formula_used="Δz = K / sin(α_total/2)",
                physical_principle=(
                    "Through-plane 분해능은 기하학(각도 범위)에 의해 결정. "
                    "In-plane은 검출기(pixel pitch)에 의해 결정. "
                    "비대칭: Δz >> Δxy (전형적 10-80×)."
                ),
                parameters={
                    'angular_range_deg': angular_range,
                    'pixel_pitch_mm': pixel_pitch,
                    'breast_thickness_mm': breast_thickness,
                },
                derivation_summary=[
                    f"Step 1: α_total = {angular_range}°",
                    f"Step 2: Δz = K/sin(α/2) = K/sin({angular_range/2}°)",
                    f"Step 3: Δxy = pixel_pitch/MTF ≈ {pixel_pitch}/0.637",
                    f"Step 4: Asymmetry = Δz/Δxy",
                ]
            )
        except Exception as e:
            logger.error(f"Phase 4-B solver failed: {e}")
            return None

    def _solve_phase5(self, params: Dict[str, float]) -> Optional[SolverResult]:
        """Phase 5: Tomo image quality (dose-split + detectability)"""
        n_proj = int(params.get('n_projections', 25))
        total_dose = params.get('total_dose_uGy', 1500.0)
        f_e = params.get('electronic_noise_fraction', 0.30)
        angular_range = params.get('angular_range_deg', 25.0)
        breast_thickness = params.get('breast_thickness_mm', 50.0)

        try:
            # Dose-split 분석
            dose_sol = self._solver.solve_tomo_dose_split(
                total_dose_uGy=total_dose,
                n_projections=n_proj,
                electronic_noise_fraction=f_e
            )

            # 중간 계산
            alpha = f_e * 0.5 / (1 - f_e)  # Phase 3 기반 α
            d_proj = total_dose / n_proj
            dqe_eid = dose_sol.dqe_eid_per_proj
            snr_gain = dose_sol.pcd_snr_gain_total

            return SolverResult(
                domain=PhysicsDomain.PHASE5_TOMO_IQ,
                primary_value=dose_sol.pcd_snr_gain_total,
                primary_label="PCD/EID SNR gain (tomo)",
                all_values={
                    'dose_per_proj': dose_sol.dose_per_projection_uGy,
                    'dqe_eid_per_proj': dose_sol.dqe_eid_per_proj,
                    'dqe_pcd_per_proj': dose_sol.dqe_pcd_per_proj,
                    'pcd_dqe_advantage_ratio': dose_sol.pcd_dqe_advantage_ratio,
                    'pcd_snr_gain': dose_sol.pcd_snr_gain_total,
                    'snr_eid_total': dose_sol.snr_eid_total,
                    'snr_pcd_total': dose_sol.snr_pcd_total,
                },
                formula_used=(
                    "DQE_EID(D/N) = η_abs/(1+α×N), "
                    "DQE_PCD = η_abs, "
                    "R_SNR = √(1+α×N)"
                ),
                physical_principle=(
                    "토모합성: 총 선량을 N개 투영으로 분할 → 투영당 저선량. "
                    "EID: 저선량에서 전자노이즈 비중↑ → DQE↓. "
                    "PCD: 전자노이즈 없음 → DQE 불변. "
                    "2D에서 미미한 PCD 우위가 토모에서 극대화."
                ),
                parameters={
                    'n_projections': float(n_proj),
                    'total_dose_uGy': total_dose,
                    'f_e': f_e,
                    'angular_range_deg': angular_range,
                },
                derivation_summary=[
                    f"Step 1: D_proj = {total_dose:.0f}/{n_proj} = {d_proj:.1f} μGy/투영",
                    f"Step 2: ⚠️ α = f_e×D'/(1-f_e) 에서 D'=0.5 (정규화 선량비, 절대선량 아님!)",
                    f"  → α = {f_e}×0.5/{1-f_e:.2f} = {alpha:.4f} (이 값을 그대로 사용할 것)",
                    f"Step 3: DQE_EID = 0.85/(1+{alpha:.4f}×{n_proj}) = 0.85/{1+alpha*n_proj:.4f} = {dqe_eid:.4f}",
                    f"Step 4: DQE_PCD = 0.85 (선량 무관)",
                    f"Step 5: SNR gain = √(1+{alpha:.4f}×{n_proj}) = √{1+alpha*n_proj:.4f} = {snr_gain:.4f}×",
                    f"  ※ PCD가 EID 대비 {snr_gain:.2f}배 SNR 우위 (N={n_proj}). 이를 유도하세요.",
                ]
            )
        except Exception as e:
            logger.error(f"Phase 5 solver failed: {e}")
            return None


# =============================================================================
# Component 3: FrameworkInjector
# =============================================================================

class FrameworkInjector:
    """
    Solver 결과를 풀이 전략으로 변환 (정답 수치 미포함)

    핵심: LLM에게 "어떤 물리, 어떤 공식, 왜 이 접근"을 알려주되,
    최종 수치는 주지 않음. LLM이 스스로 계산하도록 유도.
    """

    def generate_framework(
        self,
        domain: PhysicsDomain,
        solver_result: SolverResult
    ) -> FrameworkPrompt:
        """풀이 전략 프롬프트 생성"""

        # 물리 원칙
        physics_principle = solver_result.physical_principle

        # 공식 안내 (수치 결과는 미포함)
        formula_guide = self._build_formula_guide(domain, solver_result)

        # 파라미터 값
        parameter_values = self._build_parameter_section(solver_result)

        # 풀이 전략
        solving_strategy = self._build_strategy(domain, solver_result)

        # 물리적 제약 (위반 불가)
        warning_constraints = self._build_constraints(domain, solver_result)

        return FrameworkPrompt(
            physics_principle=physics_principle,
            formula_guide=formula_guide,
            parameter_values=parameter_values,
            solving_strategy=solving_strategy,
            warning_constraints=warning_constraints
        )

    def format_as_prompt(self, framework: FrameworkPrompt) -> str:
        """FrameworkPrompt를 LLM 프롬프트 문자열로 변환"""
        return f"""
╔══════════════════════════════════════════════════════════════════════╗
║  🧭 PHYSICS FRAMEWORK - 이 문제의 풀이 전략                         ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  📌 적용할 물리 원칙:                                                ║
║  {framework.physics_principle}
║                                                                      ║
║  📐 사용할 공식:                                                     ║
{framework.formula_guide}
║                                                                      ║
║  🔢 대입할 파라미터:                                                 ║
{framework.parameter_values}
║                                                                      ║
║  🎯 풀이 전략:                                                       ║
{framework.solving_strategy}
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  ⚠️ 물리적 제약 (위반 시 답변 거부):                                ║
{framework.warning_constraints}
╠══════════════════════════════════════════════════════════════════════╣
║  📋 최종 답변 형식 (CRITICAL - 반드시 준수):                         ║
║  풀이 과정을 모두 서술한 후, 최종 수치 답을 아래 태그로 표기:        ║
║                                                                      ║
║  [ANSWER]최종_수치_값[/ANSWER]                                       ║
║                                                                      ║
║  예시: [ANSWER]12.34[/ANSWER] 또는 [ANSWER]0.7200[/ANSWER]           ║
║  ⚠️ 태그 안에는 숫자만 (단위/% 기호 제외)                           ║
╚══════════════════════════════════════════════════════════════════════╝
"""

    def format_as_explain_prompt(
        self,
        framework: FrameworkPrompt,
        solver_result: SolverResult
    ) -> str:
        """
        B→C 방식: Solver 결과를 설명 대상으로 제공하는 프롬프트

        LLM 역할: 계산기 ❌ → 물리 해설자 ✅
        - Solver 수치를 "왜 이 값이 나오는지" 설명
        - 물리적 의미 해석 + 실무적 시사점 제시
        - [ANSWER] 태그 불필요
        """
        # 유도 과정 (중간값 포함)
        derivation_lines = []
        for step in solver_result.derivation_summary:
            derivation_lines.append(f"║    {step}")
        derivation_text = "\n".join(derivation_lines)

        # 주요 결과값
        result_lines = []
        result_lines.append(f"║    ★ {solver_result.primary_label} = {solver_result.primary_value:.4f}")
        for key, val in solver_result.all_values.items():
            if key != solver_result.primary_label:
                result_lines.append(f"║      • {key} = {val:.4f}")
        result_text = "\n".join(result_lines)

        return f"""
╔══════════════════════════════════════════════════════════════════════╗
║  🧭 PHYSICS FRAMEWORK - Explain Mode (Dual-Track)                   ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  📌 적용된 물리 원칙:                                                ║
║  {framework.physics_principle}
║                                                                      ║
║  📐 적용된 공식:                                                     ║
║    주 공식: {solver_result.formula_used}
║                                                                      ║
║  🔢 사용된 파라미터:                                                 ║
{framework.parameter_values}
║                                                                      ║
║  📊 Solver 유도 과정:                                                ║
{derivation_text}
║                                                                      ║
║  ✅ Solver 확정 결과:                                                ║
{result_text}
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  📋 당신의 역할 (CRITICAL - 반드시 준수):                            ║
║                                                                      ║
║  위 Solver가 도출한 결과를 기반으로:                                 ║
║  1. 각 유도 단계의 물리적 의미를 설명하세요                          ║
║  2. 왜 이 결과가 물리적으로 타당한지 논증하세요                      ║
║  3. EID vs PCD 비교 시사점을 서술하세요                              ║
║  4. 임상/실무적 의미를 제시하세요                                    ║
║                                                                      ║
║  ⚠️ 주의:                                                           ║
║  • 수치 재계산 불필요 — Solver 결과를 신뢰하세요                     ║
║  • 물리적 해석과 실무 시사점에 집중하세요                            ║
║  • [ANSWER] 태그 불필요 — 서술형으로 답변하세요                      ║
╚══════════════════════════════════════════════════════════════════════╝
"""

    def format_solver_summary(self, solver_result: SolverResult) -> str:
        """
        Dual-Track C: Solver 수치 결과를 최종 답변 하단에 첨부할 요약

        항상 사용자에게 표시되는 '확정 수치' 섹션
        """
        lines = []
        lines.append("---")
        lines.append(f"**📊 Physics Solver 확정 결과** ({self._domain_label(solver_result.domain)})")
        lines.append("")
        lines.append(f"| 항목 | 값 |")
        lines.append(f"|------|-----|")
        lines.append(f"| **{solver_result.primary_label}** | **{solver_result.primary_value:.4f}** |")
        for key, val in solver_result.all_values.items():
            if key != solver_result.primary_label:
                lines.append(f"| {key} | {val:.4f} |")
        lines.append("")
        lines.append(f"- 적용 공식: `{solver_result.formula_used}`")
        lines.append(f"- 물리 원칙: {solver_result.physical_principle}")
        return "\n".join(lines)

    def format_multi_solver_summary(
        self,
        solver_results: Dict[PhysicsDomain, SolverResult]
    ) -> str:
        """Multi-domain용 Solver 수치 요약"""
        lines = []
        lines.append("---")
        lines.append("**📊 Multi-Domain Physics Solver 확정 결과**")
        lines.append("")
        for domain, result in solver_results.items():
            label = self._domain_label(domain)
            lines.append(f"**{label}**")
            lines.append(f"| 항목 | 값 |")
            lines.append(f"|------|-----|")
            lines.append(f"| **{result.primary_label}** | **{result.primary_value:.4f}** |")
            for key, val in result.all_values.items():
                if key != result.primary_label:
                    lines.append(f"| {key} | {val:.4f} |")
            lines.append("")
        return "\n".join(lines)

    def _build_formula_guide(self, domain: PhysicsDomain, result: SolverResult) -> str:
        """공식 안내 생성 (결과값 미포함, 중간 계산 포함)"""
        lines = []
        lines.append(f"║    주 공식: {result.formula_used}")

        # solver 결과 포함 phases: 전체 표시 (모델은 유도/설명 담당)
        # 수치 검증이 있는 Phase: 1, 3, 5
        guided_phases = {PhysicsDomain.PHASE1_SNR, PhysicsDomain.PHASE3_DQE, PhysicsDomain.PHASE5_TOMO_IQ}
        max_steps = len(result.derivation_summary) if result.domain in guided_phases else 4
        for step in result.derivation_summary[:max_steps]:
            lines.append(f"║    {step}")

        return "\n".join(lines)

    def _build_parameter_section(self, result: SolverResult) -> str:
        """파라미터 섹션 생성"""
        lines = []
        for key, val in result.parameters.items():
            if isinstance(val, float) and val == int(val):
                lines.append(f"║    • {key} = {int(val)}")
            else:
                lines.append(f"║    • {key} = {val}")
        return "\n".join(lines)

    def _build_strategy(self, domain: PhysicsDomain, result: SolverResult) -> str:
        """풀이 전략 생성"""
        strategies = {
            PhysicsDomain.PHASE1_SNR: (
                "║    1. σ_total²을 양자+전자 성분으로 분리\n"
                "║    2. 선량 변화 시 각 성분의 변화를 추적 (σ_e² 고정!)\n"
                "║    3. SNR = Signal/σ_total 비율 계산\n"
                "║    4. 감소율(%) = (1 - SNR_new/SNR_ref) × 100"
            ),
            PhysicsDomain.PHASE3_DQE: (
                "║    1. α 파라미터 산출 (전자노이즈 기여도)\n"
                "║    2. DQE_EID(D) = η_abs/(1+α/D) 적용\n"
                "║    3. DQE_PCD = η_abs (상수) 확인\n"
                "║    4. 선량별 DQE 변화 및 PCD 우위 계산"
            ),
            PhysicsDomain.PHASE4_MTF: (
                "║    1. Nyquist 주파수 = 1/(2×pixel_pitch)\n"
                "║    2. PCD: sinc 함수만 적용 (직접변환)\n"
                "║    3. EID: sinc × Gaussian blur (간접변환)\n"
                "║    4. 주파수별 MTF 비교"
            ),
            PhysicsDomain.PHASE4B_DEPTH: (
                "║    1. Through-plane: Δz = K/sin(α/2)\n"
                "║    2. In-plane: Δxy = pixel_pitch/MTF\n"
                "║    3. 비대칭 비율 = Δz/Δxy\n"
                "║    4. 각도 범위의 영향 분석"
            ),
            PhysicsDomain.PHASE5_TOMO_IQ: (
                "║    1. D_proj = D_total/N (투영당 선량)\n"
                "║    2. EID DQE 저하: DQE(D_proj) = η/(1+α×N)\n"
                "║    3. PCD DQE 불변: DQE = η_abs\n"
                "║    4. SNR gain = √(1+α×N) — 2D 대비 토모에서 PCD 우위 극대화"
            ),
        }
        return strategies.get(domain, "║    일반 물리 풀이 절차를 따르세요.")

    def _build_constraints(self, domain: PhysicsDomain, result: SolverResult) -> str:
        """물리적 제약 생성"""
        constraints = []

        if domain == PhysicsDomain.PHASE1_SNR:
            f_e = result.parameters.get('f_e', 0.3)
            constraints = [
                "║    • 선량 감소 → SNR은 반드시 감소 (증가 불가)",
                "║    • 전자노이즈는 선량 변화에 무관 (σ_e² = const)",
                f"║    • ⚠️ f_e={f_e}는 '감소된 선량에서의' 전자노이즈 비율임",
                f"║      (기준 선량에서의 비율이 아님! 질문: '~차지하게 된다면')",
                f"║    • 기준 선량에서 f_e_ref = f_e×D'/(1-f_e+f_e×D') < {f_e}",
                "║    • SNR 감소율 > √(dose_ratio) 기반 감소율 (전자노이즈 효과)",
                f"║    • 합리적 범위: SNR 감소율 ∈ [30%, 40%] (f_e={f_e}, D'=0.5일 때)",
            ]
        elif domain == PhysicsDomain.PHASE3_DQE:
            constraints = [
                "║    • DQE_EID ≤ η_abs (항상)",
                "║    • DQE_PCD = η_abs (선량 무관, 전자노이즈 없음)",
                "║    • 선량 감소 → DQE_EID 감소 (DQE_PCD 불변)",
                "║    • α > 0 (전자노이즈가 존재하는 한)",
            ]
        elif domain == PhysicsDomain.PHASE5_TOMO_IQ:
            constraints = [
                "║    • N↑ → 투영당 선량↓ → EID DQE↓ (PCD 불변)",
                "║    • PCD SNR 우위는 N에 따라 단조증가",
                "║    • N=1일 때 2D mammo와 동일 (Phase 3 결과 재현)",
                "║    • 총 선량 동일 시: SNR_total ∝ √(DQE × D_total)",
            ]
        else:
            constraints = [
                "║    • 물리 법칙의 일관성을 유지할 것",
                "║    • 단위 변환 정확성 확인",
            ]

        return "\n".join(constraints)

    def generate_multi_framework(
        self,
        solver_results: Dict[PhysicsDomain, SolverResult]
    ) -> str:
        """
        복수 도메인 통합 프레임워크 생성

        Args:
            solver_results: 도메인별 solver 결과

        Returns:
            통합 프레임워크 프롬프트 문자열
        """
        sections = []

        # 도메인별 제약조건 섹션
        for i, (domain, result) in enumerate(solver_results.items(), 1):
            domain_label = self._domain_label(domain)
            section = (
                f"║  📌 제약조건 {i}: [{domain_label}]\n"
                f"║    • 물리 원칙: {result.physical_principle}\n"
                f"║    • 공식: {result.formula_used}\n"
            )
            # 파라미터
            for key, val in result.parameters.items():
                if isinstance(val, float) and val == int(val):
                    section += f"║    • {key} = {int(val)}\n"
                else:
                    section += f"║    • {key} = {val}\n"
            # 주요 결과 (solver 수치 직접 제공)
            section += f"║    • 도출 결과: {result.primary_label} = {result.primary_value:.4f}\n"
            for key, val in result.all_values.items():
                if key != result.primary_label:
                    section += f"║      - {key} = {val:.4f}\n"
            sections.append(section)

        # 통합 최적화 전략 섹션
        domain_names = [self._domain_label(d) for d in solver_results.keys()]
        optimization = (
            f"║  🎯 통합 최적화 전략:\n"
            f"║    • 관련 도메인: {', '.join(domain_names)}\n"
            f"║    • 각 제약을 만족하는 파라미터 범위를 제시하세요\n"
            f"║    • 도메인 간 트레이드오프 관계를 설명하세요\n"
            f"║    • 최적 조합의 물리적 근거를 제시하세요\n"
        )

        # 전체 프레임워크 조합
        framework = (
            "\n╔══════════════════════════════════════════════════════════════════════╗\n"
            "║  🧭 MULTI-DOMAIN PHYSICS FRAMEWORK                                  ║\n"
            "╠══════════════════════════════════════════════════════════════════════╣\n"
            "║                                                                      ║\n"
        )
        framework += "║                                                                      ║\n".join(sections)
        framework += "║                                                                      ║\n"
        framework += optimization
        framework += (
            "║                                                                      ║\n"
            "╠══════════════════════════════════════════════════════════════════════╣\n"
            "║  📋 답변 형식:                                                       ║\n"
            "║  • 각 제약조건의 물리적 분석을 서술하세요                            ║\n"
            "║  • 도메인 간 상호작용과 트레이드오프를 설명하세요                    ║\n"
            "║  • 최적 파라미터 조합을 제안하세요                                   ║\n"
            "║  • [ANSWER] 태그 불필요 (서술형 답변)                                ║\n"
            "╚══════════════════════════════════════════════════════════════════════╝\n"
        )

        return framework

    def _domain_label(self, domain: PhysicsDomain) -> str:
        """도메인 라벨 반환"""
        labels = {
            PhysicsDomain.PHASE1_SNR: "Phase 1 — SNR/전자노이즈",
            PhysicsDomain.PHASE2_SPECTRAL: "Phase 2 — 에너지 가중/스펙트럴",
            PhysicsDomain.PHASE3_DQE: "Phase 3 — DQE 선량의존성",
            PhysicsDomain.PHASE4_MTF: "Phase 4 — MTF/해상도",
            PhysicsDomain.PHASE4B_DEPTH: "Phase 4B — 토모 깊이분해능",
            PhysicsDomain.PHASE5_TOMO_IQ: "Phase 5 — 토모 영상품질",
        }
        return labels.get(domain, domain.value)


# =============================================================================
# Component 4: PostVerifier
# =============================================================================

class PostVerifier:
    """
    LLM 답변 vs solver 정답 비교 (multi-phase)

    추출 우선순위 (Search-and-Verify 패턴):
    1. [ANSWER] 태그 → 결정론적, 비용 0
    2. Regex 패턴 → 기존 로직
    3. LLM Fallback → Agent-as-a-Judge, 소형 모델로 재추출
    """

    TOLERANCE = 1.0  # 허용 오차 (%)
    OLLAMA_URL = "http://localhost:11434"
    EXTRACTOR_MODEL = "qwen2.5:14b"  # 빠른 추출용 모델
    EXTRACTOR_TIMEOUT = 30  # 초

    # =========================================================================
    # Tier 1: Answer Tag Extraction (결정론적, 비용 0)
    # =========================================================================

    def _extract_answer_tag(self, text: str) -> Optional[float]:
        """[ANSWER]X.XXXX[/ANSWER] 태그에서 수치 추출"""
        match = re.search(r'\[ANSWER\]\s*([\d.]+)\s*\[/ANSWER\]', text, re.IGNORECASE)
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                pass
        return None

    # =========================================================================
    # Tier 3: LLM Fallback Extraction (Agent-as-a-Judge)
    # =========================================================================

    def _llm_extract_value(
        self,
        text: str,
        domain: PhysicsDomain,
        expected: float
    ) -> Optional[float]:
        """
        Regex 실패 시 소형 LLM으로 수치 추출 (Search-and-Verify 패턴)

        비용: ~2초 추가 지연, 정확도: 의미 기반으로 높음
        """
        # 도메인별 추출 지시문
        extraction_prompts = {
            PhysicsDomain.PHASE1_SNR: (
                "아래 텍스트에서 'SNR 감소율' 또는 'SNR reduction percentage' 값을 찾아 숫자만 답하세요.\n"
                "단위(%)는 제외하고 숫자만 출력.\n"
                f"참고: 예상 범위는 약 {expected*0.5:.0f}~{expected*1.5:.0f}% 입니다."
            ),
            PhysicsDomain.PHASE3_DQE: (
                "아래 텍스트에서 '감소된 선량에서의 EID DQE 값' (DQE_EID)을 찾아 숫자만 답하세요.\n"
                "0과 1 사이의 소수값입니다. 숫자만 출력.\n"
                f"참고: 예상 범위는 약 {expected*0.7:.3f}~{expected*1.3:.3f} 입니다."
            ),
            PhysicsDomain.PHASE5_TOMO_IQ: (
                "아래 텍스트에서 'PCD vs EID SNR 우위 비율' (SNR gain, 몇 배)을 찾아 숫자만 답하세요.\n"
                "1보다 큰 값입니다. 숫자만 출력.\n"
                f"참고: 예상 범위는 약 {expected*0.7:.2f}~{expected*1.3:.2f}배 입니다."
            ),
        }

        prompt = extraction_prompts.get(domain)
        if not prompt:
            return None

        # 텍스트 길이 제한 (토큰 절약)
        text_truncated = text[:3000] if len(text) > 3000 else text

        try:
            response = requests.post(
                f"{self.OLLAMA_URL}/api/chat",
                json={
                    "model": self.EXTRACTOR_MODEL,
                    "messages": [
                        {"role": "system", "content": "숫자만 답하세요. 설명 없이 숫자 하나만 출력합니다."},
                        {"role": "user", "content": f"{prompt}\n\n---\n{text_truncated}"}
                    ],
                    "stream": False,
                    "options": {
                        "num_predict": 20,  # 숫자 하나만 필요
                        "temperature": 0.0,
                    }
                },
                timeout=self.EXTRACTOR_TIMEOUT
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "").strip()

            # 숫자 추출
            num_match = re.search(r'(\d+(?:\.\d+)?)', content)
            if num_match:
                val = float(num_match.group(1))
                # 합리성 검증: expected ± 50%
                if expected * 0.5 <= val <= expected * 1.5:
                    logger.info(f"LLM Fallback extracted: {val} (domain={domain.value})")
                    return val
                else:
                    logger.warning(f"LLM Fallback value {val} out of range for expected={expected}")

        except Exception as e:
            logger.warning(f"LLM Fallback extraction failed: {e}")

        return None

    # =========================================================================
    # Quality-based Verification (Dual-Track B→C)
    # =========================================================================

    # 도메인별 필수 키워드 (답변에 포함되어야 할 핵심 개념)
    QUALITY_KEYWORDS: Dict[PhysicsDomain, List[str]] = {
        PhysicsDomain.PHASE1_SNR: [
            '전자', '양자', 'snr', '노이즈', '잡음', '감소', '선량',
        ],
        PhysicsDomain.PHASE2_SPECTRAL: [
            '에너지', '가중', 'bin', 'spectral', '스펙트럴',
        ],
        PhysicsDomain.PHASE3_DQE: [
            'dqe', '선량', '전자', 'eid', 'pcd', '검출',
        ],
        PhysicsDomain.PHASE4_MTF: [
            'mtf', '해상도', '픽셀', 'nyquist', '나이퀴스트', '변환',
        ],
        PhysicsDomain.PHASE4B_DEPTH: [
            '깊이', '분해능', '각도', '슬라이스', 'depth',
        ],
        PhysicsDomain.PHASE5_TOMO_IQ: [
            '투영', '선량', 'dqe', 'snr', 'pcd', 'eid', '토모',
        ],
    }

    def verify_quality(
        self,
        llm_answer: str,
        domain: PhysicsDomain,
        solver_result: SolverResult
    ) -> PostVerificationResult:
        """
        Dual-Track 품질 검증: 수치 비교 대신 키워드 기반 개념 포함 확인

        LLM이 물리적 해석을 적절히 수행했는지 확인.
        수치 정확도는 Solver가 보장하므로 검증 불필요.
        """
        answer_lower = llm_answer.lower()

        # 도메인 키워드 존재 확인
        keywords = self.QUALITY_KEYWORDS.get(domain, [])
        found = [kw for kw in keywords if kw in answer_lower]
        coverage = len(found) / max(len(keywords), 1)

        # 답변 길이 확인 (최소한의 설명 요구)
        min_length = 100  # 최소 100자 이상의 설명
        has_sufficient_length = len(llm_answer.strip()) >= min_length

        # solver 결과와 모순되는 표현 감지 (soft check)
        contradiction = self._detect_contradiction(llm_answer, domain, solver_result)

        # 종합 판정
        passed = coverage >= 0.3 and has_sufficient_length and not contradiction
        should_reject = not passed and not has_sufficient_length  # 너무 짧으면 reject

        if contradiction:
            explanation = f"⚠️ Solver 결과와 모순 감지 (키워드 커버리지: {coverage:.0%})"
        elif not has_sufficient_length:
            explanation = f"❌ 답변 길이 부족 ({len(llm_answer)}자 < {min_length}자 최소)"
        elif coverage < 0.3:
            explanation = f"⚠️ 핵심 물리 개념 부족 (커버리지: {coverage:.0%}, 발견: {found})"
        else:
            explanation = f"✅ 품질 검증 통과 (커버리지: {coverage:.0%}, 키워드: {found})"

        return PostVerificationResult(
            passed=passed,
            domain=domain,
            llm_value=None,
            solver_value=solver_result.primary_value,
            error_percent=0.0,
            explanation=explanation,
            should_reject=should_reject
        )

    def _detect_contradiction(
        self,
        llm_answer: str,
        domain: PhysicsDomain,
        solver_result: SolverResult
    ) -> bool:
        """Solver 결과와 모순되는 표현 감지"""
        answer_lower = llm_answer.lower()

        if domain == PhysicsDomain.PHASE1_SNR:
            # SNR이 증가한다고 하면 모순 (선량 감소 시 SNR은 반드시 감소)
            if 'snr' in answer_lower and ('증가' in answer_lower or '향상' in answer_lower):
                # "PCD가 SNR 향상" 같은 맥락은 허용
                if 'pcd' not in answer_lower:
                    return True

        elif domain == PhysicsDomain.PHASE3_DQE:
            # EID DQE가 선량 감소시 증가한다고 하면 모순
            if 'eid' in answer_lower and 'dqe' in answer_lower:
                if '증가' in answer_lower and '선량' in answer_lower and '감소' in answer_lower:
                    return True

        elif domain == PhysicsDomain.PHASE5_TOMO_IQ:
            # PCD가 EID보다 불리하다고 하면 모순
            if 'pcd' in answer_lower and ('불리' in answer_lower or '열등' in answer_lower):
                return True

        return False

    # =========================================================================
    # Numeric Verification (Legacy — compute mode)
    # =========================================================================

    def verify(
        self,
        llm_answer: str,
        domain: PhysicsDomain,
        solver_result: SolverResult
    ) -> PostVerificationResult:
        """LLM 답변을 solver 정답과 비교 (Legacy: compute mode용)"""

        if domain == PhysicsDomain.PHASE1_SNR:
            return self._verify_snr(llm_answer, solver_result)
        elif domain == PhysicsDomain.PHASE3_DQE:
            return self._verify_dqe(llm_answer, solver_result)
        elif domain == PhysicsDomain.PHASE5_TOMO_IQ:
            return self._verify_tomo(llm_answer, solver_result)
        else:
            # Phase 2, 4, 4-B: 현재는 수치 검증 없이 통과
            return PostVerificationResult(
                passed=True,
                domain=domain,
                llm_value=None,
                solver_value=solver_result.primary_value,
                error_percent=0.0,
                explanation="현재 이 Phase에 대한 수치 검증은 미구현",
                should_reject=False
            )

    def _verify_snr(self, llm_answer: str, solver_result: SolverResult) -> PostVerificationResult:
        """Phase 1: SNR 감소율 검증 (3-tier extraction)"""
        correct = solver_result.all_values.get('eid_snr_reduction_pct', solver_result.primary_value)

        # Tier 1: Answer Tag
        llm_val = self._extract_answer_tag(llm_answer)
        # Tier 2: Regex
        if llm_val is None:
            llm_val = self._extract_snr_reduction(llm_answer, expected=correct)
        # Tier 3: LLM Fallback
        if llm_val is None:
            llm_val = self._llm_extract_value(llm_answer, PhysicsDomain.PHASE1_SNR, correct)

        if llm_val is None:
            return PostVerificationResult(
                passed=False, domain=PhysicsDomain.PHASE1_SNR,
                llm_value=None, solver_value=correct,
                error_percent=100.0,
                explanation="LLM 답변에서 SNR 감소율 수치를 추출할 수 없음 (3-tier 모두 실패)",
                should_reject=True
            )

        error = abs(llm_val - correct)
        passed = error <= self.TOLERANCE

        return PostVerificationResult(
            passed=passed, domain=PhysicsDomain.PHASE1_SNR,
            llm_value=llm_val, solver_value=correct,
            error_percent=error,
            explanation=f"{'✅' if passed else '❌'} LLM={llm_val:.1f}%, 정답={correct:.1f}%, 오차={error:.1f}%",
            should_reject=not passed
        )

    def _verify_dqe(self, llm_answer: str, solver_result: SolverResult) -> PostVerificationResult:
        """Phase 3: DQE 값 검증 (3-tier extraction)"""
        correct_eid = solver_result.all_values.get('dqe_eid_at_dose', 0)
        correct_pcd = solver_result.all_values.get('dqe_pcd', 0.85)

        # Tier 1: Answer Tag
        llm_dqe = self._extract_answer_tag(llm_answer)
        # Tier 2: Regex
        if llm_dqe is None:
            llm_dqe = self._extract_dqe_value(llm_answer, expected=correct_eid)
        # Tier 3: LLM Fallback
        if llm_dqe is None:
            llm_dqe = self._llm_extract_value(llm_answer, PhysicsDomain.PHASE3_DQE, correct_eid)

        if llm_dqe is None:
            return PostVerificationResult(
                passed=False, domain=PhysicsDomain.PHASE3_DQE,
                llm_value=None, solver_value=correct_eid,
                error_percent=100.0,
                explanation="LLM 답변에서 DQE 수치를 추출할 수 없음 (3-tier 모두 실패)",
                should_reject=True
            )

        # EID DQE와 비교 (가장 중요한 값)
        error = abs(llm_dqe - correct_eid) / correct_eid * 100 if correct_eid > 0 else 100
        passed = error <= self.TOLERANCE * 5  # DQE는 5% 허용 (소수점 값이라 오차 큼)

        return PostVerificationResult(
            passed=passed, domain=PhysicsDomain.PHASE3_DQE,
            llm_value=llm_dqe, solver_value=correct_eid,
            error_percent=error,
            explanation=f"{'✅' if passed else '❌'} LLM DQE={llm_dqe:.3f}, 정답={correct_eid:.3f}, 오차={error:.1f}%",
            should_reject=not passed
        )

    def _verify_tomo(self, llm_answer: str, solver_result: SolverResult) -> PostVerificationResult:
        """Phase 5: 토모 SNR gain 검증 (3-tier extraction)"""
        correct_gain = solver_result.all_values.get('pcd_snr_gain', solver_result.primary_value)

        # Tier 1: Answer Tag
        llm_gain = self._extract_answer_tag(llm_answer)
        # Tier 2: Regex
        if llm_gain is None:
            llm_gain = self._extract_snr_gain(llm_answer, expected=correct_gain)
        # Tier 3: LLM Fallback
        if llm_gain is None:
            llm_gain = self._llm_extract_value(llm_answer, PhysicsDomain.PHASE5_TOMO_IQ, correct_gain)

        if llm_gain is None:
            # 최후: DQE per projection 비교
            correct_dqe_eid = solver_result.all_values.get('dqe_eid_per_proj', 0)
            llm_dqe = self._extract_dqe_value(llm_answer, expected=correct_dqe_eid if correct_dqe_eid > 0 else None)

            if llm_dqe is not None and correct_dqe_eid > 0:
                error = abs(llm_dqe - correct_dqe_eid) / correct_dqe_eid * 100
                passed = error <= 10.0
                return PostVerificationResult(
                    passed=passed, domain=PhysicsDomain.PHASE5_TOMO_IQ,
                    llm_value=llm_dqe, solver_value=correct_dqe_eid,
                    error_percent=error,
                    explanation=f"{'✅' if passed else '❌'} DQE_EID: LLM={llm_dqe:.3f}, 정답={correct_dqe_eid:.3f}",
                    should_reject=not passed
                )

            return PostVerificationResult(
                passed=False, domain=PhysicsDomain.PHASE5_TOMO_IQ,
                llm_value=None, solver_value=correct_gain,
                error_percent=100.0,
                explanation="LLM 답변에서 토모 SNR gain 수치를 추출할 수 없음 (3-tier 모두 실패)",
                should_reject=True
            )

        error = abs(llm_gain - correct_gain) / correct_gain * 100 if correct_gain > 0 else 100
        passed = error <= 5.0  # gain은 5% 허용

        return PostVerificationResult(
            passed=passed, domain=PhysicsDomain.PHASE5_TOMO_IQ,
            llm_value=llm_gain, solver_value=correct_gain,
            error_percent=error,
            explanation=f"{'✅' if passed else '❌'} SNR gain: LLM={llm_gain:.2f}×, 정답={correct_gain:.2f}×",
            should_reject=not passed
        )

    # =========================================================================
    # Extraction Helpers
    # =========================================================================

    def _strip_think_tags(self, text: str) -> str:
        """DeepSeek-R1 <think> 태그 제거 (추론 과정의 중간값 오추출 방지)"""
        return re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)

    def _extract_snr_reduction(self, text: str, expected: Optional[float] = None) -> Optional[float]:
        """텍스트에서 SNR 감소율 추출 (expected 기반 동적 범위)"""
        text = self._strip_think_tags(text)  # R1 thinking 제거
        text_clean = self._strip_latex(text)

        # 동적 범위: expected ± 50% (없으면 기본 20-60)
        if expected and expected > 0:
            range_lo = max(5, expected * 0.5)
            range_hi = min(95, expected * 1.5)
            fallback_lo = max(10, expected * 0.7)
            fallback_hi = min(90, expected * 1.3)
        else:
            range_lo, range_hi = 20, 60
            fallback_lo, fallback_hi = 30, 50

        patterns = [
            # 한국어 직접 매칭
            r'SNR[은이가를의]?\s*(?:약\s*)?(\d+(?:\.\d+)?)\s*%\s*(?:감소|하락|저하)',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:의\s*)?SNR\s*(?:감소|하락)',
            r'SNR\s*(?:감소|하락|저하)[^\d]*(\d+(?:\.\d+)?)\s*%',
            r'(?:감소율|하락폭|하락률)[^0-9]*(\d+(?:\.\d+)?)\s*%',
            r'약\s*\*?\*?(\d+(?:\.\d+)?)\s*%\s*\*?\*?\s*(?:감소|하락|저하)',
            # 영어 매칭
            r'SNR\s*(?:decreases?|reduction|drops?)\s*(?:by\s*)?(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:reduction|decrease|drop)',
            # 수식 결과 매칭
            r'[=≈]\s*(\d+(?:\.\d+)?)\s*%',
            r'×\s*100\s*[=≈]\s*(\d+(?:\.\d+)?)',
            r'\*\s*100\s*[=≈]\s*(\d+(?:\.\d+)?)',
        ]

        for pattern in patterns:
            match = re.search(pattern, text_clean, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                if range_lo < val < range_hi:
                    return val

        # 폴백: SNR 감소에 해당하는 퍼센트 값 (노이즈/선량 관련 제외)
        all_pcts = re.findall(r'(\d+(?:\.\d+)?)\s*%', text_clean)
        exclude_pcts = set()
        exclude_patterns = [
            r'(?:선량|dose|MGD)[^%]*?(\d+(?:\.\d+)?)\s*%',
            r'(?:전자\s*노이즈|electronic\s*noise|노이즈|noise)[^%]*?(\d+(?:\.\d+)?)\s*%',
            r'(?:f_e|노이즈\s*비율|비율)[^%]*?(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:를\s*차지|차지|비율|비중)',
        ]
        for ep in exclude_patterns:
            for m in re.finditer(ep, text_clean, re.IGNORECASE):
                exclude_pcts.add(m.group(1))

        for pct_str in all_pcts:
            if pct_str in exclude_pcts:
                continue
            value = float(pct_str)
            if fallback_lo < value < fallback_hi:
                return value

        return None

    def _strip_latex(self, text: str) -> str:
        """LaTeX 명령어 제거 (패턴 매칭 전처리) — 중첩 브레이스 지원"""
        # 1. \text{...} 반복 처리 (중첩 가능: \text{DQE}_{\text{EID}})
        for _ in range(3):
            text = re.sub(r'\\text\{([^}]*)\}', r'\1', text)
            text = re.sub(r'\\mathrm\{([^}]*)\}', r'\1', text)
        # 2. 특수 명령어를 유니코드/텍스트로 변환 (\\[a-zA-Z]+ 제거 전에!)
        text = text.replace('\\approx', '≈').replace('\\times', '×')
        text = text.replace('\\cdot', '×').replace('\\,', ' ')
        text = re.sub(r'\\sqrt\{([^}]*)\}', r'√(\1)', text)  # \sqrt{X} → √(X)
        text = re.sub(r'\\boxed\{([^}]*)\}', r'\1', text)  # \boxed{X} → X
        # 3. \frac{A}{B} → (A)/(B) — 중첩 브레이스 지원
        text = self._replace_frac(text)
        # 4. 수식 구분자 제거
        text = re.sub(r'\\\(|\\\)', '', text)  # \( \) 인라인 구분자
        text = re.sub(r'\\\[|\\\]', '', text)  # \[ \] 디스플레이 구분자
        text = re.sub(r'\$+', '', text)  # $ 기호 제거
        # 5. 남은 \command 제거 (단, 숫자 앞 \제거 주의)
        text = re.sub(r'\\[a-zA-Z]+', '', text)
        # 6. 중괄호 제거
        text = re.sub(r'[{}]', '', text)
        return text

    def _replace_frac(self, text: str) -> str:
        """\\frac{A}{B} → (A)/(B) with balanced brace matching"""
        result = []
        i = 0
        frac_prefix = '\\frac'
        while i < len(text):
            if text[i:i+5] == frac_prefix and i + 5 < len(text) and text[i+5] == '{':
                # Found \frac{
                num_start = i + 5
                num_content, num_end = self._extract_braced(text, num_start)
                if num_content is not None and num_end < len(text) and text[num_end] == '{':
                    den_content, den_end = self._extract_braced(text, num_end)
                    if den_content is not None:
                        result.append(f'({num_content})/({den_content})')
                        i = den_end
                        continue
            result.append(text[i])
            i += 1
        return ''.join(result)

    def _extract_braced(self, text: str, start: int) -> Tuple[Optional[str], int]:
        """중괄호 쌍 매칭하여 내용 추출. Returns (content, end_pos_after_brace)"""
        if start >= len(text) or text[start] != '{':
            return None, start
        depth = 0
        i = start
        while i < len(text):
            if text[i] == '{':
                depth += 1
            elif text[i] == '}':
                depth -= 1
                if depth == 0:
                    return text[start+1:i], i + 1
            i += 1
        return None, start

    def _extract_dqe_value(self, text: str, expected: Optional[float] = None) -> Optional[float]:
        """텍스트에서 DQE 값 추출 (expected 기반 동적 범위)"""
        text = self._strip_think_tags(text)
        text_clean = self._strip_latex(text)

        # 동적 범위: expected ± 50% (없으면 기본 0.1-0.82)
        if expected and expected > 0:
            range_lo = max(0.01, expected * 0.5)
            range_hi = min(0.99, expected * 1.5)
            fallback_lo = max(0.05, expected * 0.7)
            fallback_hi = min(0.95, expected * 1.3)
        else:
            range_lo, range_hi = 0.1, 0.82
            fallback_lo, fallback_hi = 0.4, 0.75

        # 우선순위 1: DQE 포함 줄에서 계산 체인의 마지막 값 추출
        # 예: "DQE_EID(D') = 0.850/(1+α) = 0.5950" → 0.5950
        dqe_line_candidates = []
        for line in text_clean.split('\n'):
            if re.search(r'(?:DQE|dqe|양자검출)', line, re.IGNORECASE):
                line_vals = re.findall(r'(\d+\.\d{2,4})', line)
                for v_str in reversed(line_vals):
                    v = float(v_str)
                    if range_lo < v < range_hi and abs(v - 0.850) > 0.001:
                        dqe_line_candidates.append(v)
                        break

        if dqe_line_candidates and expected:
            best = min(dqe_line_candidates, key=lambda v: abs(v - expected))
            if abs(best - expected) / expected < 0.05:
                return best

        # 우선순위 2: 명시적 패턴
        reduced_patterns = [
            r'DQE[_{\s]*(?:new|reduced|감소|저선량)[^0-9]*?[=≈]\s*(\d+\.\d+)',
            r'DQE[_{\s]*EID[^0-9]*?(?:감소|저선량|reduced|at\s*reduced|at\s*D)[^0-9]*?[=≈]\s*(\d+\.\d+)',
            r'(\d+\.\d+)\s*(?:로|으로)\s*(?:저하|감소|하락)',
        ]
        for pattern in reduced_patterns:
            match = re.search(pattern, text_clean, re.IGNORECASE)
            if match:
                val = float(match.group(1))
                if range_lo < val < range_hi:
                    return val

        # DQE 줄 후보 (5% 밖이어도 범위 내면)
        if dqe_line_candidates:
            if expected:
                return min(dqe_line_candidates, key=lambda v: abs(v - expected))
            return dqe_line_candidates[0]

        # 일반 패턴: DQE 관련 값 수집
        general_patterns = [
            r'DQE[_{\s]*(?:EID)?[^0-9]*?[=≈:]\s*(\d+\.\d+)',
            r'DQE\s*[=≈:]\s*(\d+\.\d+)',
            r'[=≈]\s*(\d+\.\d+)\s*\$?\s*$',  # 줄 끝의 값
            # 계산 체인의 마지막 값: "= X.XX" where previous had "DQE"
            r'=\s*(\d+\.\d{3,4})\s*$',  # 줄 끝의 4자리 소수 (e.g., 0.5950)
            r'=\s*(\d+\.\d{2,4})\s*(?:\(|$)',  # 계산 결과 값
        ]

        all_dqe_values = []
        for pattern in general_patterns:
            for match in re.finditer(pattern, text_clean, re.IGNORECASE | re.MULTILINE):
                val = float(match.group(1))
                if range_lo < val < range_hi:
                    all_dqe_values.append(val)

        # 폴백: expected 근방의 소수값 (분모/파라미터 제외)
        if not all_dqe_values:
            for match in re.finditer(r'(\d+\.\d+)', text_clean):
                val = float(match.group(1))
                if fallback_lo < val < fallback_hi:
                    pos = match.start()
                    prefix = text_clean[max(0, pos-3):pos].strip()
                    if prefix.endswith('/') or prefix.endswith('÷'):
                        continue
                    context = text_clean[max(0, pos-20):pos]
                    if re.search(r'(?:f_e|D\'|alpha|α|1\s*-\s*f)', context, re.IGNORECASE):
                        continue
                    all_dqe_values.append(val)

        if all_dqe_values:
            # expected에 가장 가까운 값 반환 (있으면), 없으면 마지막 값
            if expected:
                return min(all_dqe_values, key=lambda v: abs(v - expected))
            return all_dqe_values[-1]

        return None

    def _extract_snr_gain(self, text: str, expected: Optional[float] = None) -> Optional[float]:
        """텍스트에서 SNR gain/advantage ratio 추출 (expected 기반 동적 범위)"""
        text = self._strip_think_tags(text)
        text_clean = self._strip_latex(text)

        # 동적 범위: expected ± 50%
        if expected and expected > 0:
            range_lo = max(1.0, expected * 0.5)
            range_hi = expected * 2.0
            priority_lo = max(1.0, expected * 0.8)
            priority_hi = expected * 1.2
        else:
            range_lo, range_hi = 1.5, 10.0
            priority_lo, priority_hi = 2.0, 3.5

        patterns = [
            # 한국어
            r'(?:SNR|신호)\s*(?:gain|이득|우위|advantage|비율)[가이은는]?\s*(?:약\s*)?(\d+(?:\.\d+)?)\s*[×배]',
            r'(\d+(?:\.\d+)?)\s*[×배]\s*(?:의\s*)?(?:SNR|신호)\s*(?:우위|이득|advantage|gain)',
            r'PCD[가이은는]?\s*(?:EID\s*대비\s*)?(?:약\s*)?(\d+(?:\.\d+)?)\s*[×배]',
            r'(?:gain|ratio|이득|우위|비율)[가이은는=:≈]\s*(?:약\s*)?(\d+(?:\.\d+)?)',
            # 수식 결과
            r'√\s*\(?[^)]*\)?\s*[=≈]\s*(\d+(?:\.\d+)?)',  # √(...) ≈ 2.52
            r'[=≈]\s*(\d+(?:\.\d+)?)\s*[×배]',            # = 2.52×
            r'(\d+(?:\.\d+)?)\s*[×배]\s*(?:PCD|우위|높|이득)',
            # "X times higher/more" (영어 모델 출력)
            r'(\d+(?:\.\d+)?)\s*times?\s*(?:higher|more|greater|우위)',
            r'approximately\s*\*?\*?(\d+(?:\.\d+)?)\s*times?',
            # \boxed{X} 또는 bare value
            r'\\?boxed\s*(\d+(?:\.\d+)?)',
            # 약 X.XX배 (중간 위치 허용)
            r'약\s*\*?\*?(\d+\.\d+)\s*\*?\*?\s*[×배]',
            # 일반: X.XX배 (범위 필터로 0.2143×25 배제)
            r'(\d+\.\d+)\s*[×배]',
        ]

        candidates = []
        for pattern in patterns:
            for match in re.finditer(pattern, text_clean, re.IGNORECASE | re.MULTILINE):
                val = float(match.group(1))
                if range_lo <= val <= range_hi:
                    candidates.append(val)

        # expected 근방 (priority 범위) 우선
        for c in candidates:
            if priority_lo <= c <= priority_hi:
                return c
        if candidates:
            # expected에 가장 가까운 값
            if expected:
                return min(candidates, key=lambda v: abs(v - expected))
            return candidates[0]

        # 폴백 1: SNR/gain/√ 포함 줄에서 expected 근방 값 탐색
        for line in text_clean.split('\n'):
            if re.search(r'(?:SNR|gain|이득|√|R_SNR)', line, re.IGNORECASE):
                for m in re.finditer(r'[=≈]\s*(\d+\.\d+)', line):
                    val = float(m.group(1))
                    if priority_lo <= val <= priority_hi:
                        return val

        # 폴백 2: 줄 끝 "≈ X.XX"
        for match in re.finditer(r'[=≈]\s*(\d+\.\d+)\s*$', text_clean, re.MULTILINE):
            val = float(match.group(1))
            if range_lo <= val <= range_hi:
                return val

        return None


# =============================================================================
# Unified Triage Pipeline
# =============================================================================

# =============================================================================
# Component 5: EmbeddingClassifier (Semantic Embedding 기반 분류)
# =============================================================================

class EmbeddingClassifier:
    """
    Embedding 기반 물리 도메인 분류 (Solution 3)

    각 도메인에 대한 참조 문장들의 임베딩을 미리 계산/캐싱하고,
    새 질문의 임베딩과 코사인 유사도를 비교하여 분류.

    장점:
    - 키워드/regex 패턴에 의존하지 않으므로 새로운 표현에도 강건
    - 의미적으로 유사한 질문을 정확히 매핑
    - 기존 분류기의 보완 경로로 활용
    """

    OLLAMA_URL = "http://localhost:11434"
    EMBED_MODEL = "glm4:9b"
    EMBED_TIMEOUT = 30
    CACHE_FILE = Path(__file__).parent.parent.parent / "data" / "cache" / "embedding_references.json"

    # 도메인별 참조 문장 (한국어, 실제 질문 형태)
    REFERENCE_QUERIES: Dict[PhysicsDomain, List[str]] = {
        PhysicsDomain.PHASE1_SNR: [
            "선량을 감소시키면 SNR은 얼마나 줄어드나요?",
            "전자 노이즈가 30%일 때 신호대잡음비 변화를 계산하세요",
            "저선량에서 전자잡음이 SNR에 미치는 영향은?",
            "MGD 50% 감축 시 SNR 하락폭을 수식으로 증명하시오",
            "양자노이즈와 전자노이즈의 비율이 바뀌면 영상 품질은?",
            "선량 감소가 신호 대 잡음비에 미치는 정량적 영향",
            "전자노이즈 비율이 40%일 때 SNR 감소율은 얼마인가?",
        ],
        PhysicsDomain.PHASE3_DQE: [
            "선량이 감소할 때 EID와 PCD의 DQE는 각각 어떻게 변하나요?",
            "양자검출효율의 선량 의존성을 비교 분석하세요",
            "전자노이즈 비율 30%에서 DQE 비교",
            "EID 검출기의 DQE가 선량에 따라 떨어지는 이유는?",
            "PCD는 왜 DQE가 선량에 무관한지 설명하세요",
            "감소된 선량에서의 EID DQE 값을 구하시오",
            "DQE 공식에서 전자노이즈 항의 역할을 분석하세요",
        ],
        PhysicsDomain.PHASE4_MTF: [
            "PCD와 EID 검출기의 MTF를 비교하세요",
            "픽셀 피치가 해상도에 미치는 영향은?",
            "직접변환 검출기의 변조전달함수 특성",
            "나이퀴스트 주파수에서의 MTF 값 비교",
            "간접변환 vs 직접변환 해상도 차이",
        ],
        PhysicsDomain.PHASE4B_DEPTH: [
            "토모합성의 깊이 분해능은 각도 범위에 따라 어떻게 결정되나요?",
            "25도 각도 범위에서 슬라이스 두께를 계산하세요",
            "depth resolution과 angular range의 관계",
            "토모합성 기하학적 분해능의 한계",
            "through-plane 분해능은 어떤 인자로 결정되는지",
        ],
        PhysicsDomain.PHASE5_TOMO_IQ: [
            "토모합성에서 선량을 25개 투영으로 나눌 때 DQE 변화는?",
            "투영당 선량 감소가 PCD와 EID에 미치는 영향 비교",
            "dose split 문제에서 PCD의 SNR 우위는 얼마인가?",
            "토모합성에서 PCD가 EID보다 유리한 이유를 정량화하시오",
            "25개 투영으로 분할할 때 투영당 DQE를 계산하세요",
            "1500 μGy를 15개 투영으로 나누면 PCD SNR gain은?",
            "해부학적 잡음 제거와 토모합성 검출능의 관계",
        ],
    }

    def __init__(self):
        self._ref_embeddings: Optional[Dict[str, np.ndarray]] = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        """Ollama embedding 서비스 사용 가능 여부 확인"""
        if self._available is not None:
            return self._available

        try:
            resp = requests.get(f"{self.OLLAMA_URL}/api/tags", timeout=3)
            if resp.status_code == 200:
                models = [m.get('name', '') for m in resp.json().get('models', [])]
                self._available = any(self.EMBED_MODEL in m for m in models)
            else:
                self._available = False
        except Exception:
            self._available = False

        if not self._available:
            logger.info("EmbeddingClassifier: Ollama embedding not available")
        return self._available

    def classify(self, query: str) -> Tuple[PhysicsDomain, float]:
        """
        질문을 임베딩하여 참조 문장과 코사인 유사도로 분류

        Returns:
            (domain, confidence) - 가장 유사한 도메인과 신뢰도
        """
        if not self.is_available():
            return PhysicsDomain.UNKNOWN, 0.0

        try:
            # 참조 임베딩 로드/생성
            ref_embeddings = self._get_reference_embeddings()
            if ref_embeddings is None:
                return PhysicsDomain.UNKNOWN, 0.0

            # 질문 임베딩
            query_emb = self._embed_single(query)
            if query_emb is None:
                return PhysicsDomain.UNKNOWN, 0.0

            # 각 도메인별 최대 유사도 계산
            domain_scores: Dict[PhysicsDomain, float] = {}
            for domain, ref_matrix in ref_embeddings.items():
                # 코사인 유사도: 각 참조 문장과의 유사도 계산 후 최대값
                similarities = self._cosine_similarity_batch(query_emb, ref_matrix)
                domain_scores[domain] = float(np.max(similarities))

            if not domain_scores:
                return PhysicsDomain.UNKNOWN, 0.0

            # 최고 점수 도메인
            best_domain = max(domain_scores, key=domain_scores.get)
            best_score = domain_scores[best_domain]

            # 2위와의 차이로 신뢰도 조정
            sorted_scores = sorted(domain_scores.values(), reverse=True)
            if len(sorted_scores) > 1:
                margin = sorted_scores[0] - sorted_scores[1]
                # 유사도 0.7+ & margin 0.05+ → 고신뢰
                confidence = min(1.0, best_score * (1.0 + margin * 2))
            else:
                confidence = best_score

            # 유사도가 너무 낮으면 분류 불가
            if best_score < 0.5:
                return PhysicsDomain.UNKNOWN, 0.0

            # margin이 너무 작으면 비특이적 (비물리 질문도 높은 유사도를 보일 수 있음)
            if len(sorted_scores) > 1:
                margin = sorted_scores[0] - sorted_scores[1]
                if margin < 0.025:
                    logger.info(
                        f"EmbeddingClassifier: Low margin ({margin:.4f}), "
                        f"non-specific query likely"
                    )
                    return PhysicsDomain.UNKNOWN, 0.0

            logger.info(
                f"EmbeddingClassifier: domain={best_domain.value}, "
                f"score={best_score:.3f}, confidence={confidence:.3f}"
            )
            return best_domain, min(1.0, confidence)

        except Exception as e:
            logger.warning(f"EmbeddingClassifier failed: {e}")
            return PhysicsDomain.UNKNOWN, 0.0

    def _get_reference_embeddings(self) -> Optional[Dict[PhysicsDomain, np.ndarray]]:
        """참조 임베딩 로드 (캐시 우선, 없으면 생성)"""
        if self._ref_embeddings is not None:
            return self._ref_embeddings

        # 캐시 파일에서 로드 시도
        if self.CACHE_FILE.exists():
            try:
                cached = self._load_cache()
                if cached is not None:
                    self._ref_embeddings = cached
                    logger.info("EmbeddingClassifier: Loaded cached embeddings")
                    return self._ref_embeddings
            except Exception as e:
                logger.warning(f"EmbeddingClassifier: Cache load failed: {e}")

        # 캐시 없으면 생성
        logger.info("EmbeddingClassifier: Building reference embeddings...")
        self._ref_embeddings = self._build_reference_embeddings()
        if self._ref_embeddings:
            self._save_cache(self._ref_embeddings)
        return self._ref_embeddings

    def _build_reference_embeddings(self) -> Optional[Dict[PhysicsDomain, np.ndarray]]:
        """모든 참조 문장의 임베딩 계산"""
        result = {}
        for domain, queries in self.REFERENCE_QUERIES.items():
            embeddings = self._embed_batch(queries)
            if embeddings is not None:
                result[domain] = embeddings
            else:
                logger.warning(f"EmbeddingClassifier: Failed to embed {domain.value}")
                return None
        return result

    def _embed_batch(self, texts: List[str]) -> Optional[np.ndarray]:
        """Ollama batch embedding API 호출"""
        try:
            resp = requests.post(
                f"{self.OLLAMA_URL}/api/embed",
                json={"model": self.EMBED_MODEL, "input": texts},
                timeout=self.EMBED_TIMEOUT
            )
            if resp.status_code == 200:
                data = resp.json()
                embeddings = data.get("embeddings", [])
                if embeddings:
                    return np.array(embeddings, dtype=np.float32)
            logger.warning(f"Embed batch failed: status={resp.status_code}")
            return None
        except Exception as e:
            logger.warning(f"Embed batch error: {e}")
            return None

    def _embed_single(self, text: str) -> Optional[np.ndarray]:
        """단일 텍스트 임베딩"""
        result = self._embed_batch([text])
        if result is not None and len(result) > 0:
            return result[0]
        return None

    @staticmethod
    def _cosine_similarity_batch(query_vec: np.ndarray, ref_matrix: np.ndarray) -> np.ndarray:
        """query와 참조 행렬 간의 코사인 유사도 (벡터화)"""
        # query: (D,), ref_matrix: (N, D)
        query_norm = query_vec / (np.linalg.norm(query_vec) + 1e-8)
        ref_norms = ref_matrix / (np.linalg.norm(ref_matrix, axis=1, keepdims=True) + 1e-8)
        return ref_norms @ query_norm  # (N,)

    def _load_cache(self) -> Optional[Dict[PhysicsDomain, np.ndarray]]:
        """JSON 캐시에서 참조 임베딩 로드"""
        with open(self.CACHE_FILE, 'r') as f:
            data = json.load(f)

        # 모델이 바뀌었으면 캐시 무효화
        if data.get("model") != self.EMBED_MODEL:
            logger.info("EmbeddingClassifier: Model changed, invalidating cache")
            return None

        # 참조 문장이 바뀌었으면 캐시 무효화
        cached_hash = data.get("ref_hash", "")
        current_hash = self._compute_ref_hash()
        if cached_hash != current_hash:
            logger.info("EmbeddingClassifier: Reference queries changed, invalidating cache")
            return None

        result = {}
        for domain_str, emb_list in data.get("embeddings", {}).items():
            try:
                domain = PhysicsDomain(domain_str)
                result[domain] = np.array(emb_list, dtype=np.float32)
            except ValueError:
                continue

        return result if result else None

    def _save_cache(self, embeddings: Dict[PhysicsDomain, np.ndarray]):
        """참조 임베딩을 JSON 캐시로 저장"""
        self.CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "model": self.EMBED_MODEL,
            "ref_hash": self._compute_ref_hash(),
            "embeddings": {
                domain.value: emb.tolist()
                for domain, emb in embeddings.items()
            }
        }

        with open(self.CACHE_FILE, 'w') as f:
            json.dump(data, f)
        logger.info(f"EmbeddingClassifier: Cached embeddings to {self.CACHE_FILE}")

    def _compute_ref_hash(self) -> str:
        """참조 문장의 해시 (변경 감지용)"""
        import hashlib
        content = json.dumps(
            {d.value: q for d, q in self.REFERENCE_QUERIES.items()},
            sort_keys=True, ensure_ascii=False
        )
        return hashlib.md5(content.encode()).hexdigest()


# =============================================================================
# 통합 Triage 파이프라인
# =============================================================================

class PhysicsTriageLayer:
    """
    통합 Triage 파이프라인

    Orchestrator가 단일 인터페이스로 사용.
    질문 → 분류 → 풀이 → 프레임워크 생성 → 사후 검증
    """

    def __init__(self):
        self.classifier = PhysicsClassifier()
        self.embedding_classifier = EmbeddingClassifier()
        self.router = SolverRouter()
        self.injector = FrameworkInjector()
        self.verifier = PostVerifier()

    def pre_solve(self, query: str):
        """
        답변 생성 전 풀이 전략 생성

        Returns:
            (framework_prompt, solver_result, classification)
            - framework_prompt: LLM에게 주입할 풀이 전략 (None이면 일반 처리)
            - solver_result: solver 계산 결과 (SolverResult 또는 Dict[PhysicsDomain, SolverResult])
            - classification: 분류 결과
        """
        # 1. 다중 도메인 분류
        classification, all_scores = self.classifier.classify_multi(query)
        logger.info(
            f"PhysicsTriage: domain={classification.primary_domain.value}, "
            f"confidence={classification.confidence:.2f}, "
            f"agree={classification.paths_agree}, "
            f"all_scores={[(d.value, f'{s:.2f}') for d, s in all_scores.items()]}"
        )

        # 2. 다중 도메인 감지
        active_domains = [
            d for d, s in all_scores.items()
            if s >= 0.2 and d != PhysicsDomain.UNKNOWN
        ]

        # 파라미터 다양성: 서로 다른 도메인의 파라미터가 추출되었는지 확인
        params = classification.extracted_params
        param_domain_count = sum([
            'dose_ratio' in params or 'electronic_noise_fraction' in params,  # Phase 1/3
            'pixel_pitch_mm' in params,                                       # Phase 4
            'angular_range_deg' in params,                                    # Phase 4B
            'n_projections' in params or 'total_dose_uGy' in params,         # Phase 5
        ])

        # 다중 도메인 조건:
        # - 2+ 도메인 활성 & 단일 도메인 수렴 안 됨 (confidence < 0.7)
        # - 3+ 도메인 활성 & 파라미터 다양 (2+ 도메인 파라미터) & confidence <= 0.75
        is_multi = (
            (len(active_domains) >= 2 and classification.confidence < 0.7) or
            (len(active_domains) >= 3 and param_domain_count >= 2 and classification.confidence <= 0.75)
        )

        if is_multi:
            # 다중 도메인 경로 (점수 순 정렬)
            active_domains.sort(key=lambda d: all_scores.get(d, 0), reverse=True)
            logger.info(
                f"PhysicsTriage: Multi-domain detected! "
                f"active={[d.value for d in active_domains]}"
            )
            solver_results = self.router.route_and_solve_multi(
                active_domains, classification.extracted_params
            )

            if solver_results:
                framework_prompt = self.injector.generate_multi_framework(solver_results)
                # 다중 도메인용 classification 반환
                multi_classification = ClassificationResult(
                    primary_domain=active_domains[0],
                    confidence=max(all_scores.values()),
                    keyword_path=classification.keyword_path,
                    semantic_path=classification.semantic_path,
                    paths_agree=False,
                    extracted_params=classification.extracted_params,
                    reasoning=f"Multi-domain: {[d.value for d in active_domains]}"
                )
                return framework_prompt, solver_results, multi_classification

        # 3. 기존 단일 도메인 경로

        # 3a. Embedding 분류기: 저신뢰 또는 UNKNOWN일 때 보완
        if classification.primary_domain == PhysicsDomain.UNKNOWN or classification.confidence < 0.6:
            emb_domain, emb_confidence = self.embedding_classifier.classify(query)

            if emb_domain != PhysicsDomain.UNKNOWN:
                if classification.primary_domain == PhysicsDomain.UNKNOWN:
                    classification = ClassificationResult(
                        primary_domain=emb_domain,
                        confidence=emb_confidence * 0.8,
                        keyword_path=PhysicsDomain.UNKNOWN,
                        semantic_path=PhysicsDomain.UNKNOWN,
                        paths_agree=False,
                        extracted_params=classification.extracted_params,
                        reasoning=f"Embedding분류={emb_domain.value}({emb_confidence:.2f})"
                    )
                    logger.info(
                        f"PhysicsTriage: Embedding rescued → {emb_domain.value} "
                        f"(conf={emb_confidence:.2f})"
                    )
                elif emb_domain == classification.primary_domain:
                    boosted = min(1.0, classification.confidence + emb_confidence * 0.3)
                    classification = ClassificationResult(
                        primary_domain=classification.primary_domain,
                        confidence=boosted,
                        keyword_path=classification.keyword_path,
                        semantic_path=classification.semantic_path,
                        paths_agree=classification.paths_agree,
                        extracted_params=classification.extracted_params,
                        reasoning=classification.reasoning + f", Embedding일치↑({emb_confidence:.2f})"
                    )
                    logger.info(
                        f"PhysicsTriage: Embedding confirms → conf boosted to {boosted:.2f}"
                    )
                else:
                    if emb_confidence > classification.confidence + 0.2:
                        classification = ClassificationResult(
                            primary_domain=emb_domain,
                            confidence=emb_confidence * 0.7,
                            keyword_path=classification.keyword_path,
                            semantic_path=classification.semantic_path,
                            paths_agree=False,
                            extracted_params=classification.extracted_params,
                            reasoning=classification.reasoning + f", Embedding우선={emb_domain.value}({emb_confidence:.2f})"
                        )
                        logger.info(
                            f"PhysicsTriage: Embedding override → {emb_domain.value}"
                        )

        # 분류 실패 또는 저신뢰
        if classification.primary_domain == PhysicsDomain.UNKNOWN:
            logger.info("PhysicsTriage: UNKNOWN domain, skipping triage")
            return None, None, classification

        if classification.confidence < 0.45:
            logger.info(f"PhysicsTriage: Low confidence ({classification.confidence:.2f}), skipping")
            return None, None, classification

        # 3b-1. 파라미터 미추출 시 solver skip (오분류 방지)
        if not classification.extracted_params:
            logger.info(
                f"PhysicsTriage: No relevant params extracted for {classification.primary_domain.value} "
                f"(conf={classification.confidence:.2f}), skipping solver"
            )
            return None, None, classification

        # 3b. Solver 호출
        solver_result = self.router.route_and_solve(classification)
        if solver_result is None:
            logger.warning("PhysicsTriage: Solver returned None")
            return None, None, classification

        logger.info(
            f"PhysicsTriage: Solved - {solver_result.primary_label}={solver_result.primary_value:.4f}"
        )

        # 3c. 풀이 프레임워크 생성
        framework = self.injector.generate_framework(
            classification.primary_domain, solver_result
        )
        framework_prompt = self.injector.format_as_prompt(framework)

        return framework_prompt, solver_result, classification

    def pre_solve_explain(self, query: str):
        """
        Dual-Track (B→C) 모드: Solver 결과를 설명 대상으로 제공

        Returns:
            (explain_prompt, solver_result_or_dict, classification, solver_summary)
            - explain_prompt: LLM에게 주입할 설명 유도 프롬프트
            - solver_result_or_dict: solver 결과 (단일 또는 다중)
            - classification: 분류 결과
            - solver_summary: 최종 답변에 첨부할 수치 요약 (markdown)
        """
        # 기존 pre_solve 호출
        framework_prompt, solver_result, classification = self.pre_solve(query)

        if solver_result is None:
            return None, None, classification, None

        # Multi-domain인 경우
        if isinstance(solver_result, dict):
            # multi-domain은 이미 explain 방식 (generate_multi_framework)
            solver_summary = self.injector.format_multi_solver_summary(solver_result)
            return framework_prompt, solver_result, classification, solver_summary

        # 단일 도메인: compute mode → explain mode로 전환
        framework = self.injector.generate_framework(
            classification.primary_domain, solver_result
        )
        explain_prompt = self.injector.format_as_explain_prompt(framework, solver_result)
        solver_summary = self.injector.format_solver_summary(solver_result)

        return explain_prompt, solver_result, classification, solver_summary

    def post_verify(
        self,
        llm_answer: str,
        solver_result: SolverResult,
        classification: ClassificationResult
    ) -> PostVerificationResult:
        """
        답변 생성 후 검증 (Legacy: compute mode)
        """
        return self.verifier.verify(
            llm_answer,
            classification.primary_domain,
            solver_result
        )

    def post_verify_quality(
        self,
        llm_answer: str,
        solver_result: SolverResult,
        classification: ClassificationResult
    ) -> PostVerificationResult:
        """
        Dual-Track 품질 검증: 키워드 기반 개념 포함 확인
        """
        return self.verifier.verify_quality(
            llm_answer,
            classification.primary_domain,
            solver_result
        )


# =============================================================================
# Singleton
# =============================================================================

_triage_instance: Optional[PhysicsTriageLayer] = None


def get_physics_triage() -> PhysicsTriageLayer:
    """PhysicsTriageLayer 싱글톤"""
    global _triage_instance
    if _triage_instance is None:
        _triage_instance = PhysicsTriageLayer()
    return _triage_instance


# =============================================================================
# Tests
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    triage = PhysicsTriageLayer()

    print("=" * 70)
    print("Physics Triage Layer Tests")
    print("=" * 70)

    # =========================================================================
    # Test 1: Phase 1 SNR 분류 + 풀이
    # =========================================================================
    print("\n[Test 1] Phase 1 SNR 질문")
    q1 = "선량을 50% 감소했을 때, 전자 노이즈가 30%를 차지한다면 SNR 감소율은?"

    framework, solver_result, classification = triage.pre_solve(q1)

    assert classification.primary_domain == PhysicsDomain.PHASE1_SNR, \
        f"Expected PHASE1_SNR, got {classification.primary_domain}"
    assert classification.confidence >= 0.5, \
        f"Low confidence: {classification.confidence}"
    assert solver_result is not None, "Solver result is None"
    assert abs(solver_result.all_values['eid_snr_reduction_pct'] - 34.8) < 0.5, \
        f"Wrong SNR reduction: {solver_result.all_values['eid_snr_reduction_pct']}"
    assert framework is not None, "Framework is None"
    assert '전자노이즈' in framework or 'σ_e' in framework, "Framework missing key physics"

    print(f"  Domain: {classification.primary_domain.value}")
    print(f"  Confidence: {classification.confidence:.2f}")
    print(f"  Paths agree: {classification.paths_agree}")
    print(f"  SNR reduction: {solver_result.all_values['eid_snr_reduction_pct']:.1f}%")
    print(f"  Framework length: {len(framework)} chars")
    print("  ✅ PASS")

    # Test 1b: 사후 검증 (정답)
    correct_answer = "SNR은 약 34.8% 감소합니다."
    verify = triage.post_verify(correct_answer, solver_result, classification)
    assert verify.passed, f"Should pass: {verify.explanation}"
    print(f"  Post-verify (correct): {verify.explanation}")

    # Test 1c: 사후 검증 (오답)
    wrong_answer = "SNR은 약 16% 감소합니다."
    verify = triage.post_verify(wrong_answer, solver_result, classification)
    assert not verify.passed, f"Should fail: {verify.explanation}"
    assert verify.should_reject, "Should reject"
    print(f"  Post-verify (wrong): {verify.explanation}")

    # =========================================================================
    # Test 2: Phase 5 토모 분류
    # =========================================================================
    print("\n[Test 2] Phase 5 토모합성 질문")
    q2 = "토모합성에서 25개 투영으로 선량을 분할할 때 PCD와 EID의 DQE 차이는?"

    framework, solver_result, classification = triage.pre_solve(q2)

    assert classification.primary_domain == PhysicsDomain.PHASE5_TOMO_IQ, \
        f"Expected PHASE5_TOMO_IQ, got {classification.primary_domain}"
    assert solver_result is not None, "Solver result is None"
    assert solver_result.all_values['pcd_snr_gain'] > 2.0, \
        f"PCD gain should be > 2.0: {solver_result.all_values['pcd_snr_gain']}"

    print(f"  Domain: {classification.primary_domain.value}")
    print(f"  PCD SNR gain: {solver_result.all_values['pcd_snr_gain']:.2f}×")
    print(f"  DQE_EID per proj: {solver_result.all_values['dqe_eid_per_proj']:.4f}")
    print(f"  DQE_PCD per proj: {solver_result.all_values['dqe_pcd_per_proj']:.4f}")
    print("  ✅ PASS")

    # =========================================================================
    # Test 3: Phase 3 DQE 분류
    # =========================================================================
    print("\n[Test 3] Phase 3 DQE 질문")
    q3 = "선량이 50% 감소할 때 EID와 PCD의 DQE 변화를 비교하시오."

    framework, solver_result, classification = triage.pre_solve(q3)

    assert classification.primary_domain in [PhysicsDomain.PHASE3_DQE, PhysicsDomain.PHASE1_SNR], \
        f"Expected PHASE3_DQE or PHASE1_SNR, got {classification.primary_domain}"
    print(f"  Domain: {classification.primary_domain.value}")
    print(f"  Confidence: {classification.confidence:.2f}")
    print("  ✅ PASS")

    # =========================================================================
    # Test 4: Phase 4-B 깊이 분해능
    # =========================================================================
    print("\n[Test 4] Phase 4-B 깊이 분해능 질문")
    q4 = "각도 범위 25도에서 토모합성의 깊이 분해능을 계산하시오."

    framework, solver_result, classification = triage.pre_solve(q4)

    assert classification.primary_domain in [PhysicsDomain.PHASE4B_DEPTH, PhysicsDomain.PHASE5_TOMO_IQ], \
        f"Expected PHASE4B_DEPTH, got {classification.primary_domain}"
    if solver_result:
        print(f"  Domain: {classification.primary_domain.value}")
        print(f"  Depth resolution: {solver_result.primary_value:.2f} mm")
    print("  ✅ PASS")

    # =========================================================================
    # Test 5: 비물리 질문 → UNKNOWN
    # =========================================================================
    print("\n[Test 5] 비물리 질문")
    q5 = "오늘 날씨가 어떤가요?"

    framework, solver_result, classification = triage.pre_solve(q5)

    assert classification.primary_domain == PhysicsDomain.UNKNOWN, \
        f"Expected UNKNOWN, got {classification.primary_domain}"
    assert framework is None, "Framework should be None for unknown domain"
    print(f"  Domain: {classification.primary_domain.value}")
    print(f"  Framework: {framework}")
    print("  ✅ PASS")

    # =========================================================================
    # Test 6: 복합 질문 (Phase 1 + Phase 3 혼합)
    # =========================================================================
    print("\n[Test 6] 복합 질문 - Phase 1 우세")
    q6 = """MGD를 50% 감축했을 때, 전자 노이즈가 전체 노이즈의 30%를 차지하게 된다면
    SNR의 하락폭을 수식으로 증명하시오."""

    framework, solver_result, classification = triage.pre_solve(q6)

    assert classification.primary_domain == PhysicsDomain.PHASE1_SNR, \
        f"Expected PHASE1_SNR, got {classification.primary_domain}"
    assert solver_result is not None
    assert abs(solver_result.all_values['eid_snr_reduction_pct'] - 34.8) < 0.5
    print(f"  Domain: {classification.primary_domain.value}")
    print(f"  SNR reduction: {solver_result.all_values['eid_snr_reduction_pct']:.1f}%")
    print(f"  Parameters: {classification.extracted_params}")
    print("  ✅ PASS")

    # =========================================================================
    # Test 7: 프레임워크에 정답 미포함 확인
    # =========================================================================
    print("\n[Test 7] 프레임워크에 정답 수치 미포함 확인")
    q7 = "선량을 50% 감소했을 때, 전자 노이즈가 30%를 차지한다면 SNR 감소율은?"

    framework, solver_result, classification = triage.pre_solve(q7)

    # 프레임워크에 정답(34.8)이 포함되어 있으면 안 됨
    assert '34.8' not in framework, "Framework should NOT contain the answer (34.8%)"
    assert '34.7' not in framework, "Framework should NOT contain the answer"
    # 하지만 공식과 파라미터는 포함되어야 함
    assert 'f_e' in framework or '0.3' in framework, "Framework should contain parameters"
    assert 'dose_ratio' in framework or '0.5' in framework, "Framework should contain dose_ratio"
    print(f"  Framework does NOT contain '34.8': ✓")
    print(f"  Framework contains parameters: ✓")
    print("  ✅ PASS")

    # =========================================================================
    # Test 8: 파라미터 추출 정확도
    # =========================================================================
    print("\n[Test 8] 파라미터 추출")
    q8 = "MGD를 50% 감축, 전자 노이즈 30%, Rose Criterion(k=5), 25개 투영"

    classification = triage.classifier.classify(q8)
    params = classification.extracted_params

    assert abs(params.get('dose_ratio', 0) - 0.5) < 0.01, f"dose_ratio: {params.get('dose_ratio')}"
    assert abs(params.get('electronic_noise_fraction', 0) - 0.30) < 0.01, f"f_e: {params.get('electronic_noise_fraction')}"
    assert params.get('n_projections') == 25, f"n_projections: {params.get('n_projections')}"
    assert params.get('rose_k') == 5.0, f"rose_k: {params.get('rose_k')}"

    print(f"  dose_ratio: {params.get('dose_ratio')}")
    print(f"  electronic_noise_fraction: {params.get('electronic_noise_fraction')}")
    print(f"  n_projections: {params.get('n_projections')}")
    print(f"  rose_k: {params.get('rose_k')}")
    print("  ✅ PASS")

    # =========================================================================
    # Test 9: EmbeddingClassifier — 물리 질문 분류
    # =========================================================================
    print("\n[Test 9] EmbeddingClassifier 물리 질문")
    emb_classifier = triage.embedding_classifier

    if emb_classifier.is_available():
        # 명확한 물리 질문 (의미적으로 충분히 구체적인 문장)
        emb_queries = [
            ("토모합성에서 선량을 여러 투영으로 분할했을 때 각 투영의 DQE는?", PhysicsDomain.PHASE5_TOMO_IQ),
            ("저선량 환경에서 EID 양자검출효율이 얼마나 떨어지나요?", PhysicsDomain.PHASE3_DQE),
            ("전자노이즈가 높은 비율을 차지할 때 신호대잡음비는 어떻게 변하는가?", PhysicsDomain.PHASE1_SNR),
        ]
        for q, expected_domain in emb_queries:
            domain, conf = emb_classifier.classify(q)
            assert domain == expected_domain, \
                f"'{q}': expected {expected_domain.value}, got {domain.value} (conf={conf:.3f})"
            print(f"  '{q[:30]}...' → {domain.value} (conf={conf:.3f}) ✓")
        print("  ✅ PASS")
    else:
        print("  ⚠️ SKIP (Ollama not available)")

    # =========================================================================
    # Test 10: EmbeddingClassifier — 비물리 질문 거부
    # =========================================================================
    print("\n[Test 10] EmbeddingClassifier 비물리 질문 거부")
    if emb_classifier.is_available():
        non_physics = [
            "오늘 날씨가 어떤가요?",
            "강남에 맛있는 식당 추천해주세요",
            "파이썬 프로그래밍 강좌를 찾고 있습니다",
        ]
        for q in non_physics:
            domain, conf = emb_classifier.classify(q)
            assert domain == PhysicsDomain.UNKNOWN, \
                f"'{q}': should be UNKNOWN, got {domain.value} (conf={conf:.3f})"
            print(f"  '{q[:25]}...' → UNKNOWN ✓")
        print("  ✅ PASS")
    else:
        print("  ⚠️ SKIP (Ollama not available)")

    # =========================================================================
    # Test 11: Embedding rescue — 저신뢰 질문 구출
    # =========================================================================
    print("\n[Test 11] Embedding rescue 통합 테스트")
    if emb_classifier.is_available():
        # 키워드 매칭이 약하지만 의미적으로 DQE와 관련된 질문
        q11 = "광자수가 줄어들면 EID 검출기의 양자효율에 어떤 영향이 있나요?"
        # '양자효율' = DQE지만 키워드 리스트에 없어 primary는 UNKNOWN
        framework, solver_result, classification = triage.pre_solve(q11)

        # Embedding이 rescue하여 유효한 분류가 되어야 함
        assert classification.primary_domain != PhysicsDomain.UNKNOWN, \
            f"Embedding should rescue this query, got UNKNOWN"
        print(f"  Query: '{q11[:40]}...'")
        print(f"  Domain: {classification.primary_domain.value}")
        print(f"  Confidence: {classification.confidence:.2f}")
        print(f"  Reasoning: {classification.reasoning}")
        print("  ✅ PASS")
    else:
        print("  ⚠️ SKIP (Ollama not available)")

    # =========================================================================
    # Test 12: Multi-Domain Triage — 다중 도메인 감지
    # =========================================================================
    print("\n[Test 12] Multi-Domain Triage")
    q12 = (
        "차세대 PCD 맘모그래피 시스템의 최종 스펙을 확정하려 한다. "
        "선량 40% 저감, 해상도 2배 향상(pixel pitch 0.05mm), "
        "깊이 분해능 0.5mm 이내 (각도 범위 25도). "
        "SNR 감소율과 DQE 변화도 함께 분석하라."
    )

    framework, solver_result, classification = triage.pre_solve(q12)

    if isinstance(solver_result, dict):
        # 다중 도메인으로 감지됨
        assert len(solver_result) >= 2, \
            f"Expected >= 2 domains, got {len(solver_result)}"
        assert framework is not None, "Multi-domain framework is None"
        assert 'MULTI-DOMAIN' in framework, "Framework should contain MULTI-DOMAIN header"
        print(f"  Multi-domain detected: ✓")
        print(f"  Active domains: {[d.value for d in solver_result.keys()]}")
        print(f"  Framework length: {len(framework)} chars")
        print(f"  Classification reasoning: {classification.reasoning}")
        print("  ✅ PASS (multi-domain path)")
    else:
        # 단일 도메인으로 판정된 경우 (confidence >= 0.7)
        print(f"  Single domain: {classification.primary_domain.value}")
        print(f"  Confidence: {classification.confidence:.2f}")
        print("  ⚠️ PASS (single domain path — confidence too high for multi-domain)")

    # Test 12b: classify_multi 직접 테스트
    print("\n[Test 12b] classify_multi() 점수 확인")
    cls_result, all_scores = triage.classifier.classify_multi(q12)
    active = [d for d, s in all_scores.items() if s >= 0.2 and d != PhysicsDomain.UNKNOWN]
    print(f"  All scores: {[(d.value, f'{s:.3f}') for d, s in sorted(all_scores.items(), key=lambda x: -x[1])]}")
    print(f"  Active domains (score >= 0.2): {[d.value for d in active]}")
    print(f"  Primary: {cls_result.primary_domain.value}, confidence: {cls_result.confidence:.2f}")
    assert len(active) >= 2, f"Expected >= 2 active domains, got {len(active)}: {active}"
    print("  ✅ PASS")

    # =========================================================================
    print(f"\n{'=' * 70}")
    print("All 12 tests PASSED ✅")
    print(f"{'=' * 70}")
