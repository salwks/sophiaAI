"""
핵심 물리 지식 (Core Physics Knowledge)
========================================
DeepSeek이 반드시 이해해야 하는 필수 물리 개념들

이 모듈의 내용은 모든 LLM 호출 시 시스템 프롬프트에 포함됩니다.

Phase 2: GOLDEN_FORMULAS - 수식 할루시네이션 방지를 위한 표준 수식 사전

===========================================================================
DEPRECATION NOTICE (Phase 7.19)
===========================================================================
이 모듈은 더 이상 사용되지 않습니다.
대신 다음을 사용하세요:
- data/knowledge/physics/core_physics.json (지식 소스)
- src/prompts/unified_builder.py (프롬프트 빌더)
- src/knowledge/manager.py:KnowledgeManager (지식 관리)

마이그레이션:
    # 기존 방식 (deprecated)
    from src.knowledge.core_physics import get_core_physics_prompt
    prompt = get_core_physics_prompt()

    # 새 방식 (recommended)
    from src.prompts.unified_builder import UnifiedPromptBuilder
    from src.knowledge.manager import get_knowledge_manager
    km = get_knowledge_manager()
    builder = UnifiedPromptBuilder(km)
    axioms = builder.get_axioms()
    system_prompt = builder.build_system_prompt()

이 파일은 하위 호환성을 위해 유지됩니다.
===========================================================================
"""

import warnings
from typing import Dict, List, Any


# =============================================================================
# GOLDEN_FORMULAS: 유방영상의학 표준 수식 사전
# =============================================================================
# LLM이 수식을 생성할 때 반드시 이 사전에 있는 수식만 사용하도록 강제합니다.
# 수식 무결성 검증기(FormulaChecker)가 이 사전을 기준으로 검증합니다.

GOLDEN_FORMULAS: Dict[str, Dict[str, Any]] = {
    # =========================================================================
    # MGD (Mean Glandular Dose) 관련 수식
    # =========================================================================
    "MGD_2D": {
        "id": "MGD_2D",
        "name": "Mean Glandular Dose (2D Mammography)",
        "formula_latex": r"MGD = K \cdot g \cdot c \cdot s",
        "formula_unicode": "MGD = K × g × c × s",
        "variables": {
            "MGD": {"description": "Mean Glandular Dose", "unit": "mGy"},
            "K": {"description": "Incident Air Kerma at breast surface", "unit": "mGy"},
            "g": {"description": "Conversion factor (breast thickness dependent)", "unit": "dimensionless"},
            "c": {"description": "Glandularity correction factor", "unit": "dimensionless"},
            "s": {"description": "Spectrum correction factor (target/filter)", "unit": "dimensionless"},
        },
        "source": "Dance DR, et al. Phys Med Biol. 2000;45:3225-3240",
        "doi": "10.1088/0031-9155/45/11/308",
        "notes": "European/UK/IAEA dosimetry protocol",
        "aliases": ["ESAK_to_MGD", "Dance_formula"],
    },

    "MGD_DBT_T": {
        "id": "MGD_DBT_T",
        "name": "Mean Glandular Dose (DBT with T-factor)",
        "formula_latex": r"MGD_{total} = K_{total} \cdot g \cdot c \cdot s \cdot T",
        "formula_unicode": "MGD_total = K_total × g × c × s × T",
        "variables": {
            "MGD_total": {"description": "Total Mean Glandular Dose for DBT scan", "unit": "mGy"},
            "K_total": {"description": "Total Incident Air Kerma", "unit": "mGy"},
            "g": {"description": "Conversion factor", "unit": "dimensionless"},
            "c": {"description": "Glandularity correction factor", "unit": "dimensionless"},
            "s": {"description": "Spectrum correction factor", "unit": "dimensionless"},
            "T": {"description": "T-factor (integrated projection series factor)", "unit": "dimensionless"},
        },
        "source": "Dance DR, et al. Phys Med Biol. 2011;56:453-471",
        "doi": "10.1088/0031-9155/56/2/011",
        "notes": "T-factor integrates all projections in DBT",
        "aliases": ["DBT_dose", "Tomosynthesis_MGD"],
    },

    "MGD_DBT_t": {
        "id": "MGD_DBT_t",
        "name": "Single Projection MGD (DBT with t-factor)",
        "formula_latex": r"MGD_i = K_i \cdot g \cdot c \cdot s \cdot t(\theta_i)",
        "formula_unicode": "MGD_i = K_i × g × c × s × t(θ_i)",
        "variables": {
            "MGD_i": {"description": "MGD for single projection at angle θ_i", "unit": "mGy"},
            "K_i": {"description": "Air Kerma for projection i", "unit": "mGy"},
            "t(θ_i)": {"description": "t-factor at angle θ_i (ratio to 0°)", "unit": "dimensionless"},
        },
        "source": "Dance DR, et al. Phys Med Biol. 2011;56:453-471",
        "doi": "10.1088/0031-9155/56/2/011",
        "notes": "t(0°) = 1.0 by definition",
    },

    "T_FACTOR": {
        "id": "T_FACTOR",
        "name": "T-factor Calculation",
        "formula_latex": r"T = \sum_{i=1}^{N} \alpha_i \cdot t(\theta_i)",
        "formula_unicode": "T = Σ αᵢ × t(θᵢ)",
        "variables": {
            "T": {"description": "Integrated T-factor for full scan", "unit": "dimensionless"},
            "α_i": {"description": "mAs fraction for projection i", "unit": "dimensionless"},
            "t(θ_i)": {"description": "t-factor at angle θ_i", "unit": "dimensionless"},
            "N": {"description": "Total number of projections", "unit": "count"},
        },
        "source": "Dance DR, et al. Phys Med Biol. 2011;56:453-471",
        "doi": "10.1088/0031-9155/56/2/011",
        "notes": "If uniform mAs: T = (1/N) × Σ t(θᵢ)",
    },

    # =========================================================================
    # 양자 노이즈 및 SNR 관련 수식
    # =========================================================================
    "QUANTUM_NOISE": {
        "id": "QUANTUM_NOISE",
        "name": "Quantum Noise (Poisson Statistics)",
        "formula_latex": r"\sigma = \sqrt{N}",
        "formula_unicode": "σ = √N",
        "variables": {
            "σ": {"description": "Standard deviation (noise)", "unit": "photons"},
            "N": {"description": "Number of detected photons", "unit": "photons"},
        },
        "source": "Basic X-ray physics (Poisson distribution)",
        "notes": "Fundamental statistical property of X-ray detection",
    },

    "SNR_QUANTUM": {
        "id": "SNR_QUANTUM",
        "name": "Signal-to-Noise Ratio (Quantum Limited)",
        "formula_latex": r"SNR = \frac{N}{\sigma} = \frac{N}{\sqrt{N}} = \sqrt{N}",
        "formula_unicode": "SNR = N/σ = N/√N = √N",
        "variables": {
            "SNR": {"description": "Signal-to-Noise Ratio", "unit": "dimensionless"},
            "N": {"description": "Number of detected photons", "unit": "photons"},
        },
        "source": "Basic X-ray physics",
        "notes": "Higher dose → more photons → higher SNR → lower noise",
        "critical_relationship": "Dose ↑ → SNR ↑ → Noise ↓",
    },

    "DOSE_SNR_RELATION": {
        "id": "DOSE_SNR_RELATION",
        "name": "Dose-SNR Relationship",
        "formula_latex": r"SNR \propto \sqrt{Dose}",
        "formula_unicode": "SNR ∝ √Dose",
        "variables": {
            "SNR": {"description": "Signal-to-Noise Ratio", "unit": "dimensionless"},
            "Dose": {"description": "Radiation dose", "unit": "mGy"},
        },
        "source": "Radiological physics fundamentals",
        "notes": "To double SNR, quadruple the dose",
    },

    # =========================================================================
    # HVL (Half Value Layer) 관련 수식
    # =========================================================================
    "HVL": {
        "id": "HVL",
        "name": "Half Value Layer",
        "formula_latex": r"HVL = \frac{0.693}{\mu} = \frac{\ln(2)}{\mu}",
        "formula_unicode": "HVL = 0.693/μ = ln(2)/μ",
        "variables": {
            "HVL": {"description": "Half Value Layer", "unit": "mm Al"},
            "μ": {"description": "Linear attenuation coefficient", "unit": "mm⁻¹"},
        },
        "source": "X-ray physics fundamentals",
        "notes": "Thickness of material to reduce intensity by 50%",
    },

    "ATTENUATION": {
        "id": "ATTENUATION",
        "name": "Exponential Attenuation",
        "formula_latex": r"I = I_0 \cdot e^{-\mu x}",
        "formula_unicode": "I = I₀ × e^(-μx)",
        "variables": {
            "I": {"description": "Transmitted intensity", "unit": "arbitrary"},
            "I_0": {"description": "Incident intensity", "unit": "arbitrary"},
            "μ": {"description": "Linear attenuation coefficient", "unit": "cm⁻¹"},
            "x": {"description": "Material thickness", "unit": "cm"},
        },
        "source": "Beer-Lambert Law",
        "notes": "Fundamental X-ray attenuation law",
    },

    # =========================================================================
    # Contrast & Resolution 관련 수식
    # =========================================================================
    "CONTRAST": {
        "id": "CONTRAST",
        "name": "Subject Contrast",
        "formula_latex": r"C = \frac{I_1 - I_2}{I_1 + I_2}",
        "formula_unicode": "C = (I₁ - I₂)/(I₁ + I₂)",
        "variables": {
            "C": {"description": "Contrast", "unit": "dimensionless"},
            "I_1": {"description": "Intensity from region 1", "unit": "arbitrary"},
            "I_2": {"description": "Intensity from region 2", "unit": "arbitrary"},
        },
        "source": "Basic imaging physics",
        "notes": "Also known as Michelson contrast",
    },

    "CNR": {
        "id": "CNR",
        "name": "Contrast-to-Noise Ratio",
        "formula_latex": r"CNR = \frac{|S_1 - S_2|}{\sigma}",
        "formula_unicode": "CNR = |S₁ - S₂|/σ",
        "variables": {
            "CNR": {"description": "Contrast-to-Noise Ratio", "unit": "dimensionless"},
            "S_1": {"description": "Signal from region 1", "unit": "arbitrary"},
            "S_2": {"description": "Signal from region 2", "unit": "arbitrary"},
            "σ": {"description": "Noise (standard deviation)", "unit": "arbitrary"},
        },
        "source": "Medical imaging quality metrics",
        "notes": "Key metric for lesion detectability",
    },

    "DQE": {
        "id": "DQE",
        "name": "Detective Quantum Efficiency",
        "formula_latex": r"DQE = \frac{SNR_{out}^2}{SNR_{in}^2}",
        "formula_unicode": "DQE = SNR_out² / SNR_in²",
        "variables": {
            "DQE": {"description": "Detective Quantum Efficiency", "unit": "dimensionless (0-1)"},
            "SNR_out": {"description": "Output SNR of detector", "unit": "dimensionless"},
            "SNR_in": {"description": "Input SNR (ideal)", "unit": "dimensionless"},
        },
        "source": "Detector performance metrics",
        "notes": "Measures detector efficiency; ideal detector has DQE = 1",
    },

    # =========================================================================
    # AEC (Automatic Exposure Control) 관련 수식
    # =========================================================================
    "MAS_CALCULATION": {
        "id": "MAS_CALCULATION",
        "name": "mAs Calculation",
        "formula_latex": r"mAs = mA \cdot t",
        "formula_unicode": "mAs = mA × t",
        "variables": {
            "mAs": {"description": "Tube current-time product", "unit": "mAs"},
            "mA": {"description": "Tube current", "unit": "mA"},
            "t": {"description": "Exposure time", "unit": "seconds"},
        },
        "source": "Basic radiography",
        "notes": "Primary factor controlling X-ray quantity",
    },

    # =========================================================================
    # Diagnostic Performance Metrics
    # =========================================================================
    "SENSITIVITY": {
        "id": "SENSITIVITY",
        "name": "Sensitivity (True Positive Rate)",
        "formula_latex": r"Sensitivity = \frac{TP}{TP + FN}",
        "formula_unicode": "Sensitivity = TP/(TP + FN)",
        "variables": {
            "TP": {"description": "True Positives", "unit": "count"},
            "FN": {"description": "False Negatives", "unit": "count"},
        },
        "source": "Diagnostic test statistics",
        "notes": "Ability to correctly identify positives",
        "aliases": ["TPR", "Recall"],
    },

    "SPECIFICITY": {
        "id": "SPECIFICITY",
        "name": "Specificity (True Negative Rate)",
        "formula_latex": r"Specificity = \frac{TN}{TN + FP}",
        "formula_unicode": "Specificity = TN/(TN + FP)",
        "variables": {
            "TN": {"description": "True Negatives", "unit": "count"},
            "FP": {"description": "False Positives", "unit": "count"},
        },
        "source": "Diagnostic test statistics",
        "notes": "Ability to correctly identify negatives",
        "aliases": ["TNR"],
    },

    "PPV": {
        "id": "PPV",
        "name": "Positive Predictive Value",
        "formula_latex": r"PPV = \frac{TP}{TP + FP}",
        "formula_unicode": "PPV = TP/(TP + FP)",
        "variables": {
            "TP": {"description": "True Positives", "unit": "count"},
            "FP": {"description": "False Positives", "unit": "count"},
        },
        "source": "Diagnostic test statistics",
        "notes": "Probability that positive finding is truly positive",
        "aliases": ["Precision"],
    },

    "NPV": {
        "id": "NPV",
        "name": "Negative Predictive Value",
        "formula_latex": r"NPV = \frac{TN}{TN + FN}",
        "formula_unicode": "NPV = TN/(TN + FN)",
        "variables": {
            "TN": {"description": "True Negatives", "unit": "count"},
            "FN": {"description": "False Negatives", "unit": "count"},
        },
        "source": "Diagnostic test statistics",
        "notes": "Probability that negative finding is truly negative",
    },

    "ACCURACY": {
        "id": "ACCURACY",
        "name": "Diagnostic Accuracy",
        "formula_latex": r"Accuracy = \frac{TP + TN}{TP + TN + FP + FN}",
        "formula_unicode": "Accuracy = (TP + TN)/(TP + TN + FP + FN)",
        "variables": {
            "TP": {"description": "True Positives", "unit": "count"},
            "TN": {"description": "True Negatives", "unit": "count"},
            "FP": {"description": "False Positives", "unit": "count"},
            "FN": {"description": "False Negatives", "unit": "count"},
        },
        "source": "Diagnostic test statistics",
    },
}


def get_formula(formula_id: str) -> Dict[str, Any]:
    """
    특정 수식 정보 조회

    DEPRECATED (Phase 7.19):
        이 함수는 더 이상 사용되지 않습니다.
        대신 KnowledgeManager를 통해 core_physics.json에서 조회하세요.

    Args:
        formula_id: 수식 ID (예: "MGD_2D", "SNR_QUANTUM")

    Returns:
        수식 정보 딕셔너리
    """
    warnings.warn(
        "get_formula() is deprecated. "
        "Use KnowledgeManager.get_knowledge_by_id('core_physics') instead.",
        DeprecationWarning,
        stacklevel=2
    )
    # 직접 ID 매칭
    if formula_id.upper() in GOLDEN_FORMULAS:
        return GOLDEN_FORMULAS[formula_id.upper()]

    # Alias 검색
    for fid, formula in GOLDEN_FORMULAS.items():
        if formula_id in formula.get("aliases", []):
            return formula

    return {}


def get_all_formula_ids() -> List[str]:
    """모든 수식 ID 목록 반환"""
    return list(GOLDEN_FORMULAS.keys())


def get_formula_variables() -> Dict[str, List[str]]:
    """
    모든 수식의 변수 목록 반환
    수식 검증 시 허용된 변수인지 확인하는 데 사용
    """
    all_vars = {}
    for fid, formula in GOLDEN_FORMULAS.items():
        all_vars[fid] = list(formula.get("variables", {}).keys())
    return all_vars


def get_formulas_by_category(category: str) -> Dict[str, Dict[str, Any]]:
    """
    카테고리별 수식 필터링

    Args:
        category: "MGD", "SNR", "HVL", "DIAGNOSTIC" 등

    Returns:
        해당 카테고리의 수식들
    """
    category_upper = category.upper()
    return {
        fid: formula
        for fid, formula in GOLDEN_FORMULAS.items()
        if category_upper in fid or category_upper in formula.get("name", "").upper()
    }


def format_formula_for_prompt(formula_id: str) -> str:
    """
    LLM 프롬프트에 삽입할 수식 포맷팅

    Args:
        formula_id: 수식 ID

    Returns:
        프롬프트용 포맷팅된 문자열
    """
    formula = get_formula(formula_id)
    if not formula:
        return f"[Formula '{formula_id}' not found]"

    lines = [
        f"### {formula['name']}",
        f"**Formula**: {formula['formula_unicode']}",
        f"**LaTeX**: `{formula['formula_latex']}`",
        "",
        "**Variables**:",
    ]

    for var, info in formula.get("variables", {}).items():
        lines.append(f"- {var}: {info['description']} ({info.get('unit', 'N/A')})")

    if formula.get("source"):
        lines.append(f"\n**Source**: {formula['source']}")

    if formula.get("notes"):
        lines.append(f"**Note**: {formula['notes']}")

    return "\n".join(lines)

# ============================================
# Dance et al. 2011 - MGD for Breast Tomosynthesis
# ============================================
MGD_TOMOSYNTHESIS_KNOWLEDGE = """
## Mean Glandular Dose (MGD) for Breast Tomosynthesis

### 출처
- **논문**: Dance DR, Young KC, van Engen RE. "Estimation of mean glandular dose for breast tomosynthesis: factors for use with the UK, European and IAEA breast dosimetry protocols"
- **저널**: Physics in Medicine and Biology, 2011;56:453-471
- **DOI**: 10.1088/0031-9155/56/2/011
- **핵심 페이지**: 8-9페이지 (t-factor와 T-factor 유도)

### 핵심 개념

#### 1. t-factor (소문자) - 단일 투영 인자
- **물리적 정의**: 0도(직사) 대비 특정 각도(θ)에서의 선량 비율
- **수식**:
  ```
  t(θ) = D(θ) / D(0)
  ```
  - D(θ): 각도 θ에서의 MGD
  - D(0): 0도(직사)에서의 MGD
- **용도**: 특정 각도(θ)에서 촬영된 **단일 영상**의 MGD 계산
- **MGD 공식**:
  ```
  MGD_projection = K × g × c × s × t(θ)
  ```
- **특성**:
  - 투사 각도(θ)와 유방 두께에 따라 변화
  - 유방 glandularity에는 거의 의존하지 않음
  - 0°에서 30°까지 각도가 증가할수록 t-factor 감소

#### 2. T-factor (대문자) - 전체 시리즈 통합 인자
- **물리적 정의**: 전체 노출 시리즈의 통합 선량 보정 계수
- **핵심 유도 공식** (논문 8-9페이지):
  ```
  T = Σ αᵢ × t(θᵢ)
  ```
  - αᵢ: 전체 tube loading(mAs) 중 i번째 각도에 할당된 비율
  - t(θᵢ): i번째 각도의 t-factor
  - **만약 모든 각도에서 동일한 mAs를 사용한다면**:
    ```
    T = (1/N) × Σ t(θᵢ)   (N = 총 projection 수)
    ```
- **용도**: 전체 DBT 검사(여러 projection)의 **총 MGD** 계산
- **MGD 공식**:
  ```
  MGD_total = K_total × g × c × s × T
  ```
- **대표값**:
  - Full-field 조사 geometry: T ≈ 0.93-1.00
  - Sectra 시스템 (narrow-beam): T = 0.76-0.98 (유방 두께에 따라)

#### 3. t-factor → T-factor 연결 (AEC 설계 핵심)
**왜 중요한가?**: AEC 알고리즘 설계 시, 각 각도별 mAs 배분(αᵢ)이 총 MGD에 직접 영향을 미침.

```
[설계 로직]
1. 각 각도 θᵢ에서의 t(θᵢ)를 테이블에서 조회
2. 각 각도에 할당할 mAs 비율 αᵢ 결정
3. T = Σ αᵢ × t(θᵢ) 계산
4. MGD_total = K_total × g × c × s × T
```

#### 4. 유방 두께별 t-factor 대표값 (Table 6, 50% glandularity, W/Al)
| 유방 두께 | t(0°) | t(10°) | t(20°) | t(30°) |
|-----------|-------|--------|--------|--------|
| 3 cm      | 1.000 | 0.975  | 0.906  | 0.803  |
| 5 cm      | 1.000 | 0.978  | 0.919  | 0.832  |
| 6.5 cm    | 1.000 | 0.980  | 0.929  | 0.852  |
| 8 cm      | 1.000 | 0.982  | 0.937  | 0.868  |

#### 5. 관련 기존 계수들
- **g-factor**: 입사 공기 kerma → MGD 변환 (유방 두께 의존)
- **c-factor**: glandularity 보정 (지방/유선 비율)
- **s-factor**: X선 스펙트럼 보정 (target/filter 조합)

### 물리적 원리

#### MGD 계산 공식 (2D Mammography)
```
MGD = K × g × c × s
```
- K: 유방 표면 입사 공기 kerma (mGy)
- g: 유방 두께에 따른 변환 계수
- c: glandularity 보정 계수
- s: 스펙트럼 보정 계수 (Mo/Mo, Mo/Rh, W/Rh 등)

#### MGD 계산 공식 (Tomosynthesis) - 방법 1: T-factor 사용
```
MGD_total = K_total × g × c × s × T
```

#### MGD 계산 공식 (Tomosynthesis) - 방법 2: 개별 합산
```
MGD_i = K_i × g × c × s × t(θ_i)
MGD_total = Σ MGD_i
```

### 실무 적용 예시

**예제**: 6.5cm 두께 유방, 15개 projection (±25° 범위), 균등 mAs 배분
1. 각도별 t-factor 조회: t(0°)=1.0, t(10°)=0.98, t(20°)=0.929, t(25°)≈0.89
2. T = (1/15) × [t(-25°) + t(-20°) + ... + t(0°) + ... + t(25°)]
3. T ≈ 0.94 (대략적 계산)
4. MGD_total = K_total × g × c × s × 0.94

### 임상적 중요성
1. **선량 최적화**: DBT가 2D 대비 어느 정도 추가 선량이 필요한지 정량화
2. **프로토콜 표준화**: UK, European, IAEA 프로토콜과 호환
3. **장비 비교**: 서로 다른 DBT 시스템 간 선량 비교 가능
4. **AEC 알고리즘 설계**: 각도별 mAs 배분 전략 수립

### Monte Carlo 시뮬레이션 조건
- 유방 두께: 20-110 mm
- 투사 각도: 0°-30°
- Target/Filter: Mo/Mo, Mo/Rh, Rh/Rh, W/Rh, W/Ag, W/Al
- Glandularity: 0.1%, 25%, 50%, 75%, 100%
"""

# ============================================
# 양자 노이즈와 선량 관계
# ============================================
QUANTUM_NOISE_KNOWLEDGE = """
## 양자 노이즈 (Quantum Mottle/Noise)

### 물리적 정의
- X선 광자의 **포아송 분포**에 따른 통계적 변동
- σ = √N (N = 검출된 광자 수)
- **SNR = N/√N = √N** → 광자 수가 많을수록 SNR 향상

### 선량과의 관계
```
선량 ↑ → 광자 수 ↑ → 노이즈 ↓ → SNR ↑
선량 ↓ → 광자 수 ↓ → 노이즈 ↑ → SNR ↓
```

### 산란선(Scatter)과의 차이
| 구분 | 양자 노이즈 | 산란선 |
|------|------------|--------|
| 원인 | 광자 통계적 변동 | X선 방향 변화 |
| 분포 | 랜덤, 포아송 | 공간적 패턴 |
| 제거 | 선량 증가, 디노이징 | 그리드, 에어갭 |
"""

# ============================================
# 필터와 MTF 관계
# ============================================
FILTER_MTF_KNOWLEDGE = """
## 디노이징 필터와 MTF (공간 해상도)

### 가우시안 필터 (Gaussian Filter)
- **MTF 손실**: 에지 블러링으로 인해 MTF 감소
- **미세석회화 검출**: 부적합 (에지 손상)
- **사용 금지**: 고해상도가 필요한 mammography에서

### 에지 보존 필터 (Edge-Preserving Filters)
1. **Bilateral Filter**: 공간적 + 밝기 기반 가중치
2. **Anisotropic Diffusion**: 에지 방향으로만 확산
3. **Non-local Means (NLM)**: 유사 패치 기반

### 딥러닝 디노이징
- **장점**: 학습을 통해 MTF 보존 가능
- **단점**: 학습 데이터 품질에 의존
- **주의**: 미세석회화 artifact 생성 가능성

### MTF 보존 원칙
```
MTF 보존 = 고주파 성분 유지 = 에지 보존 = 미세석회화 검출 가능
MTF 손실 = 고주파 손실 = 에지 블러 = 미세석회화 놓침
```
"""

# ============================================
# Target/Filter 조합과 빔 에너지
# ============================================
TARGET_FILTER_KNOWLEDGE = """
## Mammography Target/Filter 조합

### K-edge 에너지 (keV)
| Target/Filter | K-edge | 특성 |
|---------------|--------|------|
| Mo/Mo | Mo: 20.0 | Soft beam, 얇은 유방 |
| Mo/Rh | Rh: 23.2 | 중간 |
| Rh/Rh | Rh: 23.2 | 중간-Hard |
| W/Rh | Rh: 23.2 | Hard beam |
| W/Ag | Ag: 25.5 | **가장 Hard**, 두꺼운 유방 |

### 유방 두께별 권장
- **얇은 유방 (≤3cm)**: Mo/Mo, Mo/Rh (Soft beam → 대조도↑)
- **중간 유방 (3-5cm)**: Mo/Rh, Rh/Rh
- **두꺼운 유방 (≥6cm)**: W/Rh, W/Ag (Hard beam → 투과력↑)

### 중요 원칙
```
K-edge ↑ → 평균 에너지 ↑ → 투과력 ↑ → Hard beam
K-edge ↓ → 평균 에너지 ↓ → 대조도 ↑ → Soft beam
```

### 흔한 오류
- ❌ "W/Ag가 W/Rh보다 Soft하다" → Ag K-edge(25.5) > Rh K-edge(23.2)
- ❌ "두꺼운 유방에 Soft beam 사용" → 투과력 부족으로 노출 부족
"""


# =============================================================================
# CONSTITUTIONAL AXIOMS (Layer 1: 물리 헌법)
# =============================================================================
# 이 공리들은 모든 LLM 응답에서 절대 위반할 수 없는 물리 법칙입니다.
# MammoPhysicsSolver (Layer 2)가 이 공리를 기반으로 결정론적 검증을 수행합니다.

CONSTITUTIONAL_AXIOMS = """
╔══════════════════════════════════════════════════════════════════════╗
║  🔒 PHYSICS CONSTITUTIONAL LAWS (위반 시 답변 자동 거부)              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Law 1: Signal ∝ Dose                                                ║
║    → 신호는 선량에 선형 비례 (절대값)                                 ║
║    → S_new = (D_new/D_0) × S_0                                       ║
║                                                                      ║
║  Law 2: σ_quantum² ∝ Dose                                            ║
║    → 양자 노이즈 분산은 선량에 비례 (포아송 통계)                     ║
║    → σ_q_new² = (D_new/D_0) × σ_q0²                                  ║
║    → ⚠️ 흔한 오류: σ_q² ∝ 1/Dose (이것은 상대 노이즈 모델, 틀림)     ║
║                                                                      ║
║  Law 3: σ_electronic² = constant                                      ║
║    → 전자 노이즈는 선량과 무관한 하드웨어 특성                        ║
║    → 선량 감소 시 전자노이즈의 상대적 기여도 증가                     ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  핵심 유도:                                                          ║
║  SNR = S / √(σ_q² + σ_e²)                                            ║
║  SNR_new/SNR_0 = √(D × (1 - f_e × (1-D)))                            ║
║    (f_e = 선량 변화 후 전자노이즈 분산 비율)                          ║
║                                                                      ║
║  PCD: σ_e = 0 (에너지 문턱치로 제거)                                 ║
║    → SNR_PCD_new/SNR_PCD_0 = √D                                      ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  ⚠️ 이 법칙을 위반하는 수치를 제시하면 답변이 자동 거부됩니다.        ║
║  Python Deterministic Solver가 모든 수치를 1% 오차 이내로 검증합니다. ║
╚══════════════════════════════════════════════════════════════════════╝
"""


# =============================================================================
# PHASE 2 CONSTITUTIONAL AXIOMS: PCD Spectral Contrast
# =============================================================================
# Phase 2 핵심 주제: PCD의 에너지 분해 능력을 통한 대조도(CNR) 향상
# EID 대비 PCD의 구조적 우위를 물리 법칙으로 정의합니다.

PHASE2_CONTRAST_AXIOMS = """
╔══════════════════════════════════════════════════════════════════════╗
║  🔒 PHASE 2: PCD CONTRAST LAWS (위반 시 답변 자동 거부)              ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Law 4: μ(E) is energy-dependent                                    ║
║    → 감쇠계수는 에너지의 함수: μ = μ_pe(E) + μ_compton(E)           ║
║    → 광전효과: μ_pe ∝ Z³/E³ (저에너지 지배)                         ║
║    → 콤프턴 산란: μ_compton ≈ ρ (에너지 약한 의존)                   ║
║    → ∴ 대조도 Δμ(E)는 에너지에 따라 크게 변화                       ║
║                                                                      ║
║  Law 5: EID는 에너지 비례 가중(w∝E)으로 대조도 희석                  ║
║    → EID: 신호 = Σ E_i × N_i (에너지 통합 → 고에너지 과다 가중)     ║
║    → 대조도 Δμ는 저에너지에서 큼, EID는 고에너지에 더 큰 가중 부여  ║
║    → ∴ EID는 대조도를 구조적으로 희석 (sub-optimal)                  ║
║    → [Kalluri 2013, PMC3745502] "weight is inherently ∝ E deposited" ║
║                                                                      ║
║  Law 6: PCD 에너지 빈 → 최적 가중 → CNR_PCD ≥ CNR_EID              ║
║    → PCD: 각 에너지 빈별로 독립 측정 가능                            ║
║    → 최적 가중: w_i ∝ Δμ_i / σ_i (matched filter)                   ║
║    → CNR_PCD² = Σ [Δμ_i × t]² × N_i                                ║
║    → Cauchy-Schwarz 부등식: CNR_PCD ≥ CNR_EID (항상 성립)            ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  핵심 유도:                                                          ║
║  에너지 가중 이득 η = CNR_PCD / CNR_EID                              ║
║  η² = [Σ Δμ_i² × N_i] × [Σ N_i] / [Σ Δμ_i × N_i]²                ║
║  → η ≥ 1 (등호: 단색 빔 또는 Δμ가 에너지 무관일 때만)               ║
║                                                                      ║
║  K-edge 활용:                                                        ║
║  조영제(I: 33.2 keV) K-edge 전후 μ 급변                             ║
║  → PCD: K-edge 상하 빈 분리 → 극대 대조도                           ║
║  → EID: K-edge 정보가 전체 스펙트럼에 희석                           ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  ⚠️ CNR_PCD < CNR_EID 라는 결론은 물리적으로 불가능합니다.            ║
║  (동일 선량, 동일 환자, 최적 에너지 가중 적용 시)                    ║
╚══════════════════════════════════════════════════════════════════════╝
"""


# =============================================================================
# PHASE 3 CONSTITUTIONAL AXIOMS: DQE / NPS
# =============================================================================
# Phase 3 핵심 주제: DQE의 선량 의존성 — EID vs PCD
# Phase 1 Law 3 (σ_e²=const)의 주파수 도메인 표현입니다.
# EID: 저선량에서 DQE 저하 (전자노이즈 분산이 분모를 지배)
# PCD: DQE가 항상 η_abs (전자노이즈 물리적 제거)

PHASE3_DQE_AXIOMS = """
╔══════════════════════════════════════════════════════════════════════╗
║  🔒 PHASE 3: DQE/NPS LAWS (위반 시 답변 자동 거부)                  ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Law 7: DQE = SNR²_out / SNR²_in (정보 전달 효율)                   ║
║    → 0 ≤ DQE ≤ 1                                                    ║
║    → DQE=1: 이상적 검출기 (양자 노이즈만 존재)                      ║
║    → DQE<1: 검출기가 추가 노이즈 도입                                ║
║    → NEQ = DQE × q_in (Noise Equivalent Quanta)                     ║
║                                                                      ║
║  Law 8: DQE_EID(0,N) = η_abs / (1 + σ_e²/(η_abs×N))               ║
║    → 선량 N 감소 → 분모의 σ_e²/(η_abs×N) 증가 → DQE 감소           ║
║    → 저선량에서 전자노이즈가 DQE를 심각하게 저하                     ║
║    → f_e = σ_e²/(σ_q² + σ_e²) 라 하면:                             ║
║      DQE_EID(full) = η_abs × (1 - f_e)                             ║
║      DQE_EID(D) = η_abs / (1 + f_e/(η_abs×(1-f_e)×D))             ║
║    → 전형적 값: η_abs=0.85, f_e=0.30                                ║
║      DQE_EID(full) ≈ 0.700                                          ║
║      DQE_EID(half) ≈ 0.595                                          ║
║                                                                      ║
║  Law 9: DQE_PCD(0) = η_abs (선량 독립)                              ║
║    → PCD: 에너지 문턱치로 σ_e = 0                                   ║
║    → DQE_PCD = η_abs (상수, 선량 무관)                              ║
║    → 저선량에서도 DQE 유지 → 선량 절감의 물리적 근거                 ║
║    → 전형적 값: DQE_PCD = 0.850                                     ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  NPS 분해 (Noise Power Spectrum):                                    ║
║                                                                      ║
║  NPS_EID(f) = NPS_quantum(f) + NPS_electronic                       ║
║    → NPS_q = η_abs × N × D × a² (양자 노이즈 성분)                  ║
║    → NPS_e = σ_e² × a² (전자 노이즈 성분, 선량 무관)                ║
║                                                                      ║
║  NPS_PCD(f) = NPS_quantum(f) (전자 노이즈 없음)                      ║
║    → PCD는 NPS에서 전자 성분을 물리적으로 제거                       ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Phase 1 교차검증:                                                   ║
║  SNR_new/SNR_0 = √(DQE(D)×N_D / (DQE(1)×N_1))                      ║
║    = √(DQE_EID(0.5) / (2×DQE_EID(1)))                              ║
║    = √(0.595 / (2×0.700))                                           ║
║    = √0.4250 = 0.6519                                               ║
║    → Phase 1 SNR 공식 결과와 일치 ✓                                  ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  ⚠️ DQE_PCD < DQE_EID 라는 결론은 물리적으로 불가능합니다.           ║
║  (동일 η_abs에서 σ_e=0이면 DQE는 항상 최대)                         ║
╚══════════════════════════════════════════════════════════════════════╝
"""


# =============================================================================
# PHASE 4 CONSTITUTIONAL AXIOMS: MTF / Spatial Resolution / DQE(f)
# =============================================================================
# Phase 4 핵심 주제: 공간 해상도와 주파수 의존 DQE
# Phase 3의 DQE(0)을 주파수 도메인으로 확장: DQE(f) = MTF²(f) / [q₀ × NNPS(f)]
# EID: 간접 변환(CsI/GOS) 섬광체 확산 → MTF 저하
# PCD: 직접 변환(CdTe) 이상적 sinc 응답 → 3× 높은 해상도 한계

PHASE4_MTF_AXIOMS = """
╔══════════════════════════════════════════════════════════════════════╗
║  🔒 PHASE 4: MTF/SPATIAL RESOLUTION LAWS (위반 시 답변 자동 거부)    ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Law 10: MTF_direct(f) = sinc(π×f×a) — 직접 변환 검출기 aperture    ║
║    → PCD(CdTe): pixel pitch a에 의해서만 결정, 섬광체 blur 없음     ║
║    → EID(CsI): MTF_scint(f) 추가 곱 → 해상도 저하 불가피           ║
║    → MTF_EID(f) = sinc(π×f×a) × MTF_scint(f)                       ║
║    → MTF_scint(f) = exp(-(f/f_c)²) [Gaussian scintillator model]   ║
║    → PCD Nyquist: f_ny = 1/(2a), MTF_PCD(f_ny) = sinc(π/2) ≈ 0.637║
║                                                                      ║
║  Law 11: DQE(f) = MTF²(f) / [q₀ × NNPS(f)] — 주파수별 정보 전달   ║
║    → DQE(0) = Phase 3 결과와 일치 (교차 검증 필수)                  ║
║    → 고주파에서 PCD 우위: MTF 유지 + 전자노이즈 NPS 없음           ║
║    → DQE_PCD(f) = η_abs × MTF²_PCD(f) / [MTF²_PCD(f) + NPS_cs(f)] ║
║    → DQE_EID(f) = η_abs × MTF²_EID(f) / [MTF²_EID(f) + NPS_e_norm]║
║                                                                      ║
║  Law 12: Charge Sharing Trade-off — CdTe 두께↑ → QDE↑ but MTF↓     ║
║    → 3mm CdTe: QDE=0.97, charge sharing으로 MTF 10-25% 저하        ║
║    → CS_factor(f) ≈ 1 - δ×(f/f_ny)² [δ=0.10-0.25]                 ║
║    → Anti-Charge Sharing (ACS): DQE(0) 회복, 고주파 MTF 개선       ║
║    → PCD resolution limit / EID resolution limit ≈ 3× (Heismann)   ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  핵심 유도:                                                          ║
║  직접 변환 PCD: MTF_det(f) ≈ sinc(π×f×a) [이상적 aperture]         ║
║  간접 변환 EID: MTF_det(f) = sinc(π×f×a) × MTF_scint(f)           ║
║                                                                      ║
║  Phase 3 교차 검증:                                                  ║
║    DQE_PCD(f→0) = η_abs = 0.850 = Phase 3 DQE_PCD(0) ✓             ║
║    DQE_EID(f→0) = 0.700 = Phase 3 DQE_EID(full) ✓                  ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  ⚠️ MTF_PCD < MTF_EID (동일 pixel, f>0)는 물리적으로 불가능합니다.  ║
║  (직접 변환은 섬광체 blur가 없으므로 항상 MTF 우위)                  ║
╚══════════════════════════════════════════════════════════════════════╝
"""


# =============================================================================
# PHASE 4-B CONSTITUTIONAL AXIOMS: Biopsy Geometry & Calibration
# =============================================================================
# Phase 4-B 핵심 주제: 스테레오 정위 생검의 기하학적 정밀도
# Phase 4-A의 높은 공간 해상도(MTF)를 실제 시술 타겟팅 정밀도로 변환하는 브릿지
# 스테레오 시차(Parallax)로부터 3D 깊이를 산출하고 오차를 정량화합니다.

PHASE4B_BIOPSY_AXIOMS = """
╔══════════════════════════════════════════════════════════════════════╗
║  🔒 PHASE 4-B: BIOPSY GEOMETRY LAWS (위반 시 답변 자동 거부)        ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Law 13: Stereo Triangulation (스테레오 시차 원리)                   ║
║    → Z = Δx / (2 × sin(θ))                                         ║
║    → Δx = x₊ - x₋ (좌우 스테레오 영상 간 수평 변위, parallax)      ║
║    → θ = 스테레오 각도 (±15° 표준)                                  ║
║    → X = (x₊ + x₋) / 2 (수평 위치)                                 ║
║    → Y = (y₊ + y₋) / 2 (수직 위치, 양 영상에서 동일)               ║
║    → 시차(Parallax)가 3D 깊이 정보의 유일한 물리적 원천             ║
║                                                                      ║
║  Law 14: Geometric Error Amplification (기하학적 오차 증폭)         ║
║    → σ_Z = σ_Δx / (2 × sin(θ))                                     ║
║    → θ=15° 일 때: σ_Z = σ_Δx / 0.518 ≈ 1.93 × σ_Δx               ║
║    → Z축 오차는 항상 XY축 오차보다 큼 (기하학적 증폭 불가피)        ║
║    → 높은 MTF(Phase 4-A) → 작은 σ_Δx → 작은 σ_Z                   ║
║    → PCD: 더 높은 MTF → 더 정밀한 병변 위치 결정                    ║
║                                                                      ║
║  Law 15: DBT Depth Resolution (토모합성 깊이 분해능)                 ║
║    → Δz_FWHM ∝ 1 / sin(α_total/2)                                  ║
║    → α_total = 총 각도 범위 (15°-50°, 시스템 의존)                   ║
║    → 50° 시스템: Δz ≈ 1mm (높은 깊이 분해능)                        ║
║    → 15° 시스템: Δz ≈ 5mm (낮은 깊이 분해능)                        ║
║    → 삼각측량 불필요: 재구성 슬라이스에서 직접 깊이 결정              ║
║    → σ_Z_DBT = Δz_FWHM / (2√3) (균일 분포 가정)                    ║
║    → 기하학적 증폭(G) 없음: σ_Z가 σ_XY와 독립적으로 결정            ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  핵심 유도:                                                          ║
║  Total Targeting Error = √(σ_X² + σ_Y² + σ_Z² + σ_cal²)            ║
║    → σ_cal = 교정 오프셋 (바늘/홀더 기계적 오차)                    ║
║    → ACR 허용 기준: Total Error ≤ 1mm                               ║
║                                                                      ║
║  Phase 4-A 연결:                                                     ║
║    σ_Δx ≥ pixel_pitch / MTF_effective                                ║
║    PCD pixel_pitch = 0.1mm, MTF_eff ≈ 0.637 at Nyquist              ║
║    → σ_Δx_PCD ≈ 0.157mm (최소 측정 불확실성)                        ║
║    EID pixel_pitch = 0.1mm, MTF_eff ≈ 0.3-0.5 at Nyquist            ║
║    → σ_Δx_EID ≈ 0.2-0.33mm (MTF 저하로 불확실성 증가)              ║
║                                                                      ║
║  Stereo vs DBT 비교:                                                 ║
║    Stereo: σ_Z = σ_Δx × G (G>1, 기하학적 증폭 존재)                ║
║    DBT:    σ_Z = Δz/(2√3) (G 없음, 깊이 분해능에 의존)              ║
║    → 넓은 각도(50°) DBT가 stereo보다 Z축 정확도 우수                 ║
║    → 좁은 각도(15°) DBT는 stereo보다 Z축 정확도 열등할 수 있음       ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  ⚠️ σ_Z < σ_Δx 라는 결론은 물리적으로 불가능합니다 (Stereo only).   ║
║  (θ < 30° 에서는 항상 기하학적 증폭: 1/(2sinθ) > 1)                ║
║  ⚠️ DBT에서는 G 증폭이 없으므로 σ_Z < σ_XY 가 가능합니다.          ║
╚══════════════════════════════════════════════════════════════════════╝
"""


# =============================================================================
# PHASE 5 CONSTITUTIONAL AXIOMS: Tomosynthesis Image Quality Physics
# =============================================================================
# Phase 5 핵심 주제: DBT(토모합성)의 영상 품질 물리
# Phase 4-B의 기하학적 깊이 분해능을 영상 품질 지표(SNR, DQE, NEQ)로 확장
# 핵심 문제: N개 투영에 선량 분할 → 투영당 저선량 → EID DQE 저하, PCD 면역
# Phase 1(전자노이즈) + Phase 3(DQE 선량의존성)의 직접적 임상 귀결

PHASE5_TOMO_AXIOMS = """
╔══════════════════════════════════════════════════════════════════════╗
║  🔒 PHASE 5: TOMO IMAGE QUALITY LAWS (위반 시 답변 자동 거부)       ║
╠══════════════════════════════════════════════════════════════════════╣
║                                                                      ║
║  Law 16: Dose-Split DQE Degradation (토모합성 선량 분할)            ║
║    → D_proj = D_ref / N (투영당 선량, D_ref=2D 표준 선량)           ║
║    → EID: DQE(D_proj) = η_abs / (1 + α×N)                         ║
║    → PCD: DQE(D_proj) = η_abs (전자노이즈 없으므로 선량 무관)      ║
║    → PCD SNR advantage R = √(1 + α×N) (N↑ → R↑)                   ║
║    → α=0.2143 (Phase 3에서 역산): N=25 → R=2.52×                   ║
║    → 2D에서 1.10× → Tomo에서 2.52×: PCD 우위 극대화               ║
║    → 핵심 인사이트: 선량 분할이 EID 약점을 증폭시킴                 ║
║                                                                      ║
║  Law 17: Resolution Asymmetry (분해능 비대칭)                       ║
║    → In-plane: Δxy = pixel_pitch / MTF (Phase 4 결정)               ║
║    → Through-plane: Δz = K / sin(α_total/2) (Phase 4-B Law 15)     ║
║    → Asymmetry ratio: Δz/Δxy >> 1 (전형적 10-80×)                  ║
║    → 넓은 각도 → Δz↓ → 비대칭 완화 but 투영당 모션 blur↑          ║
║    → In-plane은 검출기 결정, through-plane은 기하학 결정             ║
║                                                                      ║
║  Law 18: Tomo SNR & Anatomical Clutter Rejection (해부학적 잡음 제거)║
║    → SNR_2D: 중첩 조직이 구조적 잡음(clutter)으로 작용             ║
║    → SNR_tomo: 깊이 분리로 중첩 제거 → 유효 SNR 향상               ║
║    → Clutter rejection gain: G_clutter = √(Δz/t_breast)            ║
║      (Δz=슬라이스 두께, t=유방 두께)                                ║
║    → 유효 SNR_tomo = SNR_quantum × (1/G_clutter)                   ║
║    → PCD 이중 우위: 높은 DQE(quantum) + 동일 clutter rejection     ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  Phase 교차 검증 체인:                                               ║
║  Phase 1: f_e=0.30 → Phase 3: α=0.2143 → Phase 5: DQE(D/N) 일관   ║
║  Phase 4: MTF(f) → 투영 영상의 in-plane 해상도                     ║
║  Phase 4-B: Δz = K/sin(α/2) → Phase 5: ASF_FWHM = through-plane   ║
║  Phase 5: 위 모두 통합 → 3D 영상 품질 = f(dose, N, α, detector)    ║
║                                                                      ║
║  핵심 유도:                                                          ║
║  SNR²_EID_tomo ∝ N × DQE_EID(D/N) × (D/N) = η×D / (1 + α×N)     ║
║  SNR²_PCD_tomo ∝ N × DQE_PCD × (D/N) = η×D                       ║
║  PCD/EID SNR ratio = √(1 + α×N)                                    ║
║    N=1:  R=1.10 (2D, Phase 3 일치 ✓)                               ║
║    N=15: R=2.01                                                      ║
║    N=25: R=2.52 (일반 DBT)                                          ║
║    N=49: R=3.46 (Siemens급)                                         ║
║                                                                      ║
╠══════════════════════════════════════════════════════════════════════╣
║  ⚠️ PCD advantage ratio < 1.0 이라는 결론은 물리적으로 불가능합니다.║
║  (N≥1에서 항상 √(1+α×N) ≥ 1, 등호는 α=0 즉 전자노이즈 없을 때만) ║
║  ⚠️ PCD advantage가 N 증가 시 감소한다는 결론은 틀립니다.           ║
║  (√(1+α×N)은 N에 대해 단조 증가)                                   ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def get_core_physics_prompt() -> str:
    """
    모든 LLM 호출에 포함될 핵심 물리 지식 반환

    DEPRECATED (Phase 7.19):
        이 함수는 더 이상 사용되지 않습니다.
        대신 UnifiedPromptBuilder.get_axioms()를 사용하세요.

        from src.prompts.unified_builder import UnifiedPromptBuilder
        from src.knowledge.manager import get_knowledge_manager
        builder = UnifiedPromptBuilder(get_knowledge_manager())
        axioms = builder.get_axioms()
    """
    warnings.warn(
        "get_core_physics_prompt() is deprecated. "
        "Use UnifiedPromptBuilder.get_axioms() instead. "
        "See src/prompts/unified_builder.py",
        DeprecationWarning,
        stacklevel=2
    )
    return f"""
{CONSTITUTIONAL_AXIOMS}

{PHASE2_CONTRAST_AXIOMS}

{PHASE3_DQE_AXIOMS}

{PHASE4_MTF_AXIOMS}

{PHASE4B_BIOPSY_AXIOMS}

{PHASE5_TOMO_AXIOMS}

# 📚 필수 물리 지식 (Core Physics Knowledge)

아래 내용은 반드시 이해하고 답변에 적용해야 하는 핵심 지식입니다.

{MGD_TOMOSYNTHESIS_KNOWLEDGE}

{QUANTUM_NOISE_KNOWLEDGE}

{FILTER_MTF_KNOWLEDGE}

{TARGET_FILTER_KNOWLEDGE}

---
**중요**: 위 내용과 충돌하는 답변을 생성하지 마세요.
특히 Dance et al. 2011 논문의 t-factor/T-factor 개념은 정확히 인용하세요.
위의 PHYSICS CONSTITUTIONAL LAWS를 절대 위반하지 마세요.
"""


# 테스트
if __name__ == "__main__":
    print("=" * 60)
    print("Core Physics Knowledge Test")
    print("=" * 60)

    # 기존 프롬프트 테스트
    prompt = get_core_physics_prompt()
    print(f"\nCore Physics Prompt: {len(prompt)} 문자")

    # GOLDEN_FORMULAS 테스트
    print(f"\n총 등록된 골든 포뮬러: {len(GOLDEN_FORMULAS)}개")
    print(f"수식 ID 목록: {get_all_formula_ids()}")

    # 특정 수식 조회
    print("\n" + "=" * 60)
    print("MGD_2D 수식 정보:")
    print("=" * 60)
    print(format_formula_for_prompt("MGD_2D"))

    print("\n" + "=" * 60)
    print("SNR_QUANTUM 수식 정보:")
    print("=" * 60)
    print(format_formula_for_prompt("SNR_QUANTUM"))
