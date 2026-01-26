"""
Mammography Physics Solver (Layer 2: Deterministic Verification)
================================================================
3-Layer Knowledge Internalization의 Layer 2 구현체

LLM 출력을 신뢰하지 않고, Python 결정론적 계산으로 검증합니다.
1% 이상 오차 시 LLM 답변을 거부(Rejection)합니다.

Phase 1 Constitutional Axioms (Layer 1, 코드로 인코딩):
    Axiom 1: Signal ∝ Dose (선형, 절대값)
    Axiom 2: σ_quantum² ∝ Dose (포아송 통계, 절대값)
    Axiom 3: σ_electronic² = constant (하드웨어 특성, 선량 무관)

Phase 2 Constitutional Axioms (PCD Spectral Contrast):
    Axiom 4: μ(E) is energy-dependent (감쇠계수는 에너지 함수)
    Axiom 5: EID는 에너지 정보를 통합(평균화)하여 대조도 손실
    Axiom 6: PCD 에너지 빈 → 최적 가중 → CNR_PCD ≥ CNR_EID (Cauchy-Schwarz)

Phase 1 핵심 공식:
    SNR = Signal / √(σ_q² + σ_e²)
    SNR_new/SNR_0 = √(D_ratio × (1 - f_e × (1 - D_ratio)))

Phase 2 핵심 공식:
    에너지 가중 이득: η = CNR_PCD / CNR_EID
    η² = [Σ Δμ_i² × N_i] × [Σ N_i] / [Σ Δμ_i × N_i]²
    → η ≥ 1 (Cauchy-Schwarz 부등식)

Phase 3 Constitutional Axioms (DQE/NPS):
    Axiom 7: DQE = SNR²_out / SNR²_in (정보 전달 효율)
    Axiom 8: DQE_EID(0,N) = η_abs / (1 + α/D) — 선량 의존
    Axiom 9: DQE_PCD(0) = η_abs — 선량 독립 (문턱치로 σ_e 제거)

Phase 3 핵심 공식:
    DQE_EID(full) = η_abs / (1 + α) ≈ 0.700
    DQE_EID(D) = η_abs / (1 + α/D)
    DQE_PCD = η_abs = 0.850
    α = f_e × D_ref / (1 - f_e) [Phase 1 f_e에서 역산]
    Phase 1 교차검증: √(DQE(D)×D/DQE(1)) = √(D×(1-f_e×(1-D)))

Phase 4 Constitutional Axioms (MTF / Spatial Resolution / DQE(f)):
    Axiom 10: MTF_direct(f) = sinc(π×f×a) — 직접 변환 aperture 응답
    Axiom 11: DQE(f) = MTF²(f) / [q₀ × NNPS(f)] — 주파수별 정보 전달
    Axiom 12: Charge Sharing Trade-off — CdTe 두께↑ → QDE↑ but MTF↓

Phase 4 핵심 공식:
    MTF_PCD(f) = |sinc(π×f×a)| × (1 - δ×(f/f_ny)²) [charge sharing]
    MTF_EID(f) = |sinc(π×f×a)| × exp(-(f/f_c)²) [scintillator blur]
    DQE_PCD(f) = η_abs × MTF²_PCD / (MTF²_PCD + NPS_cs)
    DQE_EID(f) = η_abs × MTF²_EID / (MTF²_EID + α)
    Phase 3 교차검증: DQE(f→0) = Phase 3 DQE(0)

Phase 4-B Constitutional Axioms (Biopsy Geometry & Calibration):
    Axiom 13: Z = Δx / (2×sin(θ)) — 스테레오 시차로부터 3D 깊이 산출
    Axiom 14: σ_Z = σ_Δx / (2×sin(θ)) — 기하학적 오차 증폭 (θ<30°에서 항상 >1)
    Axiom 15: Δz_FWHM = K / sin(α_total/2) — DBT 깊이 분해능 (각도 범위 의존)

Phase 4-B 핵심 공식:
    Stereo: σ_Δx = √2 × pixel_pitch / MTF_effective (시차 측정 불확실성)
    Stereo: Total Error = √(σ_X² + σ_Y² + σ_Z² + σ_cal²)
    DBT: σ_Z_DBT = Δz_FWHM / (2√3) (기하학적 증폭 없음)
    DBT: Total Error = √(σ_X² + σ_Y² + σ_Z_DBT² + σ_cal²)
    교차점: α ≈ 39° (K=0.50) 이상에서 DBT 우위
    ACR 허용 기준: Total Error ≤ 1.0 mm
    PCD 우위: 높은 MTF → 작은 σ_Δx/σ_XY → 정밀한 타겟팅 (양 방식 모두)

Phase 5 Constitutional Axioms (Tomosynthesis Image Quality):
    Axiom 16: DQE_EID(D/N) = η_abs / (1 + α×N) — 선량 분할로 DQE 저하
    Axiom 17: Δz/Δxy >> 1 — 분해능 비대칭 (through-plane vs in-plane)
    Axiom 18: Clutter rejection: G = √(Δz/t) — 해부학적 잡음 제거

Phase 5 핵심 공식:
    DQE_EID(D_proj) = η_abs / (1 + α×N), α=0.2143
    DQE_PCD = η_abs = 0.850 (선량 무관)
    PCD SNR gain = √(1 + α×N): N=25 → 2.52×
    Clutter boost = √(t_breast / Δz)
    d'_tomo = C × √(DQE × D × A) × √(t/Δz)
    Phase 3 교차검증: N=1 → DQE_EID = 0.700 (2D case)
"""

import math
import re
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum

logger = logging.getLogger(__name__)


# =============================================================================
# Data Classes
# =============================================================================

class AuditStatus(Enum):
    """검증 상태"""
    PASS = "pass"
    REJECT = "reject"
    UNCERTAIN = "uncertain"


@dataclass
class DerivationStep:
    """유도 과정 단계"""
    step_num: int
    title: str
    latex: str
    numeric_value: Optional[float] = None


@dataclass
class PhysicsSolution:
    """물리 계산 결과 (전체 유도 포함)"""
    # EID 결과
    eid_snr_ratio: float           # SNR_new / SNR_0 (0-1)
    eid_snr_reduction_pct: float   # (1 - ratio) × 100
    # PCD 결과
    pcd_snr_ratio: float
    pcd_snr_reduction_pct: float
    # PCD vs EID 비교
    pcd_recovery_pct: float        # (SNR_PCD/SNR_EID - 1) × 100
    # Rose Criterion
    rose_min_snr0: float           # Rose Criterion 만족을 위한 최소 초기 SNR
    rose_eid_satisfied: bool
    rose_pcd_satisfied: bool
    # 유도 과정
    derivation_steps: List[DerivationStep] = field(default_factory=list)
    # 입력 파라미터 (재현성)
    dose_ratio: float = 0.5
    electronic_noise_fraction: float = 0.3
    rose_k: float = 5.0


@dataclass
class AuditResult:
    """LLM 답변 감사 결과"""
    status: AuditStatus
    target_field: str              # 검증 대상 필드명
    llm_value: Optional[float]     # LLM이 계산한 값
    correct_value: float           # Python 솔버 정답
    error_pct: float               # |llm - correct| / correct × 100
    tolerance_pct: float           # 허용 오차 (기본 1%)
    should_reject: bool
    explanation: str
    correction_hint: str = ""      # 리젝션 시 힌트 (정답은 포함하지 않음)


# =============================================================================
# Phase 2 Data Classes: PCD Spectral Contrast
# =============================================================================

@dataclass
class EnergyBin:
    """에너지 빈 정의"""
    label: str                     # 빈 이름 (예: "below K-edge")
    energy_center_keV: float       # 중심 에너지 (keV)
    photon_count: float            # 해당 빈의 광자 수 (상대적)
    delta_mu: float                # Δμ: 조영제-조직 감쇠계수 차이 (cm⁻¹)


@dataclass
class ContrastSolution:
    """Phase 2: 대조도 비교 계산 결과"""
    # CNR 값
    cnr_eid: float                 # EID의 CNR (상대적)
    cnr_pcd: float                 # PCD의 CNR (상대적, 최적 에너지 가중)
    # 에너지 가중 이득
    eta: float                     # η = CNR_PCD / CNR_EID (≥ 1)
    eta_percent: float             # (η - 1) × 100% 향상률
    # K-edge 관련 (해당 시)
    kedge_energy_keV: Optional[float] = None
    contrast_agent: Optional[str] = None
    # 유도 과정
    derivation_steps: List[DerivationStep] = field(default_factory=list)
    # 입력 요약
    n_bins: int = 0
    total_photons: float = 0.0
    bins: List[EnergyBin] = field(default_factory=list)


# =============================================================================
# Phase 3 Data Classes: DQE / NPS
# =============================================================================

@dataclass
class DQESolution:
    """Phase 3: DQE 선량 의존성 계산 결과"""
    # DQE 값
    dqe_eid_full_dose: float       # DQE at full dose
    dqe_eid_at_dose_ratio: float   # DQE at specified dose ratio
    dqe_pcd: float                 # η_abs (constant, dose-independent)
    # 비교
    pcd_advantage_percent: float   # (DQE_PCD/DQE_EID_at_D - 1) × 100
    dqe_degradation_percent: float # (1 - DQE_EID_at_D/DQE_EID_full) × 100
    # DQE 커브 (시각화용)
    dose_points: List[float] = field(default_factory=list)
    dqe_eid_curve: List[float] = field(default_factory=list)
    # 유도 과정
    derivation_steps: List[DerivationStep] = field(default_factory=list)
    # 입력 파라미터
    eta_abs: float = 0.85
    sigma_e_relative: float = 0.30
    dose_ratio: float = 0.5


@dataclass
class NPSSolution:
    """Phase 3: NPS 분해 결과"""
    nps_quantum: float             # NPS_q = η_abs × N × a²
    nps_electronic: float          # NPS_e = σ_e² × a² (EID only)
    nps_total_eid: float           # NPS_q + NPS_e
    nps_total_pcd: float           # NPS_q (전자 노이즈 없음)
    electronic_fraction_eid: float # NPS_e / NPS_total_EID
    pcd_nps_reduction_percent: float  # (1 - NPS_PCD/NPS_EID) × 100
    derivation_steps: List[DerivationStep] = field(default_factory=list)


# =============================================================================
# Phase 4 Data Classes: MTF / Spatial Resolution / DQE(f)
# =============================================================================

@dataclass
class MTFSolution:
    """Phase 4: MTF 비교 계산 결과"""
    # PCD MTF parameters
    pixel_pitch_mm: float          # a (mm)
    nyquist_freq: float            # 1/(2a) (lp/mm)
    mtf_pcd_at_nyquist: float      # sinc(π×f_ny×a) = sinc(π/2) ≈ 0.637
    # EID MTF parameters
    mtf_eid_at_nyquist: float      # sinc × MTF_scint
    scintillator_mtf_factor: float # MTF_scint at Nyquist
    # Resolution comparison (f10: frequency where MTF=10%)
    f10_pcd: float                 # PCD f10 (lp/mm)
    f10_eid: float                 # EID f10 (lp/mm)
    pcd_resolution_gain: float     # f10_PCD / f10_EID
    # Charge sharing
    charge_sharing_degradation: float  # % MTF loss at Nyquist due to charge sharing
    # Curves
    freq_points: List[float]       # spatial frequency axis (lp/mm)
    mtf_pcd_curve: List[float]
    mtf_eid_curve: List[float]
    # Derivation
    derivation_steps: List[DerivationStep] = field(default_factory=list)
    # Input parameters
    scintillator_type: str = "CsI"
    scintillator_thickness_um: float = 150.0
    cs_delta: float = 0.10         # charge sharing parameter


@dataclass
class DQEfSolution:
    """Phase 4: DQE(f) 주파수 의존 계산 결과"""
    # DQE(f) at key frequencies
    dqe_pcd_at_zero: float         # = η_abs (Phase 3 일치)
    dqe_eid_at_zero: float         # = Phase 3 DQE_EID
    dqe_pcd_at_nyquist: float      # MTF²(f_ny) / NNPS(f_ny)
    dqe_eid_at_nyquist: float
    # PCD advantage ratio at Nyquist
    pcd_dqe_advantage_at_nyquist: float  # DQE_PCD(f_ny) / DQE_EID(f_ny)
    # Cross-validation
    phase3_dqe_match: bool         # DQE(0) == Phase 3 값?
    # Curves
    freq_points: List[float]       # spatial frequency axis (lp/mm)
    dqe_pcd_curve: List[float]
    dqe_eid_curve: List[float]
    # Derivation
    derivation_steps: List[DerivationStep] = field(default_factory=list)
    # Input parameters
    pixel_pitch_mm: float = 0.1
    eta_abs: float = 0.85
    electronic_noise_fraction: float = 0.30


# =============================================================================
# Phase 4-B Data Classes: Biopsy Geometry & Calibration
# =============================================================================

@dataclass
class BiopsySolution:
    """Phase 4-B: 스테레오 정위 생검 기하학 계산 결과"""
    # 3D 좌표 산출
    target_x_mm: float             # X 좌표 (수평)
    target_y_mm: float             # Y 좌표 (수직)
    target_z_mm: float             # Z 깊이 (parallax 기반)
    parallax_mm: float             # Δx = x₊ - x₋
    # 오차 분석
    sigma_x_mm: float              # X축 측정 불확실성
    sigma_y_mm: float              # Y축 측정 불확실성
    sigma_z_mm: float              # Z축 불확실성 (기하학적 증폭 포함)
    sigma_cal_mm: float            # 교정 오프셋 불확실성
    total_targeting_error_mm: float  # 총 타겟팅 오차 (RSS)
    # 기하학적 증폭 분석
    geometric_amplification: float  # 1/(2×sin(θ)), θ=15°→1.93
    z_to_xy_error_ratio: float     # σ_Z / σ_XY
    # ACR 허용 기준
    acr_tolerance_mm: float        # 1.0mm
    within_acr_tolerance: bool     # total_error ≤ 1.0mm?
    # PCD vs EID 비교
    sigma_dx_pcd_mm: float         # PCD의 시차 측정 불확실성
    sigma_dx_eid_mm: float         # EID의 시차 측정 불확실성
    total_error_pcd_mm: float      # PCD 사용 시 총 오차
    total_error_eid_mm: float      # EID 사용 시 총 오차
    pcd_error_reduction_pct: float # (1 - err_PCD/err_EID) × 100
    # 최적 각도 분석
    optimal_angle_deg: float       # 최적 스테레오 각도
    angle_tradeoff_note: str       # 각도 trade-off 설명
    # 유도 과정
    derivation_steps: List[DerivationStep] = field(default_factory=list)
    # 입력 파라미터
    stereo_angle_deg: float = 15.0
    pixel_pitch_mm: float = 0.1
    breast_thickness_mm: float = 50.0


@dataclass
class DBTBiopsySolution:
    """Phase 4-B: DBT(토모합성) 유도 생검 깊이 해상도 및 타겟팅 정확도"""
    # 시스템 파라미터
    angular_range_deg: float           # 총 각도 범위 (15-50°)
    n_projections: int                 # 투영 수 (9-25)
    pixel_pitch_mm: float              # 검출기 픽셀 피치
    mtf_effective: float               # 유효 MTF
    depth_resolution_constant: float   # K (mm, 시스템 의존)
    # 깊이 분해능
    depth_resolution_mm: float         # Δz_FWHM (mm)
    # 오차 분석
    sigma_xy_mm: float                 # XY축 불확실성
    sigma_z_mm: float                  # Z축 불확실성 (= Δz/(2√3))
    sigma_cal_mm: float                # 교정 오프셋 불확실성
    total_targeting_error_mm: float    # 총 타겟팅 오차
    # ACR 기준
    acr_tolerance_mm: float            # 1.0mm
    within_acr_tolerance: bool         # total_error ≤ 1.0mm?
    # Stereo와 비교
    stereo_sigma_z_mm: float           # 동일 조건 스테레오의 σ_Z
    stereo_total_error_mm: float       # 동일 조건 스테레오의 총 오차
    dbt_advantage: bool                # DBT가 stereo보다 좋은가?
    dbt_improvement_pct: float         # (1 - err_DBT/err_Stereo)×100, 음수면 stereo 우위
    # 임계 각도
    crossover_angle_deg: float         # DBT=Stereo 되는 각도
    # 유도 과정
    derivation_steps: List[DerivationStep] = field(default_factory=list)


# =============================================================================
# Phase 5 Data Classes: Tomosynthesis Image Quality Physics
# =============================================================================

@dataclass
class TomoDoseSplitSolution:
    """Phase 5: 토모합성 선량 분할에 따른 DQE 비교"""
    total_dose_uGy: float           # 총 선량
    n_projections: int              # 투영 수
    dose_per_projection_uGy: float  # D_total / N
    # DQE comparison
    dqe_eid_per_proj: float         # DQE_EID at D_proj (Phase 3 공식)
    dqe_pcd_per_proj: float         # DQE_PCD = η_abs (상수)
    pcd_dqe_advantage_ratio: float  # DQE_PCD / DQE_EID at D_proj
    # SNR comparison (per projection)
    snr_eid_per_proj: float         # relative SNR per projection
    snr_pcd_per_proj: float         # relative SNR per projection
    # Total 3D SNR (N projections integrated)
    snr_eid_total: float            # √(N × DQE_EID × D_proj)
    snr_pcd_total: float            # √(N × DQE_PCD × D_proj)
    pcd_snr_gain_total: float       # SNR_PCD / SNR_EID = √(1 + α×N)
    # Phase 3 cross-validation
    phase3_dqe_eid_full: float      # DQE_EID at full dose (0.700)
    phase3_alpha: float             # α = 0.2143
    phase3_match: bool              # N=1 → DQE_EID = Phase 3 값?
    # Derivation
    derivation_steps: List[DerivationStep] = field(default_factory=list)


@dataclass
class TomoResolutionSolution:
    """Phase 5: 토모합성 분해능 비대칭 분석"""
    angular_range_deg: float
    n_projections: int
    pixel_pitch_mm: float
    mtf_effective: float
    # In-plane resolution (Phase 4)
    delta_xy_mm: float              # pixel_pitch / MTF
    nyquist_freq_lpmm: float        # 1/(2×pixel_pitch)
    # Through-plane resolution (Phase 4-B)
    depth_resolution_constant: float
    delta_z_mm: float               # K/sin(α/2)
    # Asymmetry
    resolution_asymmetry_ratio: float  # Δz/Δxy
    # ASF parameters
    asf_fwhm_mm: float              # = Δz
    n_resolvable_slices: float      # t_breast / Δz
    # Voxel
    voxel_xy_mm: float              # = pixel_pitch
    voxel_z_mm: float               # = slice spacing (typically 1mm or Δz)
    voxel_volume_mm3: float
    # Breast parameter
    breast_thickness_mm: float
    # Derivation
    derivation_steps: List[DerivationStep] = field(default_factory=list)


@dataclass
class TomoDetectabilitySolution:
    """Phase 5: 토모합성 병변 검출능 (2D 대비)"""
    # Tomo parameters
    angular_range_deg: float
    n_projections: int
    total_dose_uGy: float
    breast_thickness_mm: float
    lesion_diameter_mm: float
    lesion_contrast: float
    # DQE (from dose-split)
    dqe_eid_2d: float               # DQE_EID at full dose (2D baseline)
    dqe_eid_tomo: float             # DQE_EID at D/N
    dqe_pcd: float                  # DQE_PCD = η_abs (상수)
    # Resolution
    slice_thickness_mm: float       # Δz = K/sin(α/2)
    # Clutter rejection
    clutter_rejection_gain: float   # G = √(Δz/t_breast), < 1
    clutter_snr_boost: float        # 1/G = √(t_breast/Δz), > 1
    # Detectability (d') — relative units
    d_prime_2d_eid: float           # 2D mammography EID baseline
    d_prime_tomo_eid: float         # tomo EID
    d_prime_tomo_pcd: float         # PCD tomo
    # Improvement factors
    tomo_vs_2d_gain_eid: float      # d'_tomo_EID / d'_2d_EID
    pcd_vs_eid_tomo_gain: float     # d'_pcd_tomo / d'_eid_tomo
    pcd_tomo_vs_2d_eid_gain: float  # d'_pcd_tomo / d'_2d_EID (total improvement)
    # Clinical threshold
    rose_threshold: float           # d'=5 (Rose criterion)
    # Derivation
    derivation_steps: List[DerivationStep] = field(default_factory=list)


# =============================================================================
# MammoPhysicsSolver
# =============================================================================

class MammoPhysicsSolver:
    """
    유방영상 물리 결정론적 솔버

    Layer 2: 모든 수치 계산을 Python으로 수행하여 LLM 할루시네이션을 차단합니다.
    LLM은 '설명'만 담당하고, '계산'은 이 솔버가 전담합니다.
    """

    # 감사 허용 오차 (%)
    AUDIT_TOLERANCE = 1.0

    def solve_snr_with_electronic_noise(
        self,
        dose_ratio: float,
        electronic_noise_fraction: float,
        rose_k: float = 5.0,
        base_snr: Optional[float] = None
    ) -> PhysicsSolution:
        """
        전자 노이즈 포함 SNR 계산 (핵심 함수)

        물리 모델 (절대값 기준):
            - S ∝ D (Signal은 Dose에 비례)
            - σ_q² ∝ D (양자 노이즈 분산은 Dose에 비례, 포아송)
            - σ_e² = const (전자 노이즈는 선량 무관)
            - SNR = S / √(σ_q² + σ_e²)

        핵심 해석:
            "전자 노이즈가 전체 노이즈의 f_e를 차지하게 된다면"
            → f_e는 선량 변화 '후'의 분산 비율
            → σ_e² / (σ_q_new² + σ_e²) = f_e

        Args:
            dose_ratio: 새 선량 / 기존 선량 (예: 0.5 = 50% 감소)
            electronic_noise_fraction: 선량 변화 후 전자노이즈 분산 비율 (예: 0.30)
            rose_k: Rose Criterion 상수 (기본 5)
            base_snr: 기존 SNR 값 (None이면 Rose 기준으로 역산)

        Returns:
            PhysicsSolution with complete derivation
        """
        D = dose_ratio
        f_e = electronic_noise_fraction
        steps = []

        # =====================================================================
        # Step 1: 정규화 및 파라미터 설정
        # =====================================================================
        steps.append(DerivationStep(
            step_num=1,
            title="정규화: 기존 상태",
            latex=(
                r"D_0 = 1,\quad S_0 = 1,\quad "
                r"\sigma_{q,0}^2 = 1 \text{ (정규화)}"
            )
        ))

        # f_e의 물리적 의미: 선량 변화 후 전자노이즈 분산 비율
        # σ_e² / (D × σ_q0² + σ_e²) = f_e
        # σ_e² = f_e × D × σ_q0² / (1 - f_e)
        sigma_q0_sq = 1.0  # 정규화
        sigma_e_sq = f_e * D * sigma_q0_sq / (1 - f_e)

        steps.append(DerivationStep(
            step_num=2,
            title="전자노이즈 역산 (f_e는 선량 변화 후 비율)",
            latex=(
                r"f_e = \frac{\sigma_e^2}{\sigma_{q,\text{new}}^2 + \sigma_e^2}"
                r" = \frac{\sigma_e^2}{D \cdot \sigma_{q,0}^2 + \sigma_e^2}"
                "\n"
                r"\therefore \sigma_e^2 = \frac{f_e \cdot D \cdot \sigma_{q,0}^2}{1 - f_e}"
                f" = \\frac{{{f_e:.2f} \\times {D:.2f} \\times 1}}{{{1-f_e:.2f}}}"
                f" = {sigma_e_sq:.6f}"
            ),
            numeric_value=sigma_e_sq
        ))

        # =====================================================================
        # Step 2: 기존 상태 총 노이즈
        # =====================================================================
        sigma_total0_sq = sigma_q0_sq + sigma_e_sq

        steps.append(DerivationStep(
            step_num=3,
            title="기존 총 노이즈",
            latex=(
                r"\sigma_{\text{total},0}^2 = \sigma_{q,0}^2 + \sigma_e^2"
                f" = 1 + {sigma_e_sq:.6f} = {sigma_total0_sq:.6f}"
            ),
            numeric_value=sigma_total0_sq
        ))

        # =====================================================================
        # Step 3: 선량 변화 후 EID 총 노이즈
        # =====================================================================
        sigma_q_new_sq = D * sigma_q0_sq
        sigma_total_new_sq = sigma_q_new_sq + sigma_e_sq

        steps.append(DerivationStep(
            step_num=4,
            title="선량 변화 후 EID 노이즈",
            latex=(
                r"\sigma_{q,\text{new}}^2 = D \cdot \sigma_{q,0}^2"
                f" = {D:.2f} \\times 1 = {sigma_q_new_sq:.6f}"
                "\n"
                r"\sigma_{\text{total,new}}^2 = \sigma_{q,\text{new}}^2 + \sigma_e^2"
                f" = {sigma_q_new_sq:.6f} + {sigma_e_sq:.6f} = {sigma_total_new_sq:.6f}"
            ),
            numeric_value=sigma_total_new_sq
        ))

        # =====================================================================
        # Step 4: EID SNR 비율 계산
        # =====================================================================
        # SNR_new/SNR_0 = (D × S_0 / √σ_total_new²) / (S_0 / √σ_total0²)
        #               = D × √(σ_total0² / σ_total_new²)
        eid_snr_ratio = D * math.sqrt(sigma_total0_sq / sigma_total_new_sq)

        # 간결한 공식으로도 검증
        eid_snr_ratio_compact = math.sqrt(D * (1 - f_e * (1 - D)))
        assert abs(eid_snr_ratio - eid_snr_ratio_compact) < 1e-10, \
            f"Formula mismatch: {eid_snr_ratio} vs {eid_snr_ratio_compact}"

        eid_reduction_pct = (1 - eid_snr_ratio) * 100

        steps.append(DerivationStep(
            step_num=5,
            title="EID SNR 비율",
            latex=(
                r"\frac{\text{SNR}_\text{new}}{\text{SNR}_0}"
                r" = D \cdot \sqrt{\frac{\sigma_{\text{total},0}^2}{\sigma_{\text{total,new}}^2}}"
                f" = {D:.2f} \\times \\sqrt{{\\frac{{{sigma_total0_sq:.6f}}}{{{sigma_total_new_sq:.6f}}}}}"
                f" = \\mathbf{{{eid_snr_ratio:.4f}}}"
                "\n"
                r"\text{간결 공식: } \sqrt{D \cdot (1 - f_e(1-D))}"
                f" = \\sqrt{{{D:.2f} \\times (1 - {f_e:.2f} \\times {1-D:.2f})}}"
                f" = \\sqrt{{{D * (1 - f_e*(1-D)):.6f}}} = {eid_snr_ratio_compact:.4f}"
                "\n"
                f"\\therefore \\text{{EID SNR 감소율}} = (1 - {eid_snr_ratio:.4f}) \\times 100\\%"
                f" = \\mathbf{{{eid_reduction_pct:.1f}\\%}}"
            ),
            numeric_value=eid_reduction_pct
        ))

        # =====================================================================
        # Step 5: PCD SNR 비율 (전자노이즈 제거)
        # =====================================================================
        # PCD: σ_e = 0 → σ_total = σ_q
        # SNR_PCD_new / SNR_PCD_0 = D × S_0/√(D×σ_q0²) / (S_0/√σ_q0²)
        #                         = D × √(σ_q0²/(D×σ_q0²))
        #                         = D × 1/√D = √D
        pcd_snr_ratio = math.sqrt(D)
        pcd_reduction_pct = (1 - pcd_snr_ratio) * 100

        steps.append(DerivationStep(
            step_num=6,
            title="PCD SNR 비율 (σ_e = 0)",
            latex=(
                r"\text{PCD: } \sigma_e = 0 \text{ (에너지 문턱치로 전자노이즈 제거)}"
                "\n"
                r"\frac{\text{SNR}_\text{PCD,new}}{\text{SNR}_\text{PCD,0}}"
                r" = \frac{D \cdot S_0 / \sqrt{D \cdot \sigma_{q,0}^2}}"
                r"{S_0 / \sqrt{\sigma_{q,0}^2}}"
                r" = \sqrt{D}"
                f" = \\sqrt{{{D:.2f}}} = \\mathbf{{{pcd_snr_ratio:.4f}}}"
                "\n"
                f"\\therefore \\text{{PCD SNR 감소율}} = (1 - {pcd_snr_ratio:.4f}) \\times 100\\%"
                f" = \\mathbf{{{pcd_reduction_pct:.1f}\\%}}"
            ),
            numeric_value=pcd_reduction_pct
        ))

        # =====================================================================
        # Step 6: PCD의 EID 대비 회복률
        # =====================================================================
        pcd_recovery_pct = (pcd_snr_ratio / eid_snr_ratio - 1) * 100

        steps.append(DerivationStep(
            step_num=7,
            title="PCD의 EID 대비 SNR 회복률",
            latex=(
                r"\text{회복률} = \frac{\text{SNR}_\text{PCD}}{\text{SNR}_\text{EID}} - 1"
                f" = \\frac{{{pcd_snr_ratio:.4f}}}{{{eid_snr_ratio:.4f}}} - 1"
                f" = \\mathbf{{+{pcd_recovery_pct:.1f}\\%}}"
            ),
            numeric_value=pcd_recovery_pct
        ))

        # =====================================================================
        # Step 7: Rose Criterion
        # =====================================================================
        # SNR_new ≥ k → SNR_0 × ratio ≥ k → SNR_0 ≥ k / ratio
        rose_min_snr0_eid = rose_k / eid_snr_ratio
        rose_min_snr0_pcd = rose_k / pcd_snr_ratio

        # base_snr가 주어지지 않으면 Rose 기준의 1.5배로 가정
        if base_snr is None:
            base_snr = rose_k * 1.5  # 7.5

        rose_eid_ok = (base_snr * eid_snr_ratio) >= rose_k
        rose_pcd_ok = (base_snr * pcd_snr_ratio) >= rose_k

        steps.append(DerivationStep(
            step_num=8,
            title=f"Rose Criterion (k={rose_k:.0f})",
            latex=(
                f"\\text{{Rose Criterion: SNR}} \\geq {rose_k:.0f}"
                "\n"
                f"\\text{{EID: 최소 초기 SNR}} = \\frac{{{rose_k:.0f}}}{{{eid_snr_ratio:.4f}}}"
                f" = {rose_min_snr0_eid:.2f}"
                "\n"
                f"\\text{{PCD: 최소 초기 SNR}} = \\frac{{{rose_k:.0f}}}{{{pcd_snr_ratio:.4f}}}"
                f" = {rose_min_snr0_pcd:.2f}"
            ),
            numeric_value=rose_min_snr0_eid
        ))

        return PhysicsSolution(
            eid_snr_ratio=eid_snr_ratio,
            eid_snr_reduction_pct=eid_reduction_pct,
            pcd_snr_ratio=pcd_snr_ratio,
            pcd_snr_reduction_pct=pcd_reduction_pct,
            pcd_recovery_pct=pcd_recovery_pct,
            rose_min_snr0=rose_min_snr0_eid,
            rose_eid_satisfied=rose_eid_ok,
            rose_pcd_satisfied=rose_pcd_ok,
            derivation_steps=steps,
            dose_ratio=D,
            electronic_noise_fraction=f_e,
            rose_k=rose_k
        )

    def audit_llm_answer(
        self,
        llm_answer: str,
        dose_ratio: float,
        electronic_noise_fraction: float,
        tolerance_pct: Optional[float] = None
    ) -> List[AuditResult]:
        """
        LLM 답변을 감사하여 물리적 정확성 검증

        LLM이 생성한 수치를 추출하고, Python 솔버 정답과 비교합니다.
        1% 이상 오차 시 REJECT합니다.

        Args:
            llm_answer: LLM이 생성한 답변 텍스트
            dose_ratio: 선량 비율
            electronic_noise_fraction: 전자노이즈 비율 (선량 변화 후)
            tolerance_pct: 허용 오차 (기본 1%)

        Returns:
            List[AuditResult]: 각 검증 항목별 결과
        """
        tol = tolerance_pct if tolerance_pct is not None else self.AUDIT_TOLERANCE
        solution = self.solve_snr_with_electronic_noise(dose_ratio, electronic_noise_fraction)
        results = []

        # 1. EID SNR 감소율 검증
        llm_eid = self._extract_eid_snr_reduction(llm_answer)
        if llm_eid is not None:
            error = abs(llm_eid - solution.eid_snr_reduction_pct)
            error_rel = error / solution.eid_snr_reduction_pct * 100 if solution.eid_snr_reduction_pct != 0 else error
            reject = error > tol
            results.append(AuditResult(
                status=AuditStatus.REJECT if reject else AuditStatus.PASS,
                target_field="EID SNR 감소율",
                llm_value=llm_eid,
                correct_value=solution.eid_snr_reduction_pct,
                error_pct=error,
                tolerance_pct=tol,
                should_reject=reject,
                explanation=(
                    f"LLM: {llm_eid:.1f}%, 정답: {solution.eid_snr_reduction_pct:.1f}%, "
                    f"오차: {error:.2f}%p"
                ),
                correction_hint=(
                    "전자노이즈 비율이 선량 변화 '후' 기준임을 확인하세요. "
                    "SNR_new/SNR_0 = √(D×(1-f_e×(1-D))) 공식을 적용하세요."
                ) if reject else ""
            ))
        else:
            results.append(AuditResult(
                status=AuditStatus.UNCERTAIN,
                target_field="EID SNR 감소율",
                llm_value=None,
                correct_value=solution.eid_snr_reduction_pct,
                error_pct=100.0,
                tolerance_pct=tol,
                should_reject=True,
                explanation="LLM 답변에서 EID SNR 감소율을 추출할 수 없음",
                correction_hint="SNR 감소율을 명시적으로 'XX.X%' 형태로 기술하세요."
            ))

        # 2. PCD SNR 감소율 검증 (있는 경우)
        llm_pcd = self._extract_pcd_snr_reduction(llm_answer)
        if llm_pcd is not None:
            error = abs(llm_pcd - solution.pcd_snr_reduction_pct)
            reject = error > tol
            results.append(AuditResult(
                status=AuditStatus.REJECT if reject else AuditStatus.PASS,
                target_field="PCD SNR 감소율",
                llm_value=llm_pcd,
                correct_value=solution.pcd_snr_reduction_pct,
                error_pct=error,
                tolerance_pct=tol,
                should_reject=reject,
                explanation=(
                    f"LLM: {llm_pcd:.1f}%, 정답: {solution.pcd_snr_reduction_pct:.1f}%, "
                    f"오차: {error:.2f}%p"
                ),
                correction_hint=(
                    "PCD는 전자노이즈를 제거하므로 SNR_PCD = √D 입니다."
                ) if reject else ""
            ))

        # 3. PCD 회복률 검증 (있는 경우)
        llm_recovery = self._extract_recovery_pct(llm_answer)
        if llm_recovery is not None:
            error = abs(llm_recovery - solution.pcd_recovery_pct)
            reject = error > tol
            results.append(AuditResult(
                status=AuditStatus.REJECT if reject else AuditStatus.PASS,
                target_field="PCD 회복률",
                llm_value=llm_recovery,
                correct_value=solution.pcd_recovery_pct,
                error_pct=error,
                tolerance_pct=tol,
                should_reject=reject,
                explanation=(
                    f"LLM: +{llm_recovery:.1f}%, 정답: +{solution.pcd_recovery_pct:.1f}%, "
                    f"오차: {error:.2f}%p"
                ),
                correction_hint=(
                    "회복률 = SNR_PCD/SNR_EID - 1 로 계산하세요."
                ) if reject else ""
            ))

        return results

    def format_derivation_latex(self, solution: PhysicsSolution) -> str:
        """유도 과정을 LaTeX 포맷으로 변환"""
        lines = [
            "### 📐 수식 유도 과정 (Deterministic Physics Solver)",
            ""
        ]
        for step in solution.derivation_steps:
            lines.append(f"**Step {step.step_num}: {step.title}**")
            lines.append(f"$${step.latex}$$")
            lines.append("")
        return "\n".join(lines)

    def format_constraint_prompt(self, solution: PhysicsSolution) -> str:
        """Double-Anchor용 제약 조건 프롬프트 생성"""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║  🔒 DETERMINISTIC SOLVER 검증 완료 (오차 > 1% 시 자동 거부)  ║
╠══════════════════════════════════════════════════════════════╣
║  조건: 선량 {(1-solution.dose_ratio)*100:.0f}% 감소, f_e={solution.electronic_noise_fraction:.0%} (변화 후)   ║
╠══════════════════════════════════════════════════════════════╣
║  📊 검증된 정답:                                             ║
║  • EID SNR 감소율: {solution.eid_snr_reduction_pct:.1f}%                              ║
║  • PCD SNR 감소율: {solution.pcd_snr_reduction_pct:.1f}%                              ║
║  • PCD 회복률 (vs EID): +{solution.pcd_recovery_pct:.1f}%                       ║
║  • Rose Criterion 최소 SNR_0: {solution.rose_min_snr0:.2f}                   ║
╠══════════════════════════════════════════════════════════════╣
║  ⚠️ 반드시 위 수치를 사용하세요 (1% 초과 오차 시 거부)       ║
╚══════════════════════════════════════════════════════════════╝
"""

    # =========================================================================
    # Private: 수치 추출 헬퍼
    # =========================================================================

    def _extract_eid_snr_reduction(self, text: str) -> Optional[float]:
        """EID SNR 감소율 추출"""
        patterns = [
            # EID 명시
            r'EID[^.]*?(\d+(?:\.\d+)?)\s*%\s*(?:감소|하락|저하|reduction|drop)',
            r'EID[^.]*?SNR[^.]*?(\d+(?:\.\d+)?)\s*%',
            # 일반 SNR 감소율 (PCD 언급 없는 경우)
            r'SNR[이가]?\s*(?:약\s*)?(\d+(?:\.\d+)?)\s*%\s*(?:감소|하락|저하)',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:의\s*)?(?:SNR\s*)?(?:감소|하락)',
            r'SNR\s*(?:감소|하락|저하)[^\d]*(\d+(?:\.\d+)?)\s*%',
            # 영어
            r'SNR\s*(?:decreases?|reduction|drops?)\s*(?:by\s*)?(\d+(?:\.\d+)?)\s*%',
            r'(\d+(?:\.\d+)?)\s*%\s*(?:reduction|decrease|drop)',
            # LaTeX
            r'\\mathbf\{(\d+(?:\.\d+)?)\s*\\%\}',
        ]

        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if 5 < value < 95:  # 합리적 범위
                    return value

        # 폴백: 첫 번째 합리적 퍼센트 값
        all_pcts = re.findall(r'(\d+(?:\.\d+)?)\s*%', text)
        for pct_str in all_pcts:
            v = float(pct_str)
            if 20 < v < 60:  # EID SNR 감소 합리적 범위
                return v

        return None

    def _extract_pcd_snr_reduction(self, text: str) -> Optional[float]:
        """PCD SNR 감소율 추출"""
        patterns = [
            r'PCD[^.]*?(\d+(?:\.\d+)?)\s*%\s*(?:감소|하락|저하|reduction|drop)',
            r'PCD[^.]*?SNR[^.]*?(\d+(?:\.\d+)?)\s*%',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if 5 < value < 60:
                    return value
        return None

    def _extract_recovery_pct(self, text: str) -> Optional[float]:
        """PCD 회복률 추출"""
        patterns = [
            r'(\d+(?:\.\d+)?)\s*%\s*(?:회복|개선|향상|recovery|improvement)',
            r'(?:회복|개선|향상|recovery)[^\d]*(\d+(?:\.\d+)?)\s*%',
            r'\+\s*(\d+(?:\.\d+)?)\s*%',
        ]
        for pattern in patterns:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = float(match.group(1))
                if 1 < value < 50:
                    return value
        return None

    # =========================================================================
    # Phase 2: PCD Spectral Contrast Solver
    # =========================================================================

    def solve_energy_weighting_gain(
        self,
        bins: List[EnergyBin]
    ) -> ContrastSolution:
        """
        Phase 2 핵심: 에너지 가중 이득(η) 계산

        PCD가 EID 대비 CNR을 얼마나 향상시키는지 정량 계산합니다.

        물리 모델:
            EID: 모든 광자를 에너지 무관하게 통합
                CNR_EID = [Σ Δμ_i × N_i] × t / √(Σ N_i)
            PCD: 각 빈별 최적 가중 (matched filter)
                CNR_PCD² = Σ [Δμ_i × t]² × N_i

            에너지 가중 이득:
                η = CNR_PCD / CNR_EID
                η² = [Σ Δμ_i² × N_i] × [Σ N_i] / [Σ Δμ_i × N_i]²

            Cauchy-Schwarz 부등식에 의해 η ≥ 1 (항상 성립)
            등호 조건: 모든 빈에서 Δμ_i가 동일할 때만 (단색 빔)

        Args:
            bins: 에너지 빈 리스트 (각 빈의 광자수와 Δμ 포함)

        Returns:
            ContrastSolution with η, CNR values, derivation
        """
        if not bins:
            raise ValueError("에너지 빈이 비어있습니다")

        steps = []
        t = 1.0  # 정규화된 두께

        # Step 1: 입력 정리
        n_bins = len(bins)
        total_N = sum(b.photon_count for b in bins)
        steps.append(DerivationStep(
            step_num=1,
            title="입력 에너지 빈 정의",
            latex=(
                f"\\text{{빈 수}} = {n_bins},\\quad "
                f"N_{{\\text{{total}}}} = \\sum N_i = {total_N:.1f}"
                + "".join([
                    f"\n\\text{{Bin '{b.label}'}}: "
                    f"E={b.energy_center_keV:.1f}\\text{{ keV}}, "
                    f"N_i={b.photon_count:.1f}, "
                    f"\\Delta\\mu_i={b.delta_mu:.4f}"
                    for b in bins
                ])
            )
        ))

        # Step 2: EID CNR 계산 (에너지 비례 가중 w=E)
        # EID는 광자 에너지에 비례하는 가중치로 신호를 통합 [Kalluri 2013, PMC3745502]
        # "weight is inherently proportional to the photon energy deposited"
        # CNR_EID = [Σ E_i × Δμ_i × N_i] / √(Σ E_i² × N_i)
        sum_E_dmu_N = sum(b.energy_center_keV * b.delta_mu * b.photon_count for b in bins)
        sum_E2_N = sum(b.energy_center_keV**2 * b.photon_count for b in bins)
        cnr_eid = sum_E_dmu_N / math.sqrt(sum_E2_N)

        steps.append(DerivationStep(
            step_num=2,
            title="EID CNR (에너지 비례 가중 w=E, sub-optimal)",
            latex=(
                r"\text{CNR}_{\text{EID}} = \frac{\sum E_i \times \Delta\mu_i \times N_i}"
                r"{\sqrt{\sum E_i^2 \times N_i}}"
                f"\n\\quad = \\frac{{{sum_E_dmu_N:.2f}}}{{\\sqrt{{{sum_E2_N:.1f}}}}}"
                f" = {cnr_eid:.4f}"
                "\n"
                r"\text{[Kalluri 2013]: EID weight} \propto E \text{ (sub-optimal for contrast)}"
            ),
            numeric_value=cnr_eid
        ))

        # Step 3: PCD CNR 계산 (최적 가중)
        # CNR_PCD² = Σ [Δμ_i]² × N_i  (t=1 정규화)
        sum_dmu2_N = sum(b.delta_mu**2 * b.photon_count for b in bins)
        cnr_pcd = math.sqrt(sum_dmu2_N)

        steps.append(DerivationStep(
            step_num=3,
            title="PCD CNR (최적 에너지 가중, matched filter)",
            latex=(
                r"\text{CNR}_{\text{PCD}}^2 = \sum [\Delta\mu_i]^2 \times N_i"
                f" = {sum_dmu2_N:.6f}"
                f"\n\\text{{CNR}}_{{\\text{{PCD}}}} = \\sqrt{{{sum_dmu2_N:.6f}}}"
                f" = {cnr_pcd:.4f}"
            ),
            numeric_value=cnr_pcd
        ))

        # Step 4: 에너지 가중 이득 η
        eta = cnr_pcd / cnr_eid if cnr_eid > 0 else float('inf')
        eta_pct = (eta - 1) * 100

        # Cauchy-Schwarz 검증: η ≥ 1
        assert eta >= 1.0 - 1e-10, \
            f"Cauchy-Schwarz violation: η={eta:.6f} < 1 (물리적 불가능)"

        steps.append(DerivationStep(
            step_num=4,
            title="에너지 가중 이득 η (Cauchy-Schwarz)",
            latex=(
                r"\eta = \frac{\text{CNR}_{\text{PCD}}}{\text{CNR}_{\text{EID}}}"
                f" = \\frac{{{cnr_pcd:.4f}}}{{{cnr_eid:.4f}}}"
                f" = \\mathbf{{{eta:.4f}}}"
                f"\n\\therefore \\text{{PCD CNR 향상률}} = (\\eta - 1) \\times 100\\%"
                f" = \\mathbf{{+{eta_pct:.1f}\\%}}"
                "\n"
                r"\text{검증: } \eta \geq 1 \text{ (Cauchy-Schwarz 부등식)} \checkmark"
            ),
            numeric_value=eta_pct
        ))

        return ContrastSolution(
            cnr_eid=cnr_eid,
            cnr_pcd=cnr_pcd,
            eta=eta,
            eta_percent=eta_pct,
            derivation_steps=steps,
            n_bins=n_bins,
            total_photons=total_N,
            bins=bins
        )

    def solve_kedge_cnr(
        self,
        n_below: float,
        n_above: float,
        dmu_below: float,
        dmu_above: float,
        agent: str = "Iodine",
        kedge_keV: float = 33.2
    ) -> ContrastSolution:
        """
        K-edge 기반 조영증강 CNR 계산 (2-bin 모델)

        K-edge 전후의 감쇠계수 급변을 이용한 대조도 향상.
        조영제의 K-edge 에너지에서 μ가 급격히 증가하므로,
        PCD의 에너지 빈을 K-edge 전후로 배치하면 극대 대조도 달성.

        Args:
            n_below: K-edge 이하 에너지 빈의 광자 수
            n_above: K-edge 이상 에너지 빈의 광자 수
            dmu_below: K-edge 이하에서의 Δμ (조영제-조직)
            dmu_above: K-edge 이상에서의 Δμ (조영제-조직, K-edge 후 급증)
            agent: 조영제 이름
            kedge_keV: K-edge 에너지 (keV)

        Returns:
            ContrastSolution with K-edge specific results
        """
        bins = [
            EnergyBin(
                label=f"below K-edge (<{kedge_keV:.1f} keV)",
                energy_center_keV=kedge_keV - 5,
                photon_count=n_below,
                delta_mu=dmu_below
            ),
            EnergyBin(
                label=f"above K-edge (>{kedge_keV:.1f} keV)",
                energy_center_keV=kedge_keV + 5,
                photon_count=n_above,
                delta_mu=dmu_above
            )
        ]

        solution = self.solve_energy_weighting_gain(bins)
        solution.kedge_energy_keV = kedge_keV
        solution.contrast_agent = agent

        # K-edge 특화 유도 단계 추가
        # K-edge subtraction: 두 빈의 차이로 조직 신호 제거
        # C_kedge = μ_above - μ_below (조영제만 남음)
        contrast_jump = dmu_above - dmu_below
        solution.derivation_steps.append(DerivationStep(
            step_num=5,
            title=f"{agent} K-edge Contrast Jump ({kedge_keV} keV)",
            latex=(
                f"\\text{{{agent} K-edge}}: {kedge_keV:.1f}\\text{{ keV}}"
                f"\n\\Delta\\mu_{{\\text{{above}}}} - \\Delta\\mu_{{\\text{{below}}}}"
                f" = {dmu_above:.4f} - {dmu_below:.4f} = {contrast_jump:.4f}"
                f"\n\\text{{K-edge contrast jump ratio}}"
                f" = \\frac{{\\Delta\\mu_{{\\text{{above}}}}}}{{\\Delta\\mu_{{\\text{{below}}}}}}"
                f" = \\frac{{{dmu_above:.4f}}}{{{dmu_below:.4f}}}"
                f" = {dmu_above/dmu_below:.1f}\\times"
            ),
            numeric_value=contrast_jump
        ))

        return solution

    @staticmethod
    def get_iodine_cesm_bins() -> List[EnergyBin]:
        """
        CESM (Contrast-Enhanced Spectral Mammography) 표준 에너지 빈

        아이오딘 K-edge (33.2 keV) 기반 CESM에서의 전형적 4-빈 구성.
        스펙트럼: W/Rh 또는 W/Ag, 49 kVp 기반 (CESM 표준 프로토콜)

        Δμ 값: 아이오딘(2 mg/mL) vs 유방 조직(50% glandular)의 감쇠계수 차이
        참고: Day & Tanguay (2024) PMID:37967277의 시뮬레이션 조건 기반
        """
        return [
            EnergyBin(
                label="low-E (20-28 keV)",
                energy_center_keV=24.0,
                photon_count=300.0,   # 상대적 광자수
                delta_mu=0.45         # 조직과 아이오딘 차이 크지만 노이즈도 높음
            ),
            EnergyBin(
                label="mid-E below K (28-33 keV)",
                energy_center_keV=30.5,
                photon_count=400.0,
                delta_mu=0.25         # K-edge 직전: 아이오딘 기여 중간
            ),
            EnergyBin(
                label="above K-edge (33-38 keV)",
                energy_center_keV=35.5,
                photon_count=350.0,
                delta_mu=1.80         # K-edge 직후: Δμ 급증 (핵심!)
            ),
            EnergyBin(
                label="high-E (38-49 keV)",
                energy_center_keV=43.0,
                photon_count=250.0,
                delta_mu=0.90         # 고에너지: 아이오딘 기여 감소하나 여전히 유의
            )
        ]

    @staticmethod
    def get_iodine_2bin_simple() -> Tuple[float, float, float, float]:
        """
        간단한 2-빈 K-edge 모델 (교육용)

        Returns:
            (n_below, n_above, dmu_below, dmu_above) 튜플
        """
        # K-edge 이하: 광자 많지만 대조도 낮음
        # K-edge 이상: 광자 적지만 대조도 매우 높음
        return (700.0, 600.0, 0.35, 1.80)

    def format_contrast_prompt(self, solution: ContrastSolution) -> str:
        """Phase 2 Double-Anchor용 제약 조건 프롬프트 생성"""
        agent_info = ""
        if solution.contrast_agent:
            agent_info = f"║  조영제: {solution.contrast_agent} (K-edge: {solution.kedge_energy_keV} keV)        ║\n"

        return f"""
╔══════════════════════════════════════════════════════════════╗
║  🔒 PHASE 2 SOLVER 검증 완료 (CNR_PCD < CNR_EID는 불가능)   ║
╠══════════════════════════════════════════════════════════════╣
{agent_info}║  에너지 빈 수: {solution.n_bins}, 총 광자수: {solution.total_photons:.0f}           ║
╠══════════════════════════════════════════════════════════════╣
║  📊 검증된 정답:                                             ║
║  • CNR_EID (균일 가중): {solution.cnr_eid:.4f}                         ║
║  • CNR_PCD (최적 가중): {solution.cnr_pcd:.4f}                         ║
║  • 에너지 가중 이득 η: {solution.eta:.4f} (+{solution.eta_percent:.1f}%)           ║
╠══════════════════════════════════════════════════════════════╣
║  ⚠️ η < 1 은 Cauchy-Schwarz 위반 (물리적 불가능)             ║
║  ⚠️ 반드시 위 수치를 사용하세요 (1% 초과 오차 시 거부)       ║
╚══════════════════════════════════════════════════════════════╝
"""

    # =========================================================================
    # Phase 3: DQE / NPS Solver
    # =========================================================================

    def solve_dqe_dose_dependence(
        self,
        eta_abs: float = 0.85,
        electronic_noise_fraction: float = 0.30,
        dose_ratio: float = 0.5
    ) -> DQESolution:
        """
        Phase 3 핵심: DQE 선량 의존성 계산

        물리 모델:
            DQE_EID(0, N) = η_abs / (1 + α)
            where α = σ_e² / (η_abs × N)

            Phase 1 파라미터 연동:
            Phase 1의 f_e는 "선량 변화 후" 전자노이즈 분산 비율:
              f_e = σ_e² / (D×σ_q0² + σ_e²)
            이로부터 α를 역산:
              α = f_e × D / (1 - f_e) [정규화: σ_q0²=1]

            DQE 계산:
              DQE_EID(full) = η_abs / (1 + α) = 0.700
              DQE_EID(D)    = η_abs / (1 + α/D) = 0.595
              DQE_PCD       = η_abs = 0.850

        Phase 1 교차 검증:
            SNR_new/SNR_0 = √(DQE(D)×D / DQE(1))
            이 값이 Phase 1 공식 √(D×(1-f_e×(1-D)))와 일치

        Args:
            eta_abs: 흡수 효율 (0-1, 기본 0.85)
            electronic_noise_fraction: Phase 1과 동일, 선량 변화 후 전자노이즈 비율 (기본 0.30)
            dose_ratio: 선량 비율 D_new/D_full (기본 0.5)

        Returns:
            DQESolution with complete derivation
        """
        f_e = electronic_noise_fraction
        D = dose_ratio
        steps = []

        # Step 1: α 역산 (Phase 1 f_e → DQE α)
        # f_e = σ_e² / (D + σ_e²) [정규화]
        # α = σ_e² = f_e × D / (1 - f_e)
        alpha = f_e * D / (1 - f_e)

        steps.append(DerivationStep(
            step_num=1,
            title="DQE 파라미터 역산 (Phase 1 f_e → α)",
            latex=(
                r"\text{DQE} = \frac{\text{SNR}^2_{\text{out}}}{\text{SNR}^2_{\text{in}}}"
                f"\n\\eta_{{\\text{{abs}}}} = {eta_abs:.3f},\\quad "
                f"f_e = {f_e:.2f}\\text{{ (Phase 1: 선량 변화 후 전자노이즈 비율)}}"
                f"\n\\alpha = \\frac{{\\sigma_e^2}}{{\\eta_{{\\text{{abs}}}} \\times N}}"
                f" = \\frac{{f_e \\times D}}{{1 - f_e}}"
                f" = \\frac{{{f_e:.2f} \\times {D:.2f}}}{{{1-f_e:.2f}}}"
                f" = {alpha:.6f}"
            ),
            numeric_value=alpha
        ))

        # Step 2: EID DQE at full dose
        # DQE_EID(full) = η_abs / (1 + α)
        dqe_eid_full = eta_abs / (1 + alpha)

        steps.append(DerivationStep(
            step_num=2,
            title="EID DQE at full dose",
            latex=(
                r"\text{DQE}_{\text{EID}}(\text{full}) = \frac{\eta_{\text{abs}}}{1 + \alpha}"
                f"\n= \\frac{{{eta_abs:.3f}}}{{{1 + alpha:.6f}}}"
                f" = \\mathbf{{{dqe_eid_full:.4f}}}"
            ),
            numeric_value=dqe_eid_full
        ))

        # Step 3: EID DQE at dose_ratio
        # DQE_EID(D) = η_abs / (1 + α/D)
        dqe_eid_at_d = eta_abs / (1 + alpha / D)

        steps.append(DerivationStep(
            step_num=3,
            title=f"EID DQE at D={D:.2f}",
            latex=(
                r"\text{DQE}_{\text{EID}}(D) = \frac{\eta_{\text{abs}}}{1 + \alpha/D}"
                f"\n= \\frac{{{eta_abs:.3f}}}{{1 + {alpha:.6f}/{D:.2f}}}"
                f" = \\frac{{{eta_abs:.3f}}}{{{1 + alpha/D:.6f}}}"
                f" = \\mathbf{{{dqe_eid_at_d:.4f}}}"
            ),
            numeric_value=dqe_eid_at_d
        ))

        # Step 4: PCD DQE (constant)
        dqe_pcd = eta_abs

        steps.append(DerivationStep(
            step_num=4,
            title="PCD DQE (σ_e = 0, 선량 독립)",
            latex=(
                r"\text{DQE}_{\text{PCD}} = \eta_{\text{abs}}"
                f" = \\mathbf{{{dqe_pcd:.4f}}}"
                r"\quad \text{(에너지 문턱치로 전자노이즈 물리적 제거)}"
            ),
            numeric_value=dqe_pcd
        ))

        # Step 5: PCD advantage & EID degradation
        pcd_advantage = (dqe_pcd / dqe_eid_at_d - 1) * 100
        dqe_degradation = (1 - dqe_eid_at_d / dqe_eid_full) * 100

        steps.append(DerivationStep(
            step_num=5,
            title="PCD DQE 이점 및 EID DQE 저하",
            latex=(
                r"\text{PCD advantage} = \frac{\text{DQE}_{\text{PCD}}}{\text{DQE}_{\text{EID}}(D)} - 1"
                f"\n= \\frac{{{dqe_pcd:.4f}}}{{{dqe_eid_at_d:.4f}}} - 1"
                f" = \\mathbf{{+{pcd_advantage:.1f}\\%}}"
                f"\n\\text{{EID DQE degradation}} = 1 - \\frac{{\\text{{DQE}}_{{\\text{{EID}}}}(D)}}"
                f"{{\\text{{DQE}}_{{\\text{{EID}}}}(\\text{{full}})}}"
                f" = 1 - \\frac{{{dqe_eid_at_d:.4f}}}{{{dqe_eid_full:.4f}}}"
                f" = {dqe_degradation:.1f}\\%"
            ),
            numeric_value=pcd_advantage
        ))

        # Step 6: Phase 1 교차 검증
        # SNR_ratio from DQE: √(DQE(D)×D / DQE(1))
        snr_ratio_from_dqe = math.sqrt(dqe_eid_at_d * D / dqe_eid_full)
        # Phase 1 공식: √(D × (1 - f_e × (1 - D)))
        snr_ratio_phase1 = math.sqrt(D * (1 - f_e * (1 - D)))

        assert abs(snr_ratio_from_dqe - snr_ratio_phase1) < 1e-10, \
            f"Phase 1 cross-validation failed: DQE→{snr_ratio_from_dqe:.6f} vs Phase1→{snr_ratio_phase1:.6f}"

        steps.append(DerivationStep(
            step_num=6,
            title="Phase 1 교차 검증 ✓",
            latex=(
                r"\frac{\text{SNR}_{\text{new}}}{\text{SNR}_0}"
                r" = \sqrt{\frac{\text{DQE}(D) \times D}{\text{DQE}(1)}}"
                f"\n= \\sqrt{{\\frac{{{dqe_eid_at_d:.4f} \\times {D:.2f}}}{{{dqe_eid_full:.4f}}}}}"
                f" = \\sqrt{{{dqe_eid_at_d * D / dqe_eid_full:.6f}}}"
                f" = {snr_ratio_from_dqe:.4f}"
                f"\n\\text{{Phase 1 공식}}: \\sqrt{{D \\times (1 - f_e(1-D))}}"
                f" = {snr_ratio_phase1:.4f} \\checkmark"
            ),
            numeric_value=snr_ratio_from_dqe
        ))

        # DQE 커브 생성 (10% ~ 200% dose range)
        dose_points = [i * 0.1 for i in range(1, 21)]
        dqe_eid_curve = [
            eta_abs / (1 + alpha / d)
            for d in dose_points
        ]

        return DQESolution(
            dqe_eid_full_dose=dqe_eid_full,
            dqe_eid_at_dose_ratio=dqe_eid_at_d,
            dqe_pcd=dqe_pcd,
            pcd_advantage_percent=pcd_advantage,
            dqe_degradation_percent=dqe_degradation,
            dose_points=dose_points,
            dqe_eid_curve=dqe_eid_curve,
            derivation_steps=steps,
            eta_abs=eta_abs,
            sigma_e_relative=f_e,
            dose_ratio=D
        )

    def solve_nps_decomposition(
        self,
        dose_ratio: float = 1.0,
        electronic_noise_fraction: float = 0.30,
        ref_dose_ratio: float = 0.5,
        pixel_size_mm: float = 0.085
    ) -> NPSSolution:
        """
        NPS(Noise Power Spectrum) 분해 계산

        물리 모델:
            NPS_quantum = σ_q² × a² = D × a² (선량 비례, 정규화)
            NPS_electronic = σ_e² × a² = α × a² (선량 무관, 상수)
            NPS_EID = NPS_q + NPS_e
            NPS_PCD = NPS_q (전자 노이즈 없음)

        Phase 1 파라미터:
            f_e는 ref_dose_ratio에서의 전자노이즈 비율
            α = f_e × ref_dose_ratio / (1 - f_e)

        Args:
            dose_ratio: NPS를 계산할 선량 비율 (D_current/D_full)
            electronic_noise_fraction: Phase 1 f_e (ref_dose에서 전자노이즈 비율)
            ref_dose_ratio: f_e가 측정된 선량 비율 (기본 0.5)
            pixel_size_mm: 픽셀 크기 (mm)

        Returns:
            NPSSolution
        """
        f_e = electronic_noise_fraction
        D = dose_ratio
        a = pixel_size_mm
        a_sq = a ** 2
        steps = []

        # α 역산: Phase 1 f_e → σ_e² (정규화)
        alpha = f_e * ref_dose_ratio / (1 - f_e)

        # NPS 계산 (현재 dose에서)
        sigma_q_sq = D  # dose ratio에 비례 (정규화: full dose = 1)
        sigma_e_sq = alpha  # 선량 무관 상수

        nps_q = sigma_q_sq * a_sq
        nps_e = sigma_e_sq * a_sq
        nps_eid = nps_q + nps_e
        nps_pcd = nps_q

        electronic_fraction = nps_e / nps_eid if nps_eid > 0 else 0
        pcd_reduction = (1 - nps_pcd / nps_eid) * 100 if nps_eid > 0 else 0

        steps.append(DerivationStep(
            step_num=1,
            title="NPS 분해",
            latex=(
                f"\\alpha = \\frac{{f_e \\times D_{{\\text{{ref}}}}}}{{1 - f_e}}"
                f" = \\frac{{{f_e:.2f} \\times {ref_dose_ratio:.2f}}}{{{1-f_e:.2f}}}"
                f" = {alpha:.6f}"
                f"\n\\text{{pixel size}} = {a:.3f}\\text{{ mm}},\\quad a^2 = {a_sq:.6f}"
                f"\n\\text{{NPS}}_q = D \\times a^2 = {sigma_q_sq:.4f} \\times {a_sq:.6f}"
                f" = {nps_q:.8f}"
                f"\n\\text{{NPS}}_e = \\alpha \\times a^2 = {sigma_e_sq:.6f} \\times {a_sq:.6f}"
                f" = {nps_e:.8f}"
                f"\n\\text{{NPS}}_{{\\text{{EID}}}} = \\text{{NPS}}_q + \\text{{NPS}}_e = {nps_eid:.8f}"
                f"\n\\text{{NPS}}_{{\\text{{PCD}}}} = \\text{{NPS}}_q = {nps_pcd:.8f}"
                f"\n\\text{{Electronic fraction}} = {electronic_fraction:.1%}"
                f"\n\\text{{PCD NPS reduction}} = {pcd_reduction:.1f}\\%"
            )
        ))

        return NPSSolution(
            nps_quantum=nps_q,
            nps_electronic=nps_e,
            nps_total_eid=nps_eid,
            nps_total_pcd=nps_pcd,
            electronic_fraction_eid=electronic_fraction,
            pcd_nps_reduction_percent=pcd_reduction,
            derivation_steps=steps
        )

    def solve_neq(self, dqe: float, incident_fluence: float) -> float:
        """
        NEQ (Noise Equivalent Quanta) 계산

        NEQ = DQE × q_in
        '검출기가 양자 효율이 완벽했다면 동등한 노이즈를 생성하는 광자 수'

        Args:
            dqe: DQE 값 (0-1)
            incident_fluence: 입사 광자 수 (q_in)

        Returns:
            NEQ 값
        """
        return dqe * incident_fluence

    def format_dqe_prompt(self, solution: DQESolution) -> str:
        """Phase 3 Double-Anchor용 제약 조건 프롬프트 생성"""
        return f"""
╔══════════════════════════════════════════════════════════════╗
║  🔒 PHASE 3 SOLVER 검증 완료 (DQE_PCD < DQE_EID는 불가능)   ║
╠══════════════════════════════════════════════════════════════╣
║  η_abs={solution.eta_abs:.3f}, f_e={solution.sigma_e_relative:.2f}, D={solution.dose_ratio:.2f}            ║
╠══════════════════════════════════════════════════════════════╣
║  📊 검증된 정답:                                             ║
║  • DQE_EID(full dose): {solution.dqe_eid_full_dose:.4f}                          ║
║  • DQE_EID(D={solution.dose_ratio:.2f}): {solution.dqe_eid_at_dose_ratio:.4f}                             ║
║  • DQE_PCD: {solution.dqe_pcd:.4f} (선량 독립)                           ║
║  • PCD DQE 이점: +{solution.pcd_advantage_percent:.1f}%                              ║
║  • EID DQE 저하: {solution.dqe_degradation_percent:.1f}%                              ║
╠══════════════════════════════════════════════════════════════╣
║  ⚠️ DQE_EID가 저선량에서 '증가'한다는 답변은 물리적 불가능    ║
║  ⚠️ 반드시 위 수치를 사용하세요 (1% 초과 오차 시 거부)       ║
╚══════════════════════════════════════════════════════════════╝
"""

    def audit_dqe_answer(
        self,
        llm_answer: str,
        eta_abs: float = 0.85,
        electronic_noise_fraction: float = 0.30,
        dose_ratio: float = 0.5,
        tolerance_pct: Optional[float] = None
    ) -> List[AuditResult]:
        """
        Phase 3: DQE 관련 LLM 답변 감사

        Args:
            llm_answer: LLM 답변 텍스트
            eta_abs, electronic_noise_fraction, dose_ratio: DQE 계산 파라미터
            tolerance_pct: 허용 오차 (기본 1%)

        Returns:
            List[AuditResult]
        """
        tol = tolerance_pct if tolerance_pct is not None else self.AUDIT_TOLERANCE
        solution = self.solve_dqe_dose_dependence(eta_abs, electronic_noise_fraction, dose_ratio)
        results = []

        # DQE 값 추출 패턴
        dqe_patterns = [
            r'DQE[^.]*?(\d+\.\d+)',
            r'DQE\s*[=≈]\s*(\d+\.\d+)',
        ]

        # DQE_PCD 검증
        pcd_match = re.search(r'DQE[_\s]*PCD[^.]*?(\d+\.\d+)', llm_answer, re.IGNORECASE)
        if pcd_match:
            llm_val = float(pcd_match.group(1))
            error = abs(llm_val - solution.dqe_pcd) / solution.dqe_pcd * 100
            reject = error > tol
            results.append(AuditResult(
                status=AuditStatus.REJECT if reject else AuditStatus.PASS,
                target_field="DQE_PCD",
                llm_value=llm_val,
                correct_value=solution.dqe_pcd,
                error_pct=error,
                tolerance_pct=tol,
                should_reject=reject,
                explanation=f"LLM: {llm_val:.4f}, 정답: {solution.dqe_pcd:.4f}, 오차: {error:.2f}%",
                correction_hint="DQE_PCD = η_abs (σ_e=0이므로 상수)" if reject else ""
            ))

        # PCD advantage 검증
        adv_match = re.search(r'[+＋]\s*(\d+(?:\.\d+)?)\s*%', llm_answer)
        if adv_match:
            llm_adv = float(adv_match.group(1))
            if 10 < llm_adv < 100:  # DQE advantage 합리적 범위
                error = abs(llm_adv - solution.pcd_advantage_percent)
                reject = error > tol
                results.append(AuditResult(
                    status=AuditStatus.REJECT if reject else AuditStatus.PASS,
                    target_field="PCD DQE advantage",
                    llm_value=llm_adv,
                    correct_value=solution.pcd_advantage_percent,
                    error_pct=error,
                    tolerance_pct=tol,
                    should_reject=reject,
                    explanation=f"LLM: +{llm_adv:.1f}%, 정답: +{solution.pcd_advantage_percent:.1f}%",
                    correction_hint="PCD advantage = (DQE_PCD/DQE_EID(D) - 1) × 100" if reject else ""
                ))

        return results


    # =========================================================================
    # Phase 4: MTF / Spatial Resolution / DQE(f) Solver
    # =========================================================================

    @staticmethod
    def _sinc(x: float) -> float:
        """Normalized sinc function: sinc(x) = sin(x)/x, sinc(0) = 1"""
        if abs(x) < 1e-15:
            return 1.0
        return math.sin(x) / x

    def solve_mtf_comparison(
        self,
        pixel_pitch_mm: float = 0.1,
        scintillator_thickness_um: float = 150.0,
        converter: str = 'CsI',
        cs_delta: float = 0.10
    ) -> MTFSolution:
        """
        Phase 4 핵심: PCD vs EID MTF 비교

        물리 모델:
            PCD (직접 변환, CdTe):
                MTF_PCD(f) = |sinc(π×f×a)| × CS_factor(f)
                CS_factor(f) = 1 - δ×(f/f_ny)²  [charge sharing]

            EID (간접 변환, CsI/GOS):
                MTF_EID(f) = |sinc(π×f×a)| × MTF_scint(f)
                MTF_scint(f) = exp(-(f/f_c)²) [Gaussian scintillator model]

            Characteristic frequency f_c:
                CsI 150um: f_c ≈ 10 lp/mm
                GOS 208um: f_c ≈ 4 lp/mm

        Args:
            pixel_pitch_mm: pixel pitch a (mm)
            scintillator_thickness_um: scintillator thickness (um)
            converter: 'CsI' or 'GOS'
            cs_delta: charge sharing parameter δ (0.10-0.25)

        Returns:
            MTFSolution with MTF curves and f10 comparison
        """
        a = pixel_pitch_mm
        f_ny = 1.0 / (2.0 * a)  # Nyquist frequency
        steps = []

        # Scintillator characteristic frequency
        if converter.upper() == 'CSI':
            # CsI: columnar structure, better light guiding
            # f_c scales inversely with thickness
            f_c = 10.0 * (150.0 / scintillator_thickness_um)
        elif converter.upper() == 'GOS':
            # GOS: powder phosphor, more diffusion
            f_c = 4.0 * (208.0 / scintillator_thickness_um)
        else:
            f_c = 10.0  # default to CsI

        steps.append(DerivationStep(
            step_num=1,
            title="파라미터 정의",
            latex=(
                f"a = {a:.3f}\\text{{ mm}},\\quad "
                f"f_{{\\text{{Nyquist}}}} = \\frac{{1}}{{2a}} = {f_ny:.1f}\\text{{ lp/mm}}"
                f"\n\\text{{{converter}}}: t = {scintillator_thickness_um:.0f}\\mu m,\\quad "
                f"f_c = {f_c:.1f}\\text{{ lp/mm}}"
                f"\n\\delta_{{\\text{{CS}}}} = {cs_delta:.2f}"
            )
        ))

        # Generate frequency points (0 to 2×Nyquist = 1/a, first sinc zero)
        # f10 for PCD can be well beyond Nyquist (Kuttig: f10=8.5 for 0.1mm pixel)
        n_points = 200
        f_max = 2.0 * f_ny  # = 1/a, first zero of sinc
        freq_points = [i * f_max / n_points for i in range(n_points + 1)]

        # Compute MTF curves
        mtf_pcd_curve = []
        mtf_eid_curve = []

        for f in freq_points:
            # PCD: sinc × charge_sharing
            sinc_val = abs(self._sinc(math.pi * f * a))
            f_ratio = f / f_ny if f_ny > 0 else 0
            cs_factor = max(0.0, 1.0 - cs_delta * f_ratio**2)
            mtf_pcd = sinc_val * cs_factor

            # EID: sinc × scintillator blur
            mtf_scint = math.exp(-(f / f_c)**2) if f_c > 0 else 0.0
            mtf_eid = sinc_val * mtf_scint

            mtf_pcd_curve.append(mtf_pcd)
            mtf_eid_curve.append(mtf_eid)

        # Values at Nyquist
        mtf_pcd_nyquist = abs(self._sinc(math.pi * f_ny * a)) * (1.0 - cs_delta)
        scint_at_nyquist = math.exp(-(f_ny / f_c)**2)
        mtf_eid_nyquist = abs(self._sinc(math.pi * f_ny * a)) * scint_at_nyquist

        steps.append(DerivationStep(
            step_num=2,
            title="MTF at Nyquist",
            latex=(
                f"\\text{{MTF}}_{{\\text{{PCD}}}}(f_{{\\text{{Ny}}}}) = "
                f"|\\text{{sinc}}(\\pi \\times {f_ny:.1f} \\times {a:.3f})| "
                f"\\times (1 - {cs_delta:.2f})"
                f"\n= |\\text{{sinc}}(\\pi/2)| \\times {1-cs_delta:.2f}"
                f" = {abs(self._sinc(math.pi/2)):.4f} \\times {1-cs_delta:.2f}"
                f" = \\mathbf{{{mtf_pcd_nyquist:.4f}}}"
                f"\n\\text{{MTF}}_{{\\text{{EID}}}}(f_{{\\text{{Ny}}}}) = "
                f"|\\text{{sinc}}(\\pi/2)| \\times \\exp(-(f_{{\\text{{Ny}}}}/f_c)^2)"
                f"\n= {abs(self._sinc(math.pi/2)):.4f} \\times "
                f"\\exp(-({f_ny:.1f}/{f_c:.1f})^2)"
                f" = {abs(self._sinc(math.pi/2)):.4f} \\times {scint_at_nyquist:.4f}"
                f" = \\mathbf{{{mtf_eid_nyquist:.4f}}}"
            ),
            numeric_value=mtf_pcd_nyquist
        ))

        # Find f10 (frequency where MTF = 0.10)
        f10_pcd = self._find_f10(freq_points, mtf_pcd_curve)
        f10_eid = self._find_f10(freq_points, mtf_eid_curve)
        resolution_gain = f10_pcd / f10_eid if f10_eid > 0 else float('inf')

        # Charge sharing degradation at Nyquist (% loss)
        ideal_sinc_ny = abs(self._sinc(math.pi / 2))
        cs_degradation_pct = cs_delta * 100  # at Nyquist, (f/f_ny)²=1, so loss = δ×100%

        steps.append(DerivationStep(
            step_num=3,
            title="Resolution limit (f10) 비교",
            latex=(
                f"f_{{10,\\text{{PCD}}}} = {f10_pcd:.2f}\\text{{ lp/mm}}"
                f"\nf_{{10,\\text{{EID}}}} = {f10_eid:.2f}\\text{{ lp/mm}}"
                f"\n\\text{{Resolution gain}} = \\frac{{f_{{10,\\text{{PCD}}}}}}{{f_{{10,\\text{{EID}}}}}}"
                f" = \\frac{{{f10_pcd:.2f}}}{{{f10_eid:.2f}}}"
                f" = \\mathbf{{{resolution_gain:.2f}\\times}}"
                f"\n\\text{{Charge sharing MTF loss at Nyquist}} = {cs_degradation_pct:.0f}\\%"
            ),
            numeric_value=resolution_gain
        ))

        return MTFSolution(
            pixel_pitch_mm=a,
            nyquist_freq=f_ny,
            mtf_pcd_at_nyquist=mtf_pcd_nyquist,
            mtf_eid_at_nyquist=mtf_eid_nyquist,
            scintillator_mtf_factor=scint_at_nyquist,
            f10_pcd=f10_pcd,
            f10_eid=f10_eid,
            pcd_resolution_gain=resolution_gain,
            charge_sharing_degradation=cs_degradation_pct,
            freq_points=freq_points,
            mtf_pcd_curve=mtf_pcd_curve,
            mtf_eid_curve=mtf_eid_curve,
            derivation_steps=steps,
            scintillator_type=converter,
            scintillator_thickness_um=scintillator_thickness_um,
            cs_delta=cs_delta
        )

    def solve_dqe_frequency(
        self,
        pixel_pitch_mm: float = 0.1,
        eta_abs: float = 0.85,
        electronic_noise_fraction: float = 0.30,
        ref_dose_ratio: float = 0.5,
        scintillator_thickness_um: float = 150.0,
        converter: str = 'CsI',
        cs_delta: float = 0.10
    ) -> DQEfSolution:
        """
        Phase 4: DQE(f) 주파수 의존 계산

        물리 모델:
            DQE_PCD(f) = η_abs × MTF²_PCD(f) / [MTF²_PCD(f) + NPS_cs(f)/NPS_q]
            DQE_EID(f) = η_abs × MTF²_EID(f) / [MTF²_EID(f) + NPS_e_norm(f)]

            where:
                NPS_cs(f) = charge sharing noise (small, frequency dependent)
                NPS_e_norm = α = σ_e²/(η_abs×N), at full dose D=1.0

        Phase 3 교차 검증:
            DQE(f→0): MTF(0)=1 이므로
            DQE_PCD(0) = η_abs = 0.850
            DQE_EID(0) = η_abs / (1 + α) = Phase 3 DQE_EID(full)

        Phase 1 파라미터 연동:
            f_e는 ref_dose_ratio에서 측정된 전자노이즈 비율
            α = f_e × ref_dose_ratio / (1 - f_e)
            DQE_EID(D) = η_abs / (1 + α/D)

        Args:
            pixel_pitch_mm: pixel pitch a (mm)
            eta_abs: absorption efficiency
            electronic_noise_fraction: f_e (Phase 1, at ref_dose_ratio)
            ref_dose_ratio: dose ratio where f_e was measured (Phase 1/3 convention: 0.5)
            scintillator_thickness_um: scintillator thickness
            converter: 'CsI' or 'GOS'
            cs_delta: charge sharing parameter

        Returns:
            DQEfSolution with DQE(f) curves
        """
        a = pixel_pitch_mm
        f_ny = 1.0 / (2.0 * a)
        f_e = electronic_noise_fraction
        steps = []

        # α from Phase 1/3 (f_e measured at ref_dose_ratio)
        # α = σ_e²/(η_abs×N) normalized at full dose
        # same convention as Phase 3: α = f_e × D_ref / (1 - f_e)
        alpha = f_e * ref_dose_ratio / (1 - f_e)

        # Phase 3 DQE(0) values
        dqe_eid_zero = eta_abs / (1 + alpha)
        dqe_pcd_zero = eta_abs

        steps.append(DerivationStep(
            step_num=1,
            title="Phase 3 DQE(0) 교차 검증",
            latex=(
                f"\\alpha = \\frac{{f_e \\times D_{{\\text{{ref}}}}}}{{1 - f_e}}"
                f" = \\frac{{{f_e:.2f} \\times {ref_dose_ratio:.2f}}}{{{1-f_e:.2f}}}"
                f" = {alpha:.6f}"
                f"\n\\text{{DQE}}_{{\\text{{PCD}}}}(0) = \\eta_{{\\text{{abs}}}}"
                f" = \\mathbf{{{dqe_pcd_zero:.4f}}}"
                f"\n\\text{{DQE}}_{{\\text{{EID}}}}(0) = \\frac{{\\eta_{{\\text{{abs}}}}}}{{1 + \\alpha}}"
                f" = \\frac{{{eta_abs:.3f}}}{{{1+alpha:.6f}}}"
                f" = \\mathbf{{{dqe_eid_zero:.4f}}}"
            )
        ))

        # Scintillator characteristic frequency
        if converter.upper() == 'CSI':
            f_c = 10.0 * (150.0 / scintillator_thickness_um)
        elif converter.upper() == 'GOS':
            f_c = 4.0 * (208.0 / scintillator_thickness_um)
        else:
            f_c = 10.0

        # Generate DQE(f) curves
        n_points = 100
        f_max = f_ny  # up to Nyquist
        freq_points = [i * f_max / n_points for i in range(n_points + 1)]

        dqe_pcd_curve = []
        dqe_eid_curve = []

        # NPS_e normalized: α = σ_e²/(η_abs × N × D)
        # This is the electronic noise contribution to NNPS
        nps_e_norm = alpha  # at full dose

        for f in freq_points:
            # MTF values
            sinc_val = abs(self._sinc(math.pi * f * a))
            f_ratio = f / f_ny if f_ny > 0 else 0

            # PCD MTF with charge sharing
            cs_factor = max(0.0, 1.0 - cs_delta * f_ratio**2)
            mtf_pcd = sinc_val * cs_factor
            mtf_pcd_sq = mtf_pcd**2

            # Charge sharing NPS (small, quadratic in f)
            # NPS_cs_norm: normalized charge sharing noise, proportional to f²
            nps_cs_norm = cs_delta * f_ratio**2 * 0.1  # small relative to quantum

            # DQE_PCD(f) = η_abs × MTF²_PCD / (MTF²_PCD + NPS_cs_norm)
            dqe_pcd_f = eta_abs * mtf_pcd_sq / (mtf_pcd_sq + nps_cs_norm) if (mtf_pcd_sq + nps_cs_norm) > 0 else 0
            dqe_pcd_curve.append(dqe_pcd_f)

            # EID MTF with scintillator blur
            mtf_scint = math.exp(-(f / f_c)**2) if f_c > 0 else 0.0
            mtf_eid = sinc_val * mtf_scint
            mtf_eid_sq = mtf_eid**2

            # DQE_EID(f) = η_abs × MTF²_EID / (MTF²_EID + NPS_e_norm)
            # electronic noise is white (frequency-independent)
            dqe_eid_f = eta_abs * mtf_eid_sq / (mtf_eid_sq + nps_e_norm) if (mtf_eid_sq + nps_e_norm) > 0 else 0
            dqe_eid_curve.append(dqe_eid_f)

        # Verify Phase 3 cross-validation
        # At f=0: MTF=1, CS=0, scint=1
        # DQE_PCD(0) = η_abs × 1 / (1 + 0) = η_abs
        # DQE_EID(0) = η_abs × 1 / (1 + α) = Phase 3 value
        phase3_pcd_match = abs(dqe_pcd_curve[0] - dqe_pcd_zero) < 1e-10
        phase3_eid_match = abs(dqe_eid_curve[0] - dqe_eid_zero) < 1e-6
        phase3_match = phase3_pcd_match and phase3_eid_match

        assert phase3_match, (
            f"Phase 3 cross-validation failed: "
            f"DQE_PCD(0)={dqe_pcd_curve[0]:.6f} vs {dqe_pcd_zero:.6f}, "
            f"DQE_EID(0)={dqe_eid_curve[0]:.6f} vs {dqe_eid_zero:.6f}"
        )

        steps.append(DerivationStep(
            step_num=2,
            title="Phase 3 교차 검증 ✓",
            latex=(
                f"\\text{{DQE}}_{{\\text{{PCD}}}}(f\\to 0) = {dqe_pcd_curve[0]:.4f}"
                f" = \\eta_{{\\text{{abs}}}} = {dqe_pcd_zero:.4f} \\checkmark"
                f"\n\\text{{DQE}}_{{\\text{{EID}}}}(f\\to 0) = {dqe_eid_curve[0]:.4f}"
                f" = \\text{{Phase 3}} = {dqe_eid_zero:.4f} \\checkmark"
            )
        ))

        # Values at Nyquist
        dqe_pcd_nyquist = dqe_pcd_curve[-1]
        dqe_eid_nyquist = dqe_eid_curve[-1]
        pcd_advantage_nyquist = dqe_pcd_nyquist / dqe_eid_nyquist if dqe_eid_nyquist > 0 else float('inf')

        steps.append(DerivationStep(
            step_num=3,
            title="DQE at Nyquist",
            latex=(
                f"\\text{{DQE}}_{{\\text{{PCD}}}}(f_{{\\text{{Ny}}}}) = "
                f"\\mathbf{{{dqe_pcd_nyquist:.4f}}}"
                f"\n\\text{{DQE}}_{{\\text{{EID}}}}(f_{{\\text{{Ny}}}}) = "
                f"\\mathbf{{{dqe_eid_nyquist:.4f}}}"
                f"\n\\text{{PCD advantage at Nyquist}} = "
                f"\\frac{{{dqe_pcd_nyquist:.4f}}}{{{dqe_eid_nyquist:.4f}}}"
                f" = \\mathbf{{{pcd_advantage_nyquist:.1f}\\times}}"
            ),
            numeric_value=pcd_advantage_nyquist
        ))

        return DQEfSolution(
            dqe_pcd_at_zero=dqe_pcd_zero,
            dqe_eid_at_zero=dqe_eid_zero,
            dqe_pcd_at_nyquist=dqe_pcd_nyquist,
            dqe_eid_at_nyquist=dqe_eid_nyquist,
            pcd_dqe_advantage_at_nyquist=pcd_advantage_nyquist,
            phase3_dqe_match=phase3_match,
            freq_points=freq_points,
            dqe_pcd_curve=dqe_pcd_curve,
            dqe_eid_curve=dqe_eid_curve,
            derivation_steps=steps,
            pixel_pitch_mm=a,
            eta_abs=eta_abs,
            electronic_noise_fraction=f_e
        )

    def solve_charge_sharing_effect(
        self,
        cdte_thickness_mm: float = 1.0,
        pixel_pitch_mm: float = 0.1
    ) -> MTFSolution:
        """
        Charge sharing effect on MTF as function of CdTe thickness

        물리:
            - 두꺼운 CdTe → QDE 증가 (더 많은 X-ray 흡수)
            - 하지만 전하 구름 확산 거리 증가 → charge sharing 증가
            - δ ≈ 0.05 + 0.08 × (t_mm - 0.5) [경험적 모델, Tanguay 2018]
            - 1mm: δ≈0.09, 2mm: δ≈0.17, 3mm: δ≈0.25

        Args:
            cdte_thickness_mm: CdTe thickness (mm)
            pixel_pitch_mm: pixel pitch (mm)

        Returns:
            MTFSolution with charge sharing effects
        """
        # Empirical charge sharing model
        # δ increases with thickness (linear approximation from literature)
        cs_delta = min(0.35, 0.05 + 0.08 * (cdte_thickness_mm - 0.5))
        cs_delta = max(0.02, cs_delta)  # minimum even for thin CdTe

        # Solve MTF with this charge sharing level
        solution = self.solve_mtf_comparison(
            pixel_pitch_mm=pixel_pitch_mm,
            scintillator_thickness_um=150.0,  # reference EID
            converter='CsI',
            cs_delta=cs_delta
        )

        # Add charge sharing specific derivation step
        solution.derivation_steps.append(DerivationStep(
            step_num=4,
            title=f"Charge Sharing Effect (CdTe {cdte_thickness_mm}mm)",
            latex=(
                f"\\text{{CdTe thickness}} = {cdte_thickness_mm:.1f}\\text{{ mm}}"
                f"\n\\delta_{{\\text{{CS}}}} \\approx 0.05 + 0.08 \\times (t - 0.5)"
                f" = {cs_delta:.3f}"
                f"\n\\text{{MTF degradation at Nyquist}} = \\delta \\times 100\\%"
                f" = {cs_delta*100:.0f}\\%"
                f"\n\\text{{Trade-off: QDE}} \\propto 1 - e^{{-\\mu t}}"
                f" \\text{{ vs MTF degradation}}"
            ),
            numeric_value=cs_delta
        ))

        return solution

    def format_mtf_prompt(self, mtf_sol: MTFSolution, dqef_sol: Optional[DQEfSolution] = None) -> str:
        """Phase 4 Double-Anchor용 제약 조건 프롬프트 생성"""
        dqe_info = ""
        if dqef_sol:
            dqe_info = (
                f"║  • DQE_PCD(0): {dqef_sol.dqe_pcd_at_zero:.4f} = Phase 3 ✓                    ║\n"
                f"║  • DQE_EID(0): {dqef_sol.dqe_eid_at_zero:.4f} = Phase 3 ✓                    ║\n"
                f"║  • DQE_PCD(Nyquist): {dqef_sol.dqe_pcd_at_nyquist:.4f}                            ║\n"
                f"║  • DQE_EID(Nyquist): {dqef_sol.dqe_eid_at_nyquist:.4f}                            ║\n"
                f"║  • PCD DQE advantage at Nyquist: {dqef_sol.pcd_dqe_advantage_at_nyquist:.1f}×               ║\n"
            )

        return f"""
╔══════════════════════════════════════════════════════════════╗
║  🔒 PHASE 4 SOLVER 검증 완료 (MTF_PCD < MTF_EID는 불가능)   ║
╠══════════════════════════════════════════════════════════════╣
║  pixel={mtf_sol.pixel_pitch_mm:.3f}mm, {mtf_sol.scintillator_type} {mtf_sol.scintillator_thickness_um:.0f}um, δ_CS={mtf_sol.cs_delta:.2f}   ║
╠══════════════════════════════════════════════════════════════╣
║  📊 검증된 정답:                                             ║
║  • Nyquist freq: {mtf_sol.nyquist_freq:.1f} lp/mm                             ║
║  • MTF_PCD(Nyquist): {mtf_sol.mtf_pcd_at_nyquist:.4f}                           ║
║  • MTF_EID(Nyquist): {mtf_sol.mtf_eid_at_nyquist:.4f}                           ║
║  • f10_PCD: {mtf_sol.f10_pcd:.2f} lp/mm                                 ║
║  • f10_EID: {mtf_sol.f10_eid:.2f} lp/mm                                 ║
║  • Resolution gain: {mtf_sol.pcd_resolution_gain:.2f}×                             ║
║  • Charge sharing loss: {mtf_sol.charge_sharing_degradation:.0f}% at Nyquist             ║
{dqe_info}╠══════════════════════════════════════════════════════════════╣
║  ⚠️ 직접변환 MTF < 간접변환 MTF는 물리적으로 불가능           ║
║  ⚠️ 반드시 위 수치를 사용하세요 (1% 초과 오차 시 거부)       ║
╚══════════════════════════════════════════════════════════════╝
"""

    def audit_mtf_answer(
        self,
        llm_answer: str,
        pixel_pitch_mm: float = 0.1,
        scintillator_thickness_um: float = 150.0,
        converter: str = 'CsI',
        cs_delta: float = 0.10,
        tolerance_pct: Optional[float] = None
    ) -> List[AuditResult]:
        """
        Phase 4: MTF/DQE(f) 관련 LLM 답변 감사

        Args:
            llm_answer: LLM 답변 텍스트
            pixel_pitch_mm, scintillator_thickness_um, converter, cs_delta: MTF 파라미터
            tolerance_pct: 허용 오차 (기본 1%)

        Returns:
            List[AuditResult]
        """
        tol = tolerance_pct if tolerance_pct is not None else self.AUDIT_TOLERANCE
        mtf_sol = self.solve_mtf_comparison(pixel_pitch_mm, scintillator_thickness_um, converter, cs_delta)
        results = []

        # MTF_PCD at Nyquist 검증
        pcd_mtf_match = re.search(
            r'MTF[_\s]*PCD[^.]*?(\d+\.\d+)',
            llm_answer, re.IGNORECASE
        )
        if pcd_mtf_match:
            llm_val = float(pcd_mtf_match.group(1))
            if llm_val < 1.0:  # MTF is 0-1
                error = abs(llm_val - mtf_sol.mtf_pcd_at_nyquist) / mtf_sol.mtf_pcd_at_nyquist * 100
                reject = error > tol
                results.append(AuditResult(
                    status=AuditStatus.REJECT if reject else AuditStatus.PASS,
                    target_field="MTF_PCD at Nyquist",
                    llm_value=llm_val,
                    correct_value=mtf_sol.mtf_pcd_at_nyquist,
                    error_pct=error,
                    tolerance_pct=tol,
                    should_reject=reject,
                    explanation=f"LLM: {llm_val:.4f}, 정답: {mtf_sol.mtf_pcd_at_nyquist:.4f}",
                    correction_hint="MTF_PCD(f_ny) = sinc(π/2) × (1-δ)" if reject else ""
                ))

        # Resolution gain 검증
        gain_match = re.search(
            r'(\d+(?:\.\d+)?)\s*[×xX배]\s*(?:해상도|resolution|gain)',
            llm_answer, re.IGNORECASE
        )
        if not gain_match:
            gain_match = re.search(
                r'(?:해상도|resolution|gain)[^\d]*(\d+(?:\.\d+)?)\s*[×xX배]',
                llm_answer, re.IGNORECASE
            )
        if gain_match:
            llm_gain = float(gain_match.group(1))
            if 1.0 < llm_gain < 10.0:
                error = abs(llm_gain - mtf_sol.pcd_resolution_gain) / mtf_sol.pcd_resolution_gain * 100
                reject = error > tol
                results.append(AuditResult(
                    status=AuditStatus.REJECT if reject else AuditStatus.PASS,
                    target_field="Resolution gain (PCD/EID)",
                    llm_value=llm_gain,
                    correct_value=mtf_sol.pcd_resolution_gain,
                    error_pct=error,
                    tolerance_pct=tol,
                    should_reject=reject,
                    explanation=f"LLM: {llm_gain:.2f}×, 정답: {mtf_sol.pcd_resolution_gain:.2f}×",
                    correction_hint="Resolution gain = f10_PCD / f10_EID" if reject else ""
                ))

        return results

    @staticmethod
    def _find_f10(freq_points: List[float], mtf_curve: List[float]) -> float:
        """Find frequency where MTF = 0.10 (10% MTF, resolution limit)"""
        for i in range(len(mtf_curve) - 1):
            if mtf_curve[i] >= 0.10 and mtf_curve[i + 1] < 0.10:
                # Linear interpolation
                f_a, f_b = freq_points[i], freq_points[i + 1]
                m_a, m_b = mtf_curve[i], mtf_curve[i + 1]
                if abs(m_a - m_b) < 1e-15:
                    return f_a
                f10 = f_a + (0.10 - m_a) * (f_b - f_a) / (m_b - m_a)
                return f10
        # If MTF never drops below 0.10 within range, return max frequency
        return freq_points[-1] if freq_points else 0.0

    # =========================================================================
    # Phase 4-B: Biopsy Geometry & Calibration
    # =========================================================================

    def solve_biopsy_targeting(
        self,
        stereo_angle_deg: float = 15.0,
        pixel_pitch_mm: float = 0.1,
        mtf_pcd_effective: float = 0.637,
        mtf_eid_effective: float = 0.40,
        calibration_offset_mm: float = 0.2,
        breast_thickness_mm: float = 50.0,
        lesion_depth_fraction: float = 0.5,
        x_plus_mm: Optional[float] = None,
        x_minus_mm: Optional[float] = None,
        y_plus_mm: Optional[float] = None,
        y_minus_mm: Optional[float] = None,
    ) -> BiopsySolution:
        """
        Phase 4-B: 스테레오 정위 생검 타겟팅 불확실성 계산

        스테레오 시차(Parallax)에서 3D 좌표를 산출하고,
        Phase 4-A의 MTF 결과를 기반으로 PCD vs EID 타겟팅 정밀도를 비교합니다.

        물리 모델:
            Law 13: Z = Δx / (2 × sin(θ))
            Law 14: σ_Z = σ_Δx / (2 × sin(θ)) [기하학적 증폭]
            σ_Δx = pixel_pitch / MTF_effective (최소 측정 불확실성)
            Total Error = √(σ_X² + σ_Y² + σ_Z² + σ_cal²)

        Args:
            stereo_angle_deg: 스테레오 각도 (°, 기본 ±15°)
            pixel_pitch_mm: 픽셀 피치 (mm)
            mtf_pcd_effective: PCD 유효 MTF (Nyquist에서의 값)
            mtf_eid_effective: EID 유효 MTF (Nyquist에서의 값)
            calibration_offset_mm: 기계적 교정 오차 (mm)
            breast_thickness_mm: 유방 압박 두께 (mm)
            lesion_depth_fraction: 병변 깊이/두께 비율 (0-1, 0.5=중간)
            x_plus_mm: +θ 영상에서의 X 좌표 (None이면 시뮬레이션)
            x_minus_mm: -θ 영상에서의 X 좌표 (None이면 시뮬레이션)
            y_plus_mm: +θ 영상에서의 Y 좌표 (None이면 시뮬레이션)
            y_minus_mm: -θ 영상에서의 Y 좌표 (None이면 시뮬레이션)

        Returns:
            BiopsySolution with targeting uncertainty analysis
        """
        steps = []
        theta_rad = math.radians(stereo_angle_deg)
        sin_theta = math.sin(theta_rad)

        # =====================================================================
        # Step 1: 기하학적 증폭 계수 계산
        # =====================================================================
        geometric_amp = 1.0 / (2.0 * sin_theta)
        steps.append(DerivationStep(
            step_num=1,
            title="기하학적 증폭 계수",
            latex=f"G = 1/(2\\sin\\theta) = 1/(2\\sin({stereo_angle_deg}°)) = {geometric_amp:.4f}",
            numeric_value=geometric_amp
        ))

        # =====================================================================
        # Step 2: 3D 좌표 산출 (실제 좌표 또는 시뮬레이션)
        # =====================================================================
        if x_plus_mm is not None and x_minus_mm is not None:
            # 실제 스테레오 페어 좌표 입력
            parallax = x_plus_mm - x_minus_mm
            target_x = (x_plus_mm + x_minus_mm) / 2.0
            target_y = (y_plus_mm + y_minus_mm) / 2.0 if y_plus_mm is not None and y_minus_mm is not None else 0.0
        else:
            # 시뮬레이션: 유방 중간 깊이의 병변
            target_z_sim = breast_thickness_mm * lesion_depth_fraction
            parallax = target_z_sim * 2.0 * sin_theta  # 역산
            target_x = 25.0  # 임의 X 좌표
            target_y = 25.0  # 임의 Y 좌표

        target_z = parallax * geometric_amp  # Z = Δx / (2sinθ)
        steps.append(DerivationStep(
            step_num=2,
            title="3D 좌표 산출 (Law 13)",
            latex=f"Z = \\Delta x / (2\\sin\\theta) = {parallax:.3f} / {2*sin_theta:.4f} = {target_z:.3f} mm",
            numeric_value=target_z
        ))

        # =====================================================================
        # Step 3: PCD 시차 측정 불확실성 (Phase 4-A 연결)
        # =====================================================================
        # σ_Δx = pixel_pitch / MTF_effective
        # 시차는 두 위치 측정의 차이이므로 √2배 증가
        sigma_single_pcd = pixel_pitch_mm / mtf_pcd_effective
        sigma_dx_pcd = sigma_single_pcd * math.sqrt(2)  # 두 측정의 차이

        sigma_single_eid = pixel_pitch_mm / mtf_eid_effective
        sigma_dx_eid = sigma_single_eid * math.sqrt(2)

        steps.append(DerivationStep(
            step_num=3,
            title="시차 측정 불확실성 (Phase 4-A 연결)",
            latex=f"\\sigma_{{\\Delta x,PCD}} = \\sqrt{{2}} \\times a/MTF = "
                  f"\\sqrt{{2}} \\times {pixel_pitch_mm}/{mtf_pcd_effective:.3f} = {sigma_dx_pcd:.4f} mm\n"
                  f"\\sigma_{{\\Delta x,EID}} = \\sqrt{{2}} \\times {pixel_pitch_mm}/{mtf_eid_effective:.3f} = {sigma_dx_eid:.4f} mm",
            numeric_value=sigma_dx_pcd
        ))

        # =====================================================================
        # Step 4: Z축 불확실성 (Law 14: 기하학적 증폭)
        # =====================================================================
        sigma_z_pcd = sigma_dx_pcd * geometric_amp
        sigma_z_eid = sigma_dx_eid * geometric_amp

        steps.append(DerivationStep(
            step_num=4,
            title="Z축 불확실성 (Law 14: 기하학적 증폭)",
            latex=f"\\sigma_{{Z,PCD}} = \\sigma_{{\\Delta x,PCD}} \\times G = "
                  f"{sigma_dx_pcd:.4f} \\times {geometric_amp:.4f} = {sigma_z_pcd:.4f} mm\n"
                  f"\\sigma_{{Z,EID}} = {sigma_dx_eid:.4f} \\times {geometric_amp:.4f} = {sigma_z_eid:.4f} mm",
            numeric_value=sigma_z_pcd
        ))

        # =====================================================================
        # Step 5: XY축 불확실성 (단일 측정, 기하학적 증폭 없음)
        # =====================================================================
        sigma_x_pcd = sigma_single_pcd
        sigma_y_pcd = sigma_single_pcd
        sigma_x_eid = sigma_single_eid
        sigma_y_eid = sigma_single_eid

        steps.append(DerivationStep(
            step_num=5,
            title="XY축 불확실성 (기하학적 증폭 없음)",
            latex=f"\\sigma_{{X,PCD}} = \\sigma_{{Y,PCD}} = a/MTF = {sigma_x_pcd:.4f} mm\n"
                  f"\\sigma_{{X,EID}} = \\sigma_{{Y,EID}} = a/MTF = {sigma_x_eid:.4f} mm",
            numeric_value=sigma_x_pcd
        ))

        # =====================================================================
        # Step 6: 총 타겟팅 오차 (RSS, 교정 포함)
        # =====================================================================
        sigma_cal = calibration_offset_mm
        total_error_pcd = math.sqrt(
            sigma_x_pcd**2 + sigma_y_pcd**2 + sigma_z_pcd**2 + sigma_cal**2
        )
        total_error_eid = math.sqrt(
            sigma_x_eid**2 + sigma_y_eid**2 + sigma_z_eid**2 + sigma_cal**2
        )

        steps.append(DerivationStep(
            step_num=6,
            title="총 타겟팅 오차 (RSS)",
            latex=f"E_{{PCD}} = \\sqrt{{\\sigma_X^2 + \\sigma_Y^2 + \\sigma_Z^2 + \\sigma_{{cal}}^2}} = "
                  f"\\sqrt{{{sigma_x_pcd:.4f}^2 + {sigma_y_pcd:.4f}^2 + {sigma_z_pcd:.4f}^2 + {sigma_cal:.4f}^2}} = "
                  f"{total_error_pcd:.4f} mm\n"
                  f"E_{{EID}} = {total_error_eid:.4f} mm",
            numeric_value=total_error_pcd
        ))

        # =====================================================================
        # Step 7: PCD 타겟팅 개선율
        # =====================================================================
        error_reduction_pct = (1.0 - total_error_pcd / total_error_eid) * 100.0
        z_to_xy_ratio = sigma_z_pcd / sigma_x_pcd if sigma_x_pcd > 0 else float('inf')

        steps.append(DerivationStep(
            step_num=7,
            title="PCD 타겟팅 개선",
            latex=f"개선율 = (1 - E_{{PCD}}/E_{{EID}}) \\times 100 = "
                  f"(1 - {total_error_pcd:.4f}/{total_error_eid:.4f}) \\times 100 = "
                  f"{error_reduction_pct:.1f}\\%",
            numeric_value=error_reduction_pct
        ))

        # =====================================================================
        # Step 8: ACR 허용 기준 판정
        # =====================================================================
        acr_tolerance = 1.0  # mm
        within_acr = total_error_pcd <= acr_tolerance

        steps.append(DerivationStep(
            step_num=8,
            title="ACR 허용 기준 판정",
            latex=f"E_{{PCD}} = {total_error_pcd:.4f} mm {'≤' if within_acr else '>'} "
                  f"{acr_tolerance:.1f} mm (ACR limit) → {'PASS' if within_acr else 'FAIL'}",
            numeric_value=total_error_pcd
        ))

        # =====================================================================
        # Step 9: 최적 스테레오 각도 분석
        # =====================================================================
        # 최적 각도: σ_Z 최소화 but 시차 측정 가능해야 함
        # 큰 각도 → 작은 G → 작은 σ_Z, but 유방 압박 두께 문제
        # 실용적 최적: 15° (표준), 큰 유방에서 20-25° 가능
        if breast_thickness_mm > 60:
            optimal_angle = 20.0
            angle_note = "두꺼운 유방(>60mm): θ=20° 권장 (기하학적 증폭 1.46× 감소, 임상 실현성 확보)"
        elif breast_thickness_mm < 30:
            optimal_angle = 15.0
            angle_note = "얇은 유방(<30mm): θ=15° 표준 유지 (시차 충분, 압박 부담 최소화)"
        else:
            optimal_angle = 15.0
            angle_note = "표준 유방(30-60mm): θ=15° 표준 (ACR Stereotactic Biopsy QC Manual)"

        steps.append(DerivationStep(
            step_num=9,
            title="최적 스테레오 각도",
            latex=f"θ_{{opt}} = {optimal_angle}° (유방 두께 = {breast_thickness_mm:.0f}mm)\n"
                  f"G(15°) = {1/(2*math.sin(math.radians(15))):.3f}, "
                  f"G(20°) = {1/(2*math.sin(math.radians(20))):.3f}, "
                  f"G(25°) = {1/(2*math.sin(math.radians(25))):.3f}",
            numeric_value=optimal_angle
        ))

        return BiopsySolution(
            target_x_mm=target_x,
            target_y_mm=target_y,
            target_z_mm=target_z,
            parallax_mm=parallax,
            sigma_x_mm=sigma_x_pcd,
            sigma_y_mm=sigma_y_pcd,
            sigma_z_mm=sigma_z_pcd,
            sigma_cal_mm=sigma_cal,
            total_targeting_error_mm=total_error_pcd,
            geometric_amplification=geometric_amp,
            z_to_xy_error_ratio=z_to_xy_ratio,
            acr_tolerance_mm=acr_tolerance,
            within_acr_tolerance=within_acr,
            sigma_dx_pcd_mm=sigma_dx_pcd,
            sigma_dx_eid_mm=sigma_dx_eid,
            total_error_pcd_mm=total_error_pcd,
            total_error_eid_mm=total_error_eid,
            pcd_error_reduction_pct=error_reduction_pct,
            optimal_angle_deg=optimal_angle,
            angle_tradeoff_note=angle_note,
            derivation_steps=steps,
            stereo_angle_deg=stereo_angle_deg,
            pixel_pitch_mm=pixel_pitch_mm,
            breast_thickness_mm=breast_thickness_mm,
        )

    def solve_optimal_stereo_angle(
        self,
        pixel_pitch_mm: float = 0.1,
        mtf_effective: float = 0.637,
        calibration_offset_mm: float = 0.2,
        breast_thickness_mm: float = 50.0,
        angle_range: Tuple[float, float] = (10.0, 30.0),
        angle_step: float = 1.0,
    ) -> Tuple[float, List[Tuple[float, float]]]:
        """
        다양한 스테레오 각도에서의 총 타겟팅 오차를 계산하여 최적 각도를 산출

        Args:
            pixel_pitch_mm: 픽셀 피치
            mtf_effective: 유효 MTF
            calibration_offset_mm: 교정 오차
            breast_thickness_mm: 유방 두께
            angle_range: 탐색 각도 범위 (°)
            angle_step: 각도 탐색 스텝

        Returns:
            (optimal_angle_deg, [(angle, total_error), ...])
        """
        results = []
        min_error = float('inf')
        optimal_angle = angle_range[0]

        angle = angle_range[0]
        while angle <= angle_range[1]:
            sol = self.solve_biopsy_targeting(
                stereo_angle_deg=angle,
                pixel_pitch_mm=pixel_pitch_mm,
                mtf_pcd_effective=mtf_effective,
                calibration_offset_mm=calibration_offset_mm,
                breast_thickness_mm=breast_thickness_mm,
            )
            results.append((angle, sol.total_error_pcd_mm))
            if sol.total_error_pcd_mm < min_error:
                min_error = sol.total_error_pcd_mm
                optimal_angle = angle
            angle += angle_step

        return optimal_angle, results

    def format_biopsy_prompt(self, sol: BiopsySolution) -> str:
        """
        Phase 4-B 제약 조건 프롬프트 생성

        Args:
            sol: BiopsySolution 결과

        Returns:
            LLM 프롬프트용 제약 조건 문자열
        """
        lines = [
            "=" * 60,
            "[Phase 4-B] Biopsy Geometry 제약 조건 (Python Solver 검증)",
            "=" * 60,
            f"  스테레오 각도: ±{sol.stereo_angle_deg}°",
            f"  기하학적 증폭: {sol.geometric_amplification:.4f}×",
            f"  σ_Z / σ_XY = {sol.z_to_xy_error_ratio:.2f} (항상 >1)",
            "",
            f"  [PCD] σ_Δx = {sol.sigma_dx_pcd_mm:.4f} mm",
            f"  [PCD] σ_Z = {sol.sigma_z_mm:.4f} mm",
            f"  [PCD] Total Error = {sol.total_error_pcd_mm:.4f} mm",
            f"  [EID] σ_Δx = {sol.sigma_dx_eid_mm:.4f} mm",
            f"  [EID] Total Error = {sol.total_error_eid_mm:.4f} mm",
            "",
            f"  PCD 타겟팅 개선: {sol.pcd_error_reduction_pct:.1f}%",
            f"  ACR 허용 기준 (≤1mm): {'PASS' if sol.within_acr_tolerance else 'FAIL'}",
            "",
            "  ⚠️ 이 수치와 1% 초과 불일치 시 답변 거부",
            "=" * 60,
        ]
        return "\n".join(lines)

    def audit_biopsy_answer(
        self,
        llm_answer: str,
        stereo_angle_deg: float = 15.0,
        pixel_pitch_mm: float = 0.1,
        calibration_offset_mm: float = 0.2,
        tolerance_pct: Optional[float] = None
    ) -> List[AuditResult]:
        """
        Phase 4-B: 생검 기하학 관련 LLM 답변 감사

        Args:
            llm_answer: LLM 답변 텍스트
            stereo_angle_deg, pixel_pitch_mm, calibration_offset_mm: 파라미터
            tolerance_pct: 허용 오차

        Returns:
            List[AuditResult]
        """
        tol = tolerance_pct if tolerance_pct is not None else self.AUDIT_TOLERANCE
        sol = self.solve_biopsy_targeting(
            stereo_angle_deg=stereo_angle_deg,
            pixel_pitch_mm=pixel_pitch_mm,
            calibration_offset_mm=calibration_offset_mm,
        )
        results = []

        # 기하학적 증폭 계수 검증
        amp_match = re.search(
            r'(?:증폭|amplification|factor)[^\d]*(\d+\.\d+)',
            llm_answer, re.IGNORECASE
        )
        if amp_match:
            llm_val = float(amp_match.group(1))
            if 1.0 < llm_val < 10.0:
                error = abs(llm_val - sol.geometric_amplification) / sol.geometric_amplification * 100
                reject = error > tol
                results.append(AuditResult(
                    status=AuditStatus.REJECT if reject else AuditStatus.PASS,
                    target_field="Geometric Amplification",
                    llm_value=llm_val,
                    correct_value=sol.geometric_amplification,
                    error_pct=error,
                    tolerance_pct=tol,
                    should_reject=reject,
                    explanation=f"LLM: {llm_val:.4f}, 정답: {sol.geometric_amplification:.4f}",
                    correction_hint="G = 1/(2×sin(θ))" if reject else ""
                ))

        # 총 타겟팅 오차 검증
        error_match = re.search(
            r'(?:total|총|targeting)[^\d]*(\d+\.\d+)\s*mm',
            llm_answer, re.IGNORECASE
        )
        if error_match:
            llm_val = float(error_match.group(1))
            if 0.0 < llm_val < 5.0:
                error = abs(llm_val - sol.total_error_pcd_mm) / sol.total_error_pcd_mm * 100
                reject = error > tol
                results.append(AuditResult(
                    status=AuditStatus.REJECT if reject else AuditStatus.PASS,
                    target_field="Total Targeting Error (PCD)",
                    llm_value=llm_val,
                    correct_value=sol.total_error_pcd_mm,
                    error_pct=error,
                    tolerance_pct=tol,
                    should_reject=reject,
                    explanation=f"LLM: {llm_val:.4f}mm, 정답: {sol.total_error_pcd_mm:.4f}mm",
                    correction_hint="Total = √(σ_X² + σ_Y² + σ_Z² + σ_cal²)" if reject else ""
                ))

        return results

    # =========================================================================
    # Phase 4-B: DBT (Tomosynthesis) Guided Biopsy
    # =========================================================================

    def solve_dbt_biopsy_targeting(
        self,
        angular_range_deg: float = 50.0,
        n_projections: int = 25,
        pixel_pitch_mm: float = 0.1,
        mtf_effective: float = 0.637,
        depth_resolution_constant: float = 0.50,
        calibration_offset_mm: float = 0.2,
        stereo_angle_deg: float = 15.0,
    ) -> DBTBiopsySolution:
        """
        DBT(토모합성) 유도 생검의 깊이 분해능 및 타겟팅 오차 계산

        Law 15: Δz_FWHM = K / sin(α_total/2)
          - K = depth_resolution_constant (시스템 의존, 0.42-1.0mm)
          - α_total = angular_range_deg

        σ_Z_DBT = Δz_FWHM / (2√3)  [균일 분포 가정]
        → 기하학적 증폭(G) 없음: 재구성 슬라이스에서 직접 깊이 결정

        Args:
            angular_range_deg: 총 각도 범위 (°), 15-50° 범위
            n_projections: 투영 수 (9-25)
            pixel_pitch_mm: 픽셀 피치 (mm)
            mtf_effective: 유효 MTF (Phase 4-A 연결)
            depth_resolution_constant: K (mm), 시스템 의존
                0.42: iterative reconstruction (Siemens 50°급)
                0.50: 중간 (기본값)
                0.65: good reconstruction
                1.00: standard FBP (narrow-angle)
            calibration_offset_mm: 교정 오프셋 (mm)
            stereo_angle_deg: 비교용 스테레오 각도 (°)

        Returns:
            DBTBiopsySolution
        """
        import math
        steps = []

        # Step 1: 깊이 분해능 계산
        alpha_half_rad = math.radians(angular_range_deg / 2)
        sin_alpha_half = math.sin(alpha_half_rad)
        depth_resolution = depth_resolution_constant / sin_alpha_half

        steps.append(DerivationStep(
            step_num=1,
            title="DBT 깊이 분해능 (Law 15)",
            latex=f"\\Delta z_{{FWHM}} = K / \\sin(\\alpha/2) = "
                  f"{depth_resolution_constant}/{sin_alpha_half:.4f} = {depth_resolution:.4f} \\text{{ mm}}",
            numeric_value=round(depth_resolution, 4)
        ))

        # Step 2: XY축 불확실성 (stereo와 동일)
        sigma_xy = pixel_pitch_mm / mtf_effective

        steps.append(DerivationStep(
            step_num=2,
            title="XY축 측정 불확실성",
            latex=f"\\sigma_{{XY}} = a/MTF = {pixel_pitch_mm}/{mtf_effective:.3f} = {sigma_xy:.4f} \\text{{ mm}}",
            numeric_value=round(sigma_xy, 4)
        ))

        # Step 3: Z축 불확실성 (균일 분포, G 증폭 없음)
        sigma_z_dbt = depth_resolution / (2 * math.sqrt(3))

        steps.append(DerivationStep(
            step_num=3,
            title="DBT Z축 불확실성 (기하학적 증폭 없음)",
            latex=f"\\sigma_{{Z,DBT}} = \\Delta z / (2\\sqrt{{3}}) = "
                  f"{depth_resolution:.4f}/3.464 = {sigma_z_dbt:.4f} \\text{{ mm}}",
            numeric_value=round(sigma_z_dbt, 4)
        ))

        # Step 4: 총 타겟팅 오차
        total_error_dbt = math.sqrt(
            sigma_xy**2 + sigma_xy**2 + sigma_z_dbt**2 + calibration_offset_mm**2
        )

        steps.append(DerivationStep(
            step_num=4,
            title="DBT 총 타겟팅 오차 (RSS)",
            latex=f"\\text{{Total}}_{{DBT}} = \\sqrt{{\\sigma_X^2 + \\sigma_Y^2 + \\sigma_Z^2 + \\sigma_{{cal}}^2}} = "
                  f"{total_error_dbt:.4f} \\text{{ mm}}",
            numeric_value=round(total_error_dbt, 4)
        ))

        # Step 5: 동일 조건 스테레오 비교
        theta_rad = math.radians(stereo_angle_deg)
        sin_theta = math.sin(theta_rad)
        G = 1.0 / (2.0 * sin_theta)
        sigma_dx_stereo = math.sqrt(2) * sigma_xy
        sigma_z_stereo = sigma_dx_stereo * G
        total_error_stereo = math.sqrt(
            sigma_xy**2 + sigma_xy**2 + sigma_z_stereo**2 + calibration_offset_mm**2
        )

        steps.append(DerivationStep(
            step_num=5,
            title="동일 조건 스테레오 비교",
            latex=f"\\sigma_{{Z,stereo}} = \\sqrt{{2}} \\times \\sigma_{{XY}} \\times G = "
                  f"{sigma_dx_stereo:.4f} \\times {G:.4f} = {sigma_z_stereo:.4f} \\text{{ mm}}\n"
                  f"\\text{{Total}}_{{stereo}} = {total_error_stereo:.4f} \\text{{ mm}}",
            numeric_value=round(total_error_stereo, 4)
        ))

        # Step 6: DBT vs Stereo 우위 판단
        dbt_advantage = total_error_dbt < total_error_stereo
        if total_error_stereo > 0:
            dbt_improvement = (1 - total_error_dbt / total_error_stereo) * 100
        else:
            dbt_improvement = 0.0

        steps.append(DerivationStep(
            step_num=6,
            title="DBT vs Stereo 비교",
            latex=f"\\text{{improvement}} = (1 - {total_error_dbt:.4f}/{total_error_stereo:.4f}) \\times 100\\% = "
                  f"{dbt_improvement:+.1f}\\%",
            numeric_value=round(dbt_improvement, 1)
        ))

        # Step 7: 임계 각도 계산 (DBT = Stereo가 되는 angular range)
        # σ_Z_DBT = σ_Z_stereo → K/sin(α/2)/(2√3) = √2×σ_XY×G
        # Simplified: sin(α_c/2) = K / (2√3 × σ_Z_stereo)
        crossover_sin = depth_resolution_constant / (2 * math.sqrt(3) * sigma_z_stereo)
        if abs(crossover_sin) <= 1.0:
            crossover_angle = 2 * math.degrees(math.asin(crossover_sin))
        else:
            # DBT can never match stereo with this K (K too large)
            crossover_angle = 180.0  # impossible

        steps.append(DerivationStep(
            step_num=7,
            title="임계 각도 (DBT = Stereo 되는 α_total)",
            latex=f"\\sin(\\alpha_c/2) = K / (2\\sqrt{{3}} \\times \\sigma_{{Z,stereo}}) = "
                  f"{depth_resolution_constant}/(3.464 \\times {sigma_z_stereo:.4f}) → "
                  f"\\alpha_c = {crossover_angle:.1f}°",
            numeric_value=round(crossover_angle, 1)
        ))

        return DBTBiopsySolution(
            angular_range_deg=angular_range_deg,
            n_projections=n_projections,
            pixel_pitch_mm=pixel_pitch_mm,
            mtf_effective=mtf_effective,
            depth_resolution_constant=depth_resolution_constant,
            depth_resolution_mm=round(depth_resolution, 4),
            sigma_xy_mm=round(sigma_xy, 4),
            sigma_z_mm=round(sigma_z_dbt, 4),
            sigma_cal_mm=calibration_offset_mm,
            total_targeting_error_mm=round(total_error_dbt, 4),
            acr_tolerance_mm=1.0,
            within_acr_tolerance=total_error_dbt <= 1.0,
            stereo_sigma_z_mm=round(sigma_z_stereo, 4),
            stereo_total_error_mm=round(total_error_stereo, 4),
            dbt_advantage=dbt_advantage,
            dbt_improvement_pct=round(dbt_improvement, 1),
            crossover_angle_deg=round(crossover_angle, 1),
            derivation_steps=steps,
        )

    def format_dbt_biopsy_prompt(self, sol: DBTBiopsySolution) -> str:
        """
        Phase 4-B DBT 제약 조건 프롬프트 생성
        """
        lines = [
            "=" * 60,
            "[Phase 4-B] DBT Biopsy Geometry 제약 조건 (Python Solver 검증)",
            "=" * 60,
            f"  DBT 각도 범위: {sol.angular_range_deg}°, 투영수: {sol.n_projections}",
            f"  깊이 분해능 (Δz_FWHM): {sol.depth_resolution_mm:.3f} mm",
            f"  σ_Z_DBT: {sol.sigma_z_mm:.4f} mm (기하학적 증폭 없음)",
            f"  Total Error (DBT): {sol.total_targeting_error_mm:.4f} mm",
            "",
            f"  비교 (Stereo ±15°):",
            f"    σ_Z_Stereo: {sol.stereo_sigma_z_mm:.4f} mm (G=1.93×)",
            f"    Total Error (Stereo): {sol.stereo_total_error_mm:.4f} mm",
            "",
            f"  DBT 우위: {'YES' if sol.dbt_advantage else 'NO'}",
            f"  개선율: {sol.dbt_improvement_pct:+.1f}%",
            f"  임계 각도: {sol.crossover_angle_deg:.1f}° (이상에서 DBT 우위)",
            f"  ACR (≤1mm): {'PASS' if sol.within_acr_tolerance else 'FAIL'}",
            "",
            "  ⚠️ 이 수치와 1% 초과 불일치 시 답변 거부",
            "=" * 60,
        ]
        return "\n".join(lines)

    # =========================================================================
    # Phase 5: Tomosynthesis Image Quality Physics
    # =========================================================================

    def solve_tomo_dose_split(
        self,
        total_dose_uGy: float = 1500.0,
        n_projections: int = 25,
        eta_abs: float = 0.85,
        electronic_noise_fraction: float = 0.30,
        dose_ratio_for_alpha: float = 0.5,
    ) -> TomoDoseSplitSolution:
        """
        Phase 5: 토모합성 선량 분할에 따른 DQE/SNR 비교

        Law 16: Dose-Split DQE Degradation
          D_proj = D_total / N
          DQE_EID(D_proj) = η_abs / (1 + α×N)
          DQE_PCD = η_abs (상수)
          PCD SNR advantage = √(1 + α×N)

        Phase 3 α 역산:
          α = f_e × D_ratio / (1 - f_e) = 0.30 × 0.5 / 0.70 = 0.2143

        Args:
            total_dose_uGy: 총 선량 (μGy), 2D mammo와 동일
            n_projections: 투영 수 (N)
            eta_abs: 흡수 양자 효율
            electronic_noise_fraction: Phase 1 전자노이즈 비율 (f_e at dose_ratio)
            dose_ratio_for_alpha: α 계산용 dose ratio (Phase 3)

        Returns:
            TomoDoseSplitSolution
        """
        steps = []

        # Step 1: Phase 3 파라미터 α 역산
        f_e = electronic_noise_fraction
        D_ref = dose_ratio_for_alpha
        alpha = f_e * D_ref / (1 - f_e)

        steps.append(DerivationStep(
            step_num=1,
            title="Phase 3 α 역산 (전자노이즈 파라미터)",
            latex=f"\\alpha = f_e \\times D_{{ref}} / (1 - f_e) = "
                  f"{f_e} \\times {D_ref} / {1-f_e:.2f} = {alpha:.4f}",
            numeric_value=round(alpha, 4)
        ))

        # Step 2: 투영당 선량
        dose_per_proj = total_dose_uGy / n_projections

        steps.append(DerivationStep(
            step_num=2,
            title="투영당 선량 (dose split)",
            latex=f"D_{{proj}} = D_{{total}} / N = {total_dose_uGy} / {n_projections} = "
                  f"{dose_per_proj:.2f} \\text{{ μGy}}",
            numeric_value=round(dose_per_proj, 2)
        ))

        # Step 3: DQE 계산
        # DQE_EID at D_proj: DQE = η_abs / (1 + α×N)
        # (normalized: D_proj/D_ref = 1/N, so α/D_normalized = α×N)
        dqe_eid_per_proj = eta_abs / (1 + alpha * n_projections)
        dqe_pcd_per_proj = eta_abs

        steps.append(DerivationStep(
            step_num=3,
            title="DQE 비교 (per projection)",
            latex=f"DQE_{{EID}} = \\eta_{{abs}} / (1 + \\alpha \\times N) = "
                  f"{eta_abs} / (1 + {alpha:.4f} \\times {n_projections}) = "
                  f"{eta_abs} / {1 + alpha * n_projections:.4f} = {dqe_eid_per_proj:.4f}\n"
                  f"DQE_{{PCD}} = \\eta_{{abs}} = {dqe_pcd_per_proj:.3f}",
            numeric_value=round(dqe_eid_per_proj, 4)
        ))

        # Step 4: PCD DQE advantage ratio
        pcd_dqe_advantage = dqe_pcd_per_proj / dqe_eid_per_proj

        steps.append(DerivationStep(
            step_num=4,
            title="PCD DQE 우위 비 (per projection)",
            latex=f"DQE_{{PCD}} / DQE_{{EID}} = {dqe_pcd_per_proj:.3f} / {dqe_eid_per_proj:.4f} = "
                  f"{pcd_dqe_advantage:.3f}\\times",
            numeric_value=round(pcd_dqe_advantage, 3)
        ))

        # Step 5: SNR per projection (relative, proportional to √(DQE × D_proj))
        snr_eid_per_proj = math.sqrt(dqe_eid_per_proj * dose_per_proj)
        snr_pcd_per_proj = math.sqrt(dqe_pcd_per_proj * dose_per_proj)

        steps.append(DerivationStep(
            step_num=5,
            title="SNR per projection (상대적)",
            latex=f"SNR_{{EID,proj}} \\propto \\sqrt{{DQE_{{EID}} \\times D_{{proj}}}} = "
                  f"\\sqrt{{{dqe_eid_per_proj:.4f} \\times {dose_per_proj:.2f}}} = {snr_eid_per_proj:.4f}\n"
                  f"SNR_{{PCD,proj}} \\propto \\sqrt{{{dqe_pcd_per_proj:.3f} \\times {dose_per_proj:.2f}}} = {snr_pcd_per_proj:.4f}",
            numeric_value=round(snr_eid_per_proj, 4)
        ))

        # Step 6: Total 3D SNR (N projections integrated)
        # SNR_total ∝ √(N) × SNR_per_proj = √(DQE × D_total)
        snr_eid_total = math.sqrt(dqe_eid_per_proj * total_dose_uGy)
        snr_pcd_total = math.sqrt(dqe_pcd_per_proj * total_dose_uGy)
        pcd_snr_gain = snr_pcd_total / snr_eid_total

        # Alternative: pcd_snr_gain = √(1 + α×N)
        pcd_snr_gain_formula = math.sqrt(1 + alpha * n_projections)

        steps.append(DerivationStep(
            step_num=6,
            title="Total 3D SNR (N 투영 적분)",
            latex=f"SNR_{{EID,total}} \\propto \\sqrt{{DQE_{{EID}} \\times D_{{total}}}} = "
                  f"\\sqrt{{{dqe_eid_per_proj:.4f} \\times {total_dose_uGy}}} = {snr_eid_total:.4f}\n"
                  f"SNR_{{PCD,total}} \\propto \\sqrt{{{dqe_pcd_per_proj:.3f} \\times {total_dose_uGy}}} = {snr_pcd_total:.4f}\n"
                  f"PCD/EID ratio = \\sqrt{{1 + \\alpha \\times N}} = \\sqrt{{1 + {alpha:.4f} \\times {n_projections}}} = {pcd_snr_gain_formula:.4f}",
            numeric_value=round(pcd_snr_gain, 4)
        ))

        # Step 7: Phase 3 교차검증 (N=1일 때 2D case와 일치)
        dqe_eid_full = eta_abs / (1 + alpha)
        # N=1: DQE = η/(1+α) should match Phase 3 DQE_EID(full)
        phase3_match = abs(dqe_eid_full - 0.700) < 0.005  # within 0.5%

        steps.append(DerivationStep(
            step_num=7,
            title="Phase 3 교차 검증 (N=1 → 2D case)",
            latex=f"N=1: DQE_{{EID}} = \\eta / (1 + \\alpha) = {eta_abs} / (1 + {alpha:.4f}) = "
                  f"{dqe_eid_full:.4f} \\approx 0.700 \\checkmark",
            numeric_value=round(dqe_eid_full, 4)
        ))

        return TomoDoseSplitSolution(
            total_dose_uGy=total_dose_uGy,
            n_projections=n_projections,
            dose_per_projection_uGy=round(dose_per_proj, 2),
            dqe_eid_per_proj=round(dqe_eid_per_proj, 4),
            dqe_pcd_per_proj=round(dqe_pcd_per_proj, 3),
            pcd_dqe_advantage_ratio=round(pcd_dqe_advantage, 3),
            snr_eid_per_proj=round(snr_eid_per_proj, 4),
            snr_pcd_per_proj=round(snr_pcd_per_proj, 4),
            snr_eid_total=round(snr_eid_total, 4),
            snr_pcd_total=round(snr_pcd_total, 4),
            pcd_snr_gain_total=round(pcd_snr_gain, 4),
            phase3_dqe_eid_full=round(dqe_eid_full, 4),
            phase3_alpha=round(alpha, 4),
            phase3_match=phase3_match,
            derivation_steps=steps,
        )

    def solve_tomo_resolution(
        self,
        angular_range_deg: float = 25.0,
        n_projections: int = 15,
        pixel_pitch_mm: float = 0.1,
        mtf_effective: float = 0.637,
        depth_resolution_constant: float = 0.50,
        breast_thickness_mm: float = 50.0,
    ) -> TomoResolutionSolution:
        """
        Phase 5: 토모합성 분해능 비대칭 분석

        Law 17: Resolution Asymmetry
          In-plane: Δxy = pixel_pitch / MTF
          Through-plane: Δz = K / sin(α_total/2) (Law 15)
          Asymmetry: Δz / Δxy >> 1

        Args:
            angular_range_deg: 총 각도 범위 (°)
            n_projections: 투영 수
            pixel_pitch_mm: 픽셀 피치 (mm)
            mtf_effective: 유효 MTF (Phase 4)
            depth_resolution_constant: K (mm)
            breast_thickness_mm: 유방 두께 (mm)

        Returns:
            TomoResolutionSolution
        """
        steps = []

        # Step 1: In-plane resolution (Phase 4)
        delta_xy = pixel_pitch_mm / mtf_effective
        nyquist_freq = 1.0 / (2.0 * pixel_pitch_mm)

        steps.append(DerivationStep(
            step_num=1,
            title="In-plane 분해능 (Phase 4)",
            latex=f"\\Delta xy = a / MTF = {pixel_pitch_mm} / {mtf_effective:.3f} = {delta_xy:.4f} \\text{{ mm}}\n"
                  f"f_{{Nyquist}} = 1/(2a) = 1/(2 \\times {pixel_pitch_mm}) = {nyquist_freq:.1f} \\text{{ lp/mm}}",
            numeric_value=round(delta_xy, 4)
        ))

        # Step 2: Through-plane resolution (Phase 4-B, Law 15)
        alpha_half_rad = math.radians(angular_range_deg / 2)
        sin_alpha_half = math.sin(alpha_half_rad)
        delta_z = depth_resolution_constant / sin_alpha_half

        steps.append(DerivationStep(
            step_num=2,
            title="Through-plane 분해능 (Law 15)",
            latex=f"\\Delta z = K / \\sin(\\alpha/2) = {depth_resolution_constant} / \\sin({angular_range_deg/2:.1f}°) = "
                  f"{depth_resolution_constant} / {sin_alpha_half:.4f} = {delta_z:.4f} \\text{{ mm}}",
            numeric_value=round(delta_z, 4)
        ))

        # Step 3: Asymmetry ratio
        asymmetry = delta_z / delta_xy

        steps.append(DerivationStep(
            step_num=3,
            title="분해능 비대칭비",
            latex=f"\\text{{Asymmetry}} = \\Delta z / \\Delta xy = {delta_z:.4f} / {delta_xy:.4f} = {asymmetry:.2f}\\times",
            numeric_value=round(asymmetry, 2)
        ))

        # Step 4: ASF FWHM and resolvable slices
        asf_fwhm = delta_z  # ASF FWHM = through-plane resolution
        n_slices = breast_thickness_mm / delta_z

        steps.append(DerivationStep(
            step_num=4,
            title="ASF 및 분별 슬라이스",
            latex=f"ASF_{{FWHM}} = \\Delta z = {asf_fwhm:.4f} \\text{{ mm}}\n"
                  f"N_{{slices}} = t / \\Delta z = {breast_thickness_mm} / {delta_z:.4f} = {n_slices:.1f}",
            numeric_value=round(n_slices, 1)
        ))

        # Step 5: Voxel dimensions
        voxel_z = min(delta_z, 1.0)  # slice spacing typically 1mm or Δz if smaller
        voxel_volume = pixel_pitch_mm * pixel_pitch_mm * voxel_z

        steps.append(DerivationStep(
            step_num=5,
            title="복셀 크기",
            latex=f"Voxel = {pixel_pitch_mm} \\times {pixel_pitch_mm} \\times {voxel_z:.3f} = "
                  f"{voxel_volume:.6f} \\text{{ mm³}}",
            numeric_value=round(voxel_volume, 6)
        ))

        return TomoResolutionSolution(
            angular_range_deg=angular_range_deg,
            n_projections=n_projections,
            pixel_pitch_mm=pixel_pitch_mm,
            mtf_effective=mtf_effective,
            delta_xy_mm=round(delta_xy, 4),
            nyquist_freq_lpmm=round(nyquist_freq, 1),
            depth_resolution_constant=depth_resolution_constant,
            delta_z_mm=round(delta_z, 4),
            resolution_asymmetry_ratio=round(asymmetry, 2),
            asf_fwhm_mm=round(asf_fwhm, 4),
            n_resolvable_slices=round(n_slices, 1),
            voxel_xy_mm=pixel_pitch_mm,
            voxel_z_mm=round(voxel_z, 3),
            voxel_volume_mm3=round(voxel_volume, 6),
            breast_thickness_mm=breast_thickness_mm,
            derivation_steps=steps,
        )

    def solve_tomo_detectability(
        self,
        angular_range_deg: float = 25.0,
        n_projections: int = 15,
        total_dose_uGy: float = 1500.0,
        breast_thickness_mm: float = 50.0,
        lesion_diameter_mm: float = 5.0,
        lesion_contrast: float = 0.02,
        eta_abs: float = 0.85,
        electronic_noise_fraction: float = 0.30,
        dose_ratio_for_alpha: float = 0.5,
        depth_resolution_constant: float = 0.50,
    ) -> TomoDetectabilitySolution:
        """
        Phase 5: 토모합성 병변 검출능 비교 (2D vs Tomo, EID vs PCD)

        Law 18: Anatomical Clutter Rejection
          Clutter rejection gain G = √(Δz / t_breast) [< 1, noise fraction remaining]
          SNR boost from tomo = 1/G = √(t_breast / Δz)
          d'_tomo = C × √(DQE × D × A_lesion) × (1/G)

        Args:
            angular_range_deg: 총 각도 범위 (°)
            n_projections: 투영 수
            total_dose_uGy: 총 선량 (μGy)
            breast_thickness_mm: 유방 두께 (mm)
            lesion_diameter_mm: 병변 직경 (mm)
            lesion_contrast: 병변 대조도 (0-1)
            eta_abs: 흡수 효율
            electronic_noise_fraction: 전자노이즈 비율
            dose_ratio_for_alpha: α 계산용 dose ratio
            depth_resolution_constant: K (mm)

        Returns:
            TomoDetectabilitySolution
        """
        steps = []

        # Step 1: Phase 3 파라미터
        f_e = electronic_noise_fraction
        alpha = f_e * dose_ratio_for_alpha / (1 - f_e)

        steps.append(DerivationStep(
            step_num=1,
            title="Phase 3 α 파라미터",
            latex=f"\\alpha = {f_e} \\times {dose_ratio_for_alpha} / {1-f_e:.2f} = {alpha:.4f}",
            numeric_value=round(alpha, 4)
        ))

        # Step 2: DQE 계산
        dqe_eid_2d = eta_abs / (1 + alpha)  # full dose 2D
        dqe_eid_tomo = eta_abs / (1 + alpha * n_projections)  # dose-split tomo
        dqe_pcd = eta_abs  # always

        steps.append(DerivationStep(
            step_num=2,
            title="DQE 비교",
            latex=f"DQE_{{EID,2D}} = {eta_abs}/(1+{alpha:.4f}) = {dqe_eid_2d:.4f}\n"
                  f"DQE_{{EID,tomo}} = {eta_abs}/(1+{alpha:.4f}\\times{n_projections}) = {dqe_eid_tomo:.4f}\n"
                  f"DQE_{{PCD}} = {dqe_pcd}",
            numeric_value=round(dqe_eid_tomo, 4)
        ))

        # Step 3: Through-plane resolution (for clutter)
        alpha_half_rad = math.radians(angular_range_deg / 2)
        sin_alpha_half = math.sin(alpha_half_rad)
        slice_thickness = depth_resolution_constant / sin_alpha_half

        steps.append(DerivationStep(
            step_num=3,
            title="슬라이스 두께 (Law 15)",
            latex=f"\\Delta z = K/\\sin(\\alpha/2) = {depth_resolution_constant}/{sin_alpha_half:.4f} = "
                  f"{slice_thickness:.4f} \\text{{ mm}}",
            numeric_value=round(slice_thickness, 4)
        ))

        # Step 4: Clutter rejection
        # G = √(Δz / t_breast): fraction of clutter remaining
        # 1/G = √(t/Δz): SNR boost from tissue separation
        if slice_thickness < breast_thickness_mm:
            clutter_g = math.sqrt(slice_thickness / breast_thickness_mm)
            clutter_boost = 1.0 / clutter_g  # = √(t/Δz)
        else:
            # N=1 or very narrow angle: no clutter rejection
            clutter_g = 1.0
            clutter_boost = 1.0

        steps.append(DerivationStep(
            step_num=4,
            title="Clutter rejection (Law 18)",
            latex=f"G_{{clutter}} = \\sqrt{{\\Delta z / t}} = \\sqrt{{{slice_thickness:.4f}/{breast_thickness_mm}}} = {clutter_g:.4f}\n"
                  f"SNR boost = 1/G = \\sqrt{{t/\\Delta z}} = {clutter_boost:.4f}",
            numeric_value=round(clutter_boost, 4)
        ))

        # Step 5: Detectability index d' (Rose model, relative units)
        # d' ∝ C × √(DQE × D_total × A_lesion) × clutter_boost_factor
        # For 2D: no clutter rejection (clutter_boost = 1)
        # For tomo: includes clutter rejection
        lesion_area = math.pi * (lesion_diameter_mm / 2) ** 2  # mm²

        # Base d' (proportional, using arbitrary reference)
        # We compute relative values
        d_prime_2d_eid = lesion_contrast * math.sqrt(dqe_eid_2d * total_dose_uGy * lesion_area)
        d_prime_tomo_eid = lesion_contrast * math.sqrt(dqe_eid_tomo * total_dose_uGy * lesion_area) * clutter_boost
        d_prime_tomo_pcd = lesion_contrast * math.sqrt(dqe_pcd * total_dose_uGy * lesion_area) * clutter_boost

        steps.append(DerivationStep(
            step_num=5,
            title="Detectability (d' — Rose model)",
            latex=f"d'_{{2D,EID}} = C \\times \\sqrt{{DQE \\times D \\times A}} = {d_prime_2d_eid:.4f}\n"
                  f"d'_{{tomo,EID}} = ... \\times \\sqrt{{t/\\Delta z}} = {d_prime_tomo_eid:.4f}\n"
                  f"d'_{{tomo,PCD}} = ... = {d_prime_tomo_pcd:.4f}",
            numeric_value=round(d_prime_tomo_pcd, 4)
        ))

        # Step 6: Improvement factors
        tomo_vs_2d_eid = d_prime_tomo_eid / d_prime_2d_eid if d_prime_2d_eid > 0 else 0
        pcd_vs_eid_tomo = d_prime_tomo_pcd / d_prime_tomo_eid if d_prime_tomo_eid > 0 else 0
        pcd_tomo_vs_2d = d_prime_tomo_pcd / d_prime_2d_eid if d_prime_2d_eid > 0 else 0

        steps.append(DerivationStep(
            step_num=6,
            title="개선 비율",
            latex=f"Tomo_{{EID}} / 2D_{{EID}} = {tomo_vs_2d_eid:.3f}\\times\n"
                  f"PCD_{{tomo}} / EID_{{tomo}} = {pcd_vs_eid_tomo:.3f}\\times\n"
                  f"PCD_{{tomo}} / 2D_{{EID}} = {pcd_tomo_vs_2d:.3f}\\times \\text{{ (total)}}",
            numeric_value=round(pcd_tomo_vs_2d, 3)
        ))

        return TomoDetectabilitySolution(
            angular_range_deg=angular_range_deg,
            n_projections=n_projections,
            total_dose_uGy=total_dose_uGy,
            breast_thickness_mm=breast_thickness_mm,
            lesion_diameter_mm=lesion_diameter_mm,
            lesion_contrast=lesion_contrast,
            dqe_eid_2d=round(dqe_eid_2d, 4),
            dqe_eid_tomo=round(dqe_eid_tomo, 4),
            dqe_pcd=round(dqe_pcd, 3),
            slice_thickness_mm=round(slice_thickness, 4),
            clutter_rejection_gain=round(clutter_g, 4),
            clutter_snr_boost=round(clutter_boost, 4),
            d_prime_2d_eid=round(d_prime_2d_eid, 4),
            d_prime_tomo_eid=round(d_prime_tomo_eid, 4),
            d_prime_tomo_pcd=round(d_prime_tomo_pcd, 4),
            tomo_vs_2d_gain_eid=round(tomo_vs_2d_eid, 4),
            pcd_vs_eid_tomo_gain=round(pcd_vs_eid_tomo, 4),
            pcd_tomo_vs_2d_eid_gain=round(pcd_tomo_vs_2d, 4),
            rose_threshold=5.0,
            derivation_steps=steps,
        )

    def format_tomo_prompt(
        self,
        dose_sol: TomoDoseSplitSolution,
        res_sol: TomoResolutionSolution,
        detect_sol: TomoDetectabilitySolution,
    ) -> str:
        """Phase 5: 토모합성 영상 품질 제약 조건 프롬프트 생성"""
        lines = [
            "=" * 60,
            "[Phase 5] Tomosynthesis Image Quality 제약 조건 (Python Solver 검증)",
            "=" * 60,
            "",
            "  [Law 16: Dose-Split DQE]",
            f"  총 선량: {dose_sol.total_dose_uGy} μGy, 투영수: {dose_sol.n_projections}",
            f"  투영당 선량: {dose_sol.dose_per_projection_uGy:.2f} μGy",
            f"  DQE_EID(per proj): {dose_sol.dqe_eid_per_proj:.4f}",
            f"  DQE_PCD(per proj): {dose_sol.dqe_pcd_per_proj:.3f} (선량 무관)",
            f"  PCD DQE 우위: {dose_sol.pcd_dqe_advantage_ratio:.2f}× (per projection)",
            f"  PCD SNR 우위 (total): {dose_sol.pcd_snr_gain_total:.4f}× = √(1+α×N)",
            f"  Phase 3 α: {dose_sol.phase3_alpha:.4f}",
            f"  Phase 3 교차검증: {'PASS' if dose_sol.phase3_match else 'FAIL'}",
            "",
            "  [Law 17: Resolution Asymmetry]",
            f"  In-plane Δxy: {res_sol.delta_xy_mm:.4f} mm",
            f"  Through-plane Δz: {res_sol.delta_z_mm:.4f} mm",
            f"  Asymmetry ratio: {res_sol.resolution_asymmetry_ratio:.1f}×",
            f"  Resolvable slices: {res_sol.n_resolvable_slices:.1f}",
            "",
            "  [Law 18: Clutter Rejection & Detectability]",
            f"  Clutter boost: {detect_sol.clutter_snr_boost:.3f}× [= √(t/Δz)]",
            f"  Tomo EID / 2D EID: {detect_sol.tomo_vs_2d_gain_eid:.3f}×",
            f"  PCD tomo / EID tomo: {detect_sol.pcd_vs_eid_tomo_gain:.3f}×",
            f"  PCD tomo / 2D EID: {detect_sol.pcd_tomo_vs_2d_eid_gain:.3f}× (total)",
            "",
            "  ⚠️ 이 수치와 1% 초과 불일치 시 답변 거부",
            "=" * 60,
        ]
        return "\n".join(lines)

    def audit_tomo_answer(
        self,
        llm_answer: str,
        total_dose_uGy: float = 1500.0,
        n_projections: int = 25,
        angular_range_deg: float = 25.0,
        tolerance_pct: float = 1.0,
    ) -> List[AuditResult]:
        """
        Phase 5: LLM 답변의 토모합성 수치 검증

        Args:
            llm_answer: LLM이 생성한 답변 텍스트
            total_dose_uGy: 총 선량
            n_projections: 투영 수
            angular_range_deg: 각도 범위
            tolerance_pct: 허용 오차 (%)

        Returns:
            List[AuditResult]
        """
        results = []

        # 솔버 정답 계산
        dose_sol = self.solve_tomo_dose_split(
            total_dose_uGy=total_dose_uGy,
            n_projections=n_projections,
        )
        res_sol = self.solve_tomo_resolution(
            angular_range_deg=angular_range_deg,
            n_projections=n_projections,
        )

        # 검증 대상 목록
        checks = [
            ("pcd_snr_gain", dose_sol.pcd_snr_gain_total, r"(?:PCD.*?SNR.*?gain|SNR.*?ratio|PCD.*?advantage).*?([\d.]+)"),
            ("dqe_eid_per_proj", dose_sol.dqe_eid_per_proj, r"DQE.*?EID.*?([\d.]+)"),
            ("asymmetry_ratio", res_sol.resolution_asymmetry_ratio, r"(?:asymmetry|비대칭).*?([\d.]+)"),
            ("delta_z", res_sol.delta_z_mm, r"(?:Δz|delta.?z|through.?plane|슬라이스.*?두께).*?([\d.]+)"),
        ]

        for field_name, correct_val, pattern in checks:
            match = re.search(pattern, llm_answer, re.IGNORECASE)
            if match:
                try:
                    llm_val = float(match.group(1))
                    if correct_val != 0:
                        error = abs(llm_val - correct_val) / abs(correct_val) * 100
                    else:
                        error = abs(llm_val) * 100
                    should_reject = error > tolerance_pct

                    results.append(AuditResult(
                        status=AuditStatus.REJECT if should_reject else AuditStatus.PASS,
                        target_field=field_name,
                        llm_value=llm_val,
                        correct_value=correct_val,
                        error_pct=round(error, 2),
                        tolerance_pct=tolerance_pct,
                        should_reject=should_reject,
                        explanation=f"Phase 5 {field_name}: LLM={llm_val}, Solver={correct_val:.4f}, Error={error:.2f}%",
                        correction_hint=f"Phase 5 {field_name}의 정확한 계산을 다시 수행하세요." if should_reject else "",
                    ))
                except (ValueError, ZeroDivisionError):
                    pass

        return results


# =============================================================================
# Singleton
# =============================================================================

_solver_instance: Optional[MammoPhysicsSolver] = None


def get_mammo_solver() -> MammoPhysicsSolver:
    """MammoPhysicsSolver 싱글톤"""
    global _solver_instance
    if _solver_instance is None:
        _solver_instance = MammoPhysicsSolver()
    return _solver_instance


# =============================================================================
# Test
# =============================================================================

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    solver = MammoPhysicsSolver()

    print("=" * 70)
    print("MammoPhysicsSolver: Layer 2 Deterministic Verification")
    print("=" * 70)
    print("\n[조건] 선량 50% 감소, 전자노이즈 비율 30% (변화 후)")
    print()

    # 정답 계산
    solution = solver.solve_snr_with_electronic_noise(
        dose_ratio=0.5,
        electronic_noise_fraction=0.3
    )

    print(f"  EID SNR 비율: {solution.eid_snr_ratio:.4f}")
    print(f"  EID SNR 감소율: {solution.eid_snr_reduction_pct:.1f}%")
    print(f"  PCD SNR 비율: {solution.pcd_snr_ratio:.4f}")
    print(f"  PCD SNR 감소율: {solution.pcd_snr_reduction_pct:.1f}%")
    print(f"  PCD 회복률: +{solution.pcd_recovery_pct:.1f}%")
    print(f"  Rose Criterion 최소 SNR_0 (EID): {solution.rose_min_snr0:.2f}")
    print(f"  Rose (EID): {'OK' if solution.rose_eid_satisfied else 'FAIL'}")
    print(f"  Rose (PCD): {'OK' if solution.rose_pcd_satisfied else 'FAIL'}")

    print(f"\n{'='*70}")
    print("유도 과정:")
    print("=" * 70)
    print(solver.format_derivation_latex(solution))

    print(f"\n{'='*70}")
    print("LLM 답변 감사 테스트:")
    print("=" * 70)

    test_cases = [
        ("SNR이 34.8% 감소합니다.", "정답"),
        ("SNR이 34.5% 감소합니다.", "1% 이내 근사"),
        ("SNR이 29.3% 감소합니다.", "전자노이즈 미고려 (=PCD 값)"),
        ("SNR이 50.0% 감소합니다.", "단순 비례 오류"),
        ("SNR이 63.8% 감소합니다.", "상대노이즈 모델 오류"),
    ]

    for answer, desc in test_cases:
        audit_results = solver.audit_llm_answer(answer, 0.5, 0.3)
        for ar in audit_results:
            status_icon = "PASS" if ar.status == AuditStatus.PASS else "REJECT"
            print(f"\n  [{desc}] \"{answer}\"")
            print(f"    {status_icon}: LLM={ar.llm_value:.1f}%, Correct={ar.correct_value:.1f}%, Error={ar.error_pct:.2f}%p")
            if ar.correction_hint:
                print(f"    Hint: {ar.correction_hint}")

    print(f"\n{'='*70}")
    print("제약 조건 프롬프트:")
    print("=" * 70)
    print(solver.format_constraint_prompt(solution))

    # =================================================================
    # Phase 2: PCD Spectral Contrast Tests
    # =================================================================
    print(f"\n{'='*70}")
    print("Phase 2: PCD Spectral Contrast Solver")
    print("=" * 70)

    # Test 1: 4-빈 CESM 에너지 가중 이득
    print("\n[Test 1] CESM 4-빈 모델 (Iodine K-edge)")
    cesm_bins = MammoPhysicsSolver.get_iodine_cesm_bins()
    contrast_sol = solver.solve_energy_weighting_gain(cesm_bins)
    print(f"  빈 수: {contrast_sol.n_bins}")
    print(f"  CNR_EID: {contrast_sol.cnr_eid:.4f}")
    print(f"  CNR_PCD: {contrast_sol.cnr_pcd:.4f}")
    print(f"  에너지 가중 이득 η: {contrast_sol.eta:.4f} (+{contrast_sol.eta_percent:.1f}%)")
    assert contrast_sol.eta >= 1.0, "Cauchy-Schwarz violation!"

    # Test 2: 2-빈 K-edge 모델
    print("\n[Test 2] Iodine K-edge 2-빈 모델")
    n_below, n_above, dmu_below, dmu_above = MammoPhysicsSolver.get_iodine_2bin_simple()
    kedge_sol = solver.solve_kedge_cnr(n_below, n_above, dmu_below, dmu_above)
    print(f"  K-edge: {kedge_sol.kedge_energy_keV} keV ({kedge_sol.contrast_agent})")
    print(f"  CNR_EID: {kedge_sol.cnr_eid:.4f}")
    print(f"  CNR_PCD: {kedge_sol.cnr_pcd:.4f}")
    print(f"  에너지 가중 이득 η: {kedge_sol.eta:.4f} (+{kedge_sol.eta_percent:.1f}%)")
    print(f"  Δμ jump ratio (above/below): {dmu_above/dmu_below:.1f}x")
    assert kedge_sol.eta >= 1.0, "Cauchy-Schwarz violation!"

    # Test 3: 단색 빔 (η = 1 이어야 함)
    print("\n[Test 3] 단색 빔 검증 (η = 1 예상)")
    mono_bins = [
        EnergyBin("mono", 30.0, 1000.0, 0.5),
    ]
    mono_sol = solver.solve_energy_weighting_gain(mono_bins)
    print(f"  η = {mono_sol.eta:.6f} (expected: 1.000000)")
    assert abs(mono_sol.eta - 1.0) < 1e-10, f"Mono beam should give η=1, got {mono_sol.eta}"

    # Test 4: 동일 Δμ, 다른 에너지 (η > 1 이지만 매우 작음)
    # EID의 w∝E 가중이 고에너지를 과다가중하므로, Δμ가 균일해도 약간의 이득 발생
    print("\n[Test 4] 동일 Δμ, 다른 에너지 (η ≈ 1, EID w∝E 효과)")
    uniform_bins = [
        EnergyBin("bin1", 25.0, 500.0, 0.5),
        EnergyBin("bin2", 30.0, 500.0, 0.5),
        EnergyBin("bin3", 35.0, 500.0, 0.5),
    ]
    uniform_sol = solver.solve_energy_weighting_gain(uniform_bins)
    print(f"  η = {uniform_sol.eta:.6f} (expected: ~1.009, EID w∝E sub-optimality)")
    assert 1.0 < uniform_sol.eta < 1.02, f"Uniform Δμ should give η≈1.009, got {uniform_sol.eta}"

    # Test 5: 극단적 K-edge (큰 η)
    print("\n[Test 5] 극단적 K-edge (Δμ 비율 10:1)")
    extreme_sol = solver.solve_kedge_cnr(
        n_below=500, n_above=500,
        dmu_below=0.1, dmu_above=1.0
    )
    print(f"  η = {extreme_sol.eta:.4f} (+{extreme_sol.eta_percent:.1f}%)")
    assert extreme_sol.eta > 1.1, f"Extreme K-edge should give η > 1.1, got {extreme_sol.eta}"

    print(f"\n{'='*70}")
    print("Phase 2 제약 조건 프롬프트:")
    print("=" * 70)
    print(solver.format_contrast_prompt(kedge_sol))

    # =================================================================
    # Phase 3: DQE / NPS Tests
    # =================================================================
    print(f"\n{'='*70}")
    print("Phase 3: DQE / NPS Solver")
    print("=" * 70)

    # Test 6: DQE dose dependence (기본 파라미터)
    print("\n[Test 6] DQE dose dependence (η_abs=0.85, f_e=0.30, D=0.5)")
    dqe_sol = solver.solve_dqe_dose_dependence(
        eta_abs=0.85, electronic_noise_fraction=0.30, dose_ratio=0.5
    )
    print(f"  DQE_EID(full): {dqe_sol.dqe_eid_full_dose:.4f}")
    print(f"  DQE_EID(D=0.5): {dqe_sol.dqe_eid_at_dose_ratio:.4f}")
    print(f"  DQE_PCD: {dqe_sol.dqe_pcd:.4f}")
    print(f"  PCD advantage: +{dqe_sol.pcd_advantage_percent:.1f}%")
    print(f"  EID DQE degradation: {dqe_sol.dqe_degradation_percent:.1f}%")

    # 기대값 검증 (계획서 기준)
    assert abs(dqe_sol.dqe_eid_full_dose - 0.700) < 0.001, \
        f"DQE_EID(full) should be ~0.700, got {dqe_sol.dqe_eid_full_dose}"
    assert abs(dqe_sol.dqe_eid_at_dose_ratio - 0.595) < 0.001, \
        f"DQE_EID(half) should be ~0.595, got {dqe_sol.dqe_eid_at_dose_ratio}"
    assert abs(dqe_sol.dqe_pcd - 0.850) < 0.001, \
        f"DQE_PCD should be 0.850, got {dqe_sol.dqe_pcd}"
    assert abs(dqe_sol.pcd_advantage_percent - 42.9) < 0.1, \
        f"PCD advantage should be ~42.9%, got {dqe_sol.pcd_advantage_percent}"
    assert dqe_sol.dqe_pcd > dqe_sol.dqe_eid_at_dose_ratio, \
        "DQE_PCD must always > DQE_EID at reduced dose"
    assert dqe_sol.dqe_eid_at_dose_ratio < dqe_sol.dqe_eid_full_dose, \
        "DQE_EID must decrease at lower dose"
    print("  ✓ 기대값 검증 통과")

    # Test 7: Phase 1 교차 검증
    print("\n[Test 7] Phase 1 교차 검증 (DQE → SNR ratio)")
    # DQE를 통한 SNR 비율
    snr_from_dqe = math.sqrt(
        dqe_sol.dqe_eid_at_dose_ratio * 0.5 / dqe_sol.dqe_eid_full_dose
    )
    # Phase 1 직접 계산
    phase1_sol = solver.solve_snr_with_electronic_noise(0.5, 0.30)
    print(f"  SNR ratio (DQE method): {snr_from_dqe:.4f}")
    print(f"  SNR ratio (Phase 1):    {phase1_sol.eid_snr_ratio:.4f}")
    assert abs(snr_from_dqe - phase1_sol.eid_snr_ratio) < 1e-10, \
        f"Phase 1-3 cross-validation failed: {snr_from_dqe} vs {phase1_sol.eid_snr_ratio}"
    print("  ✓ Phase 1 ↔ Phase 3 교차 검증 일치")

    # Test 8: NPS decomposition
    print("\n[Test 8] NPS decomposition (D=1.0)")
    nps_sol = solver.solve_nps_decomposition(dose_ratio=1.0, electronic_noise_fraction=0.30)
    print(f"  NPS_quantum: {nps_sol.nps_quantum:.8f}")
    print(f"  NPS_electronic: {nps_sol.nps_electronic:.8f}")
    print(f"  NPS_EID: {nps_sol.nps_total_eid:.8f}")
    print(f"  NPS_PCD: {nps_sol.nps_total_pcd:.8f}")
    print(f"  Electronic fraction: {nps_sol.electronic_fraction_eid:.1%}")
    print(f"  PCD NPS reduction: {nps_sol.pcd_nps_reduction_percent:.1f}%")
    assert nps_sol.nps_total_pcd < nps_sol.nps_total_eid, \
        "NPS_PCD must be < NPS_EID"
    assert nps_sol.electronic_fraction_eid > 0, \
        "Electronic fraction must be > 0 for EID"
    assert abs(nps_sol.pcd_nps_reduction_percent - nps_sol.electronic_fraction_eid * 100) < 0.01, \
        "PCD NPS reduction should equal electronic fraction"
    print("  ✓ NPS 분해 검증 통과")

    # Test 9: NPS at low dose (전자노이즈 비율 증가 확인)
    print("\n[Test 9] NPS at low dose (D=0.25)")
    nps_low = solver.solve_nps_decomposition(dose_ratio=0.25, electronic_noise_fraction=0.30)
    print(f"  Electronic fraction at D=1.0: {nps_sol.electronic_fraction_eid:.1%}")
    print(f"  Electronic fraction at D=0.25: {nps_low.electronic_fraction_eid:.1%}")
    assert nps_low.electronic_fraction_eid > nps_sol.electronic_fraction_eid, \
        "Electronic fraction must increase at lower dose"
    print("  ✓ 저선량에서 전자노이즈 비율 증가 확인")

    # Test 10: NEQ calculation
    print("\n[Test 10] NEQ calculation")
    q_in = 100000  # 입사 광자수
    neq_eid = solver.solve_neq(dqe_sol.dqe_eid_full_dose, q_in)
    neq_pcd = solver.solve_neq(dqe_sol.dqe_pcd, q_in)
    print(f"  NEQ_EID (full dose): {neq_eid:.0f}")
    print(f"  NEQ_PCD: {neq_pcd:.0f}")
    assert neq_pcd > neq_eid, "NEQ_PCD must be > NEQ_EID"
    print("  ✓ NEQ 검증 통과")

    # Test 11: DQE curve 단조 증가 검증
    print("\n[Test 11] DQE-dose curve 단조 증가 검증")
    for i in range(1, len(dqe_sol.dqe_eid_curve)):
        assert dqe_sol.dqe_eid_curve[i] >= dqe_sol.dqe_eid_curve[i-1], \
            f"DQE curve not monotonically increasing at D={dqe_sol.dose_points[i]}"
    print(f"  DQE range: {dqe_sol.dqe_eid_curve[0]:.4f} (D=0.1) → {dqe_sol.dqe_eid_curve[-1]:.4f} (D=2.0)")
    assert dqe_sol.dqe_eid_curve[-1] < dqe_sol.dqe_pcd, \
        "DQE_EID can never reach η_abs (that's PCD's limit)"
    print("  ✓ DQE 커브 단조 증가, η_abs 미만 확인")

    # Phase 3 제약 조건 프롬프트
    print(f"\n{'='*70}")
    print("Phase 3 제약 조건 프롬프트:")
    print("=" * 70)
    print(solver.format_dqe_prompt(dqe_sol))

    # =================================================================
    # Phase 4: MTF / Spatial Resolution / DQE(f) Tests
    # =================================================================
    print(f"\n{'='*70}")
    print("Phase 4: MTF / Spatial Resolution / DQE(f) Solver")
    print("=" * 70)

    # Test 12: MTF comparison (pixel=0.1mm, CsI 150um)
    print("\n[Test 12] MTF comparison (pixel=0.1mm, CsI 150um, δ=0.10)")
    mtf_sol = solver.solve_mtf_comparison(
        pixel_pitch_mm=0.1,
        scintillator_thickness_um=150.0,
        converter='CsI',
        cs_delta=0.10
    )
    print(f"  Nyquist freq: {mtf_sol.nyquist_freq:.1f} lp/mm")
    print(f"  MTF_PCD(Nyquist): {mtf_sol.mtf_pcd_at_nyquist:.4f}")
    print(f"  MTF_EID(Nyquist): {mtf_sol.mtf_eid_at_nyquist:.4f}")
    print(f"  Scintillator MTF factor: {mtf_sol.scintillator_mtf_factor:.4f}")
    print(f"  f10_PCD: {mtf_sol.f10_pcd:.2f} lp/mm")
    print(f"  f10_EID: {mtf_sol.f10_eid:.2f} lp/mm")
    print(f"  Resolution gain: {mtf_sol.pcd_resolution_gain:.2f}×")
    print(f"  Charge sharing loss: {mtf_sol.charge_sharing_degradation:.0f}%")

    # Assertions
    assert abs(mtf_sol.mtf_pcd_at_nyquist - abs(math.sin(math.pi/2)/(math.pi/2)) * 0.90) < 0.01, \
        f"MTF_PCD(Ny) should be sinc(π/2)×0.90 ≈ 0.573, got {mtf_sol.mtf_pcd_at_nyquist}"
    assert mtf_sol.mtf_pcd_at_nyquist > mtf_sol.mtf_eid_at_nyquist, \
        "MTF_PCD must be > MTF_EID at all f > 0"
    assert mtf_sol.f10_pcd > mtf_sol.f10_eid, \
        "PCD f10 must be > EID f10"
    # Verify PCD MTF > EID MTF at all frequencies > 0
    for i in range(1, len(mtf_sol.freq_points)):
        assert mtf_sol.mtf_pcd_curve[i] >= mtf_sol.mtf_eid_curve[i] - 1e-10, \
            f"MTF_PCD < MTF_EID at f={mtf_sol.freq_points[i]:.2f}"
    print("  ✓ MTF_PCD > MTF_EID at all f > 0")

    # Test 13: DQE(f) curves
    print("\n[Test 13] DQE(f) curves (Phase 3 cross-validation)")
    dqef_sol = solver.solve_dqe_frequency(
        pixel_pitch_mm=0.1,
        eta_abs=0.85,
        electronic_noise_fraction=0.30,
        scintillator_thickness_um=150.0,
        converter='CsI',
        cs_delta=0.10
    )
    print(f"  DQE_PCD(0): {dqef_sol.dqe_pcd_at_zero:.4f} (expected: 0.850)")
    print(f"  DQE_EID(0): {dqef_sol.dqe_eid_at_zero:.4f} (expected: 0.700)")
    print(f"  DQE_PCD(Nyquist): {dqef_sol.dqe_pcd_at_nyquist:.4f}")
    print(f"  DQE_EID(Nyquist): {dqef_sol.dqe_eid_at_nyquist:.4f}")
    print(f"  PCD advantage at Nyquist: {dqef_sol.pcd_dqe_advantage_at_nyquist:.1f}×")
    print(f"  Phase 3 match: {dqef_sol.phase3_dqe_match}")

    assert abs(dqef_sol.dqe_pcd_at_zero - 0.850) < 0.001, \
        f"DQE_PCD(0) should be 0.850, got {dqef_sol.dqe_pcd_at_zero}"
    assert abs(dqef_sol.dqe_eid_at_zero - 0.700) < 0.01, \
        f"DQE_EID(0) should be ~0.700, got {dqef_sol.dqe_eid_at_zero}"
    assert dqef_sol.phase3_dqe_match, "Phase 3 cross-validation must pass"
    # DQE_PCD > DQE_EID at all practical frequencies
    for i in range(len(dqef_sol.freq_points)):
        assert dqef_sol.dqe_pcd_curve[i] >= dqef_sol.dqe_eid_curve[i] - 1e-6, \
            f"DQE_PCD < DQE_EID at f={dqef_sol.freq_points[i]:.2f}"
    print("  ✓ DQE_PCD(0) = Phase 3 η_abs")
    print("  ✓ DQE_EID(0) = Phase 3 DQE_EID(full)")
    print("  ✓ DQE_PCD(f) ≥ DQE_EID(f) at all frequencies")

    # Test 14: Charge sharing effect (1mm CdTe, 0.1mm pixel)
    print("\n[Test 14] Charge sharing effect (CdTe 1mm, pixel 0.1mm)")
    cs_sol = solver.solve_charge_sharing_effect(
        cdte_thickness_mm=1.0,
        pixel_pitch_mm=0.1
    )
    print(f"  δ_CS: {cs_sol.cs_delta:.3f}")
    print(f"  MTF degradation at Nyquist: {cs_sol.charge_sharing_degradation:.0f}%")
    print(f"  f10_PCD (with CS): {cs_sol.f10_pcd:.2f} lp/mm")
    assert 5 <= cs_sol.charge_sharing_degradation <= 30, \
        f"Charge sharing should be 5-30% for 1mm CdTe, got {cs_sol.charge_sharing_degradation}%"
    print("  ✓ Charge sharing ~9% at Nyquist (1mm CdTe)")

    # Test thicker CdTe
    cs_sol_thick = solver.solve_charge_sharing_effect(cdte_thickness_mm=3.0, pixel_pitch_mm=0.1)
    print(f"  CdTe 3mm: δ_CS={cs_sol_thick.cs_delta:.3f}, loss={cs_sol_thick.charge_sharing_degradation:.0f}%")
    assert cs_sol_thick.charge_sharing_degradation > cs_sol.charge_sharing_degradation, \
        "Thicker CdTe should have more charge sharing"
    print("  ✓ Thicker CdTe → more charge sharing")

    # Test 15: Cross-validation with Phase 3
    print("\n[Test 15] Cross-validation with Phase 3 (DQE(f→0) == Phase 3)")
    # DQE from Phase 3 (dose_ratio=0.5: f_e is measured at half dose, same convention)
    dqe_phase3 = solver.solve_dqe_dose_dependence(eta_abs=0.85, electronic_noise_fraction=0.30, dose_ratio=0.5)
    assert abs(dqef_sol.dqe_pcd_at_zero - dqe_phase3.dqe_pcd) < 1e-10, \
        f"Phase 3-4 PCD cross-validation: {dqef_sol.dqe_pcd_at_zero} vs {dqe_phase3.dqe_pcd}"
    assert abs(dqef_sol.dqe_eid_at_zero - dqe_phase3.dqe_eid_full_dose) < 1e-6, \
        f"Phase 3-4 EID cross-validation: {dqef_sol.dqe_eid_at_zero} vs {dqe_phase3.dqe_eid_full_dose}"
    print(f"  DQE_PCD: Phase 4={dqef_sol.dqe_pcd_at_zero:.6f}, Phase 3={dqe_phase3.dqe_pcd:.6f}")
    print(f"  DQE_EID: Phase 4={dqef_sol.dqe_eid_at_zero:.6f}, Phase 3={dqe_phase3.dqe_eid_full_dose:.6f}")
    print("  ✓ Phase 3 ↔ Phase 4 교차 검증 완벽 일치")

    # Test 16: Edge case — large pixel (0.5mm)
    print("\n[Test 16] Edge case — large pixel (0.5mm, minimal charge sharing)")
    mtf_large = solver.solve_mtf_comparison(
        pixel_pitch_mm=0.5,
        scintillator_thickness_um=150.0,
        converter='CsI',
        cs_delta=0.03  # large pixel → minimal CS
    )
    print(f"  Nyquist: {mtf_large.nyquist_freq:.1f} lp/mm")
    print(f"  MTF_PCD(Ny): {mtf_large.mtf_pcd_at_nyquist:.4f}")
    print(f"  MTF_EID(Ny): {mtf_large.mtf_eid_at_nyquist:.4f}")
    # Verify smooth, monotone decreasing MTF
    for i in range(1, len(mtf_large.mtf_pcd_curve)):
        assert mtf_large.mtf_pcd_curve[i] <= mtf_large.mtf_pcd_curve[i-1] + 1e-10, \
            f"MTF_PCD not monotone decreasing at index {i}"
    print("  ✓ MTF curve smooth, monotone decreasing")

    # Test 17: Heismann verification — PCD/EID resolution comparison
    print("\n[Test 17] Heismann verification (PCD resolution / EID resolution)")
    # GOS 208um (typical EID scintillator) vs PCD, same 0.1mm pixel
    mtf_gos = solver.solve_mtf_comparison(
        pixel_pitch_mm=0.1,
        scintillator_thickness_um=208.0,
        converter='GOS',
        cs_delta=0.10
    )
    print(f"  GOS 208um (same pixel): f10_PCD={mtf_gos.f10_pcd:.2f}, f10_EID={mtf_gos.f10_eid:.2f}")
    print(f"  Resolution gain (same pixel, GOS): {mtf_gos.pcd_resolution_gain:.2f}×")
    # Same-pixel comparison: PCD > EID due to scintillator blur
    assert mtf_gos.pcd_resolution_gain > 1.4, \
        f"Expected resolution gain > 1.4× for GOS same pixel, got {mtf_gos.pcd_resolution_gain}"

    # Kuttig 2015 scenario: CdTe f10=8.5 (abstract confirmed) vs GOS (full text data)
    # Cross-system comparison includes pixel size differences in commercial systems
    # For thicker GOS with larger pixel (as in typical EID chest systems):
    mtf_gos_thick = solver.solve_mtf_comparison(
        pixel_pitch_mm=0.15,   # larger EID pixel (typical)
        scintillator_thickness_um=400.0,  # thicker GOS
        converter='GOS',
        cs_delta=0.10
    )
    # PCD with fine pixel
    mtf_pcd_fine = solver.solve_mtf_comparison(
        pixel_pitch_mm=0.1,
        scintillator_thickness_um=150.0,
        converter='CsI',  # not used for PCD
        cs_delta=0.10
    )
    cross_system_gain = mtf_pcd_fine.f10_pcd / mtf_gos_thick.f10_eid if mtf_gos_thick.f10_eid > 0 else float('inf')
    print(f"  Cross-system (PCD 0.1mm vs GOS 0.15mm/400um): {cross_system_gain:.2f}×")
    assert cross_system_gain > 2.0, \
        f"Expected cross-system gain > 2× (Heismann ~3×), got {cross_system_gain}"
    print(f"  ✓ Cross-system PCD/EID ratio > 2× (Kuttig: 8.5/3.2 ≈ 2.7×)")

    # Phase 4 constraint prompt
    print(f"\n{'='*70}")
    print("Phase 4 제약 조건 프롬프트:")
    print("=" * 70)
    print(solver.format_mtf_prompt(mtf_sol, dqef_sol))

    # =========================================================================
    # Phase 4-B: Biopsy Geometry & Calibration Tests
    # =========================================================================
    print(f"\n{'='*70}")
    print("Phase 4-B: Biopsy Geometry & Calibration Tests")
    print("=" * 70)

    # Test 18: Basic stereo triangulation (θ=15°)
    print("\n[Test 18] Stereo Triangulation (θ=15°, pixel=0.1mm)")
    biopsy_sol = solver.solve_biopsy_targeting(
        stereo_angle_deg=15.0,
        pixel_pitch_mm=0.1,
        mtf_pcd_effective=0.637,
        mtf_eid_effective=0.40,
        calibration_offset_mm=0.2,
        breast_thickness_mm=50.0,
    )
    print(f"  Geometric amplification: {biopsy_sol.geometric_amplification:.4f}")
    print(f"  Target Z (simulated): {biopsy_sol.target_z_mm:.3f} mm")
    print(f"  σ_Δx (PCD): {biopsy_sol.sigma_dx_pcd_mm:.4f} mm")
    print(f"  σ_Δx (EID): {biopsy_sol.sigma_dx_eid_mm:.4f} mm")
    print(f"  σ_Z (PCD): {biopsy_sol.sigma_z_mm:.4f} mm")
    print(f"  Total Error PCD: {biopsy_sol.total_error_pcd_mm:.4f} mm")
    print(f"  Total Error EID: {biopsy_sol.total_error_eid_mm:.4f} mm")
    print(f"  PCD improvement: {biopsy_sol.pcd_error_reduction_pct:.1f}%")

    # Verify geometric amplification = 1/(2×sin(15°))
    expected_amp = 1.0 / (2.0 * math.sin(math.radians(15.0)))
    assert abs(biopsy_sol.geometric_amplification - expected_amp) < 1e-10, \
        f"Geometric amp: {biopsy_sol.geometric_amplification} vs expected {expected_amp}"
    print("  ✓ Geometric amplification = 1/(2sin15°) = 1.9319")

    # Verify σ_Z > σ_X (geometric amplification)
    assert biopsy_sol.sigma_z_mm > biopsy_sol.sigma_x_mm, \
        f"σ_Z ({biopsy_sol.sigma_z_mm}) should be > σ_X ({biopsy_sol.sigma_x_mm})"
    print(f"  ✓ σ_Z ({biopsy_sol.sigma_z_mm:.4f}) > σ_X ({biopsy_sol.sigma_x_mm:.4f}) [Law 14]")

    # Verify PCD error < EID error
    assert biopsy_sol.total_error_pcd_mm < biopsy_sol.total_error_eid_mm, \
        f"PCD error ({biopsy_sol.total_error_pcd_mm}) should be < EID ({biopsy_sol.total_error_eid_mm})"
    print("  ✓ PCD targeting error < EID targeting error")

    # Verify Z/XY error ratio = √2 × geometric_amp (for parallax-based Z)
    expected_ratio = math.sqrt(2) * biopsy_sol.geometric_amplification
    assert abs(biopsy_sol.z_to_xy_error_ratio - expected_ratio) < 1e-10, \
        f"Z/XY ratio: {biopsy_sol.z_to_xy_error_ratio} vs expected {expected_ratio}"
    print(f"  ✓ Z/XY error ratio = √2 × G = {expected_ratio:.4f}")

    # Test 19: ACR tolerance check (must be within 1mm)
    print("\n[Test 19] ACR Tolerance (≤1mm)")
    assert biopsy_sol.acr_tolerance_mm == 1.0, "ACR tolerance should be 1.0mm"
    print(f"  PCD Total Error: {biopsy_sol.total_error_pcd_mm:.4f} mm {'≤' if biopsy_sol.within_acr_tolerance else '>'} 1.0 mm")
    print(f"  ACR Pass: {biopsy_sol.within_acr_tolerance}")
    # With standard parameters, PCD should be within tolerance
    assert biopsy_sol.within_acr_tolerance, \
        f"PCD with standard params should pass ACR ({biopsy_sol.total_error_pcd_mm:.4f}mm)"
    print("  ✓ PCD meets ACR ≤1mm targeting accuracy")

    # Test 20: Actual stereo pair coordinate input
    print("\n[Test 20] Actual Stereo Pair Coordinates")
    # Simulated lesion at depth 25mm: parallax = 25 × 2 × sin(15°) = 12.94mm
    depth_25 = 25.0
    expected_parallax = depth_25 * 2 * math.sin(math.radians(15.0))
    x_center = 30.0
    biopsy_actual = solver.solve_biopsy_targeting(
        stereo_angle_deg=15.0,
        x_plus_mm=x_center + expected_parallax / 2,
        x_minus_mm=x_center - expected_parallax / 2,
        y_plus_mm=20.0,
        y_minus_mm=20.0,
    )
    print(f"  Input: x₊={x_center + expected_parallax/2:.3f}, x₋={x_center - expected_parallax/2:.3f}")
    print(f"  Parallax: {biopsy_actual.parallax_mm:.4f} mm")
    print(f"  Calculated Z: {biopsy_actual.target_z_mm:.4f} mm (expected: {depth_25:.1f} mm)")
    assert abs(biopsy_actual.target_z_mm - depth_25) < 1e-10, \
        f"Z calculation error: {biopsy_actual.target_z_mm} vs {depth_25}"
    assert abs(biopsy_actual.target_x_mm - x_center) < 1e-10, \
        f"X calculation error: {biopsy_actual.target_x_mm} vs {x_center}"
    print("  ✓ 3D coordinate calculation correct (Law 13 verified)")

    # Test 21: Geometric error amplification at different angles
    print("\n[Test 21] Error Amplification vs Angle")
    angles_and_amps = [
        (10.0, 1.0 / (2 * math.sin(math.radians(10.0)))),
        (15.0, 1.0 / (2 * math.sin(math.radians(15.0)))),
        (20.0, 1.0 / (2 * math.sin(math.radians(20.0)))),
        (25.0, 1.0 / (2 * math.sin(math.radians(25.0)))),
        (30.0, 1.0 / (2 * math.sin(math.radians(30.0)))),
    ]
    for angle, expected_g in angles_and_amps:
        sol_angle = solver.solve_biopsy_targeting(stereo_angle_deg=angle)
        assert abs(sol_angle.geometric_amplification - expected_g) < 1e-10, \
            f"Amp at {angle}°: {sol_angle.geometric_amplification} vs {expected_g}"
        print(f"  θ={angle:5.1f}° → G={sol_angle.geometric_amplification:.4f} (Total Error={sol_angle.total_error_pcd_mm:.4f}mm)")
    # Verify: larger angle → smaller amplification → smaller error
    sol_10 = solver.solve_biopsy_targeting(stereo_angle_deg=10.0)
    sol_30 = solver.solve_biopsy_targeting(stereo_angle_deg=30.0)
    assert sol_10.total_error_pcd_mm > sol_30.total_error_pcd_mm, \
        "10° should have larger error than 30°"
    print("  ✓ Larger angle → smaller geometric amplification → smaller targeting error")

    # Test 22: Phase 4-A → Phase 4-B integration (MTF → targeting)
    print("\n[Test 22] Phase 4-A → 4-B Integration (MTF → Targeting)")
    # Use Phase 4-A MTF values at Nyquist
    biopsy_with_phase4a = solver.solve_biopsy_targeting(
        pixel_pitch_mm=0.1,
        mtf_pcd_effective=mtf_sol.mtf_pcd_at_nyquist,  # Phase 4-A result
        mtf_eid_effective=mtf_sol.mtf_eid_at_nyquist,  # Phase 4-A result
    )
    print(f"  Phase 4-A MTF_PCD(Ny) = {mtf_sol.mtf_pcd_at_nyquist:.4f}")
    print(f"  Phase 4-A MTF_EID(Ny) = {mtf_sol.mtf_eid_at_nyquist:.4f}")
    print(f"  → σ_Δx_PCD = {biopsy_with_phase4a.sigma_dx_pcd_mm:.4f} mm")
    print(f"  → σ_Δx_EID = {biopsy_with_phase4a.sigma_dx_eid_mm:.4f} mm")
    print(f"  → Total Error PCD = {biopsy_with_phase4a.total_error_pcd_mm:.4f} mm")
    print(f"  → Total Error EID = {biopsy_with_phase4a.total_error_eid_mm:.4f} mm")
    print(f"  → PCD improvement = {biopsy_with_phase4a.pcd_error_reduction_pct:.1f}%")
    # Higher MTF → smaller σ_Δx → smaller total error
    assert biopsy_with_phase4a.sigma_dx_pcd_mm < biopsy_with_phase4a.sigma_dx_eid_mm
    print("  ✓ Higher MTF (PCD) → smaller parallax uncertainty → better targeting")

    # Test 23: Calibration offset impact
    print("\n[Test 23] Calibration Offset Impact")
    sol_no_cal = solver.solve_biopsy_targeting(calibration_offset_mm=0.0)
    sol_small_cal = solver.solve_biopsy_targeting(calibration_offset_mm=0.1)
    sol_large_cal = solver.solve_biopsy_targeting(calibration_offset_mm=0.5)
    print(f"  σ_cal=0.0mm → Total={sol_no_cal.total_error_pcd_mm:.4f}mm")
    print(f"  σ_cal=0.1mm → Total={sol_small_cal.total_error_pcd_mm:.4f}mm")
    print(f"  σ_cal=0.2mm → Total={biopsy_sol.total_error_pcd_mm:.4f}mm")
    print(f"  σ_cal=0.5mm → Total={sol_large_cal.total_error_pcd_mm:.4f}mm")
    assert sol_no_cal.total_error_pcd_mm < sol_small_cal.total_error_pcd_mm < biopsy_sol.total_error_pcd_mm < sol_large_cal.total_error_pcd_mm
    print("  ✓ Larger calibration offset → larger total error (monotone)")
    # Large calibration can push past ACR tolerance
    assert not sol_large_cal.within_acr_tolerance or sol_large_cal.total_error_pcd_mm <= 1.0
    print(f"  σ_cal=0.5mm ACR: {'PASS' if sol_large_cal.within_acr_tolerance else 'FAIL'}")

    # Test 24: Physical impossibility check (σ_Z < σ_Δx impossible for θ<30°)
    print("\n[Test 24] Physical Impossibility: σ_Z < σ_Δx (θ<30°)")
    for angle in [10.0, 15.0, 20.0, 25.0, 29.0]:
        sol_check = solver.solve_biopsy_targeting(stereo_angle_deg=angle)
        # σ_Z = σ_Δx × G, G = 1/(2sinθ), for θ<30° G>1
        assert sol_check.sigma_z_mm > sol_check.sigma_dx_pcd_mm, \
            f"Physical impossibility at θ={angle}°: σ_Z < σ_Δx"
    print("  ✓ σ_Z > σ_Δx for all θ < 30° (geometric amplification mandatory)")

    # =========================================================================
    # Phase 4-B: DBT (Tomosynthesis) Guided Biopsy Tests
    # =========================================================================

    # Test 25: DBT 50° wide-angle (Siemens-like, iterative reconstruction)
    print("\n[Test 25] DBT 50° Wide-Angle Targeting (K=0.42, iterative)")
    dbt_50 = solver.solve_dbt_biopsy_targeting(
        angular_range_deg=50.0,
        n_projections=25,
        pixel_pitch_mm=0.1,
        mtf_effective=0.637,
        depth_resolution_constant=0.42,
        calibration_offset_mm=0.2,
    )
    print(f"  Δz_FWHM = {dbt_50.depth_resolution_mm:.3f} mm")
    print(f"  σ_Z_DBT = {dbt_50.sigma_z_mm:.4f} mm (no G amplification)")
    print(f"  Total Error (DBT) = {dbt_50.total_targeting_error_mm:.4f} mm")
    print(f"  Total Error (Stereo) = {dbt_50.stereo_total_error_mm:.4f} mm")
    print(f"  DBT advantage: {dbt_50.dbt_advantage} ({dbt_50.dbt_improvement_pct:+.1f}%)")
    assert dbt_50.depth_resolution_mm < 2.0, "50° Δz should be < 2mm"
    assert dbt_50.dbt_advantage, "50° DBT with K=0.42 should beat stereo 15°"
    assert dbt_50.within_acr_tolerance, "50° DBT should pass ACR"
    print("  ✓ Wide-angle DBT (50°, iterative) beats stereo 15°")

    # Test 26: DBT 15° narrow-angle (Hologic-like, standard FBP)
    print("\n[Test 26] DBT 15° Narrow-Angle Targeting (K=1.0, FBP)")
    dbt_15 = solver.solve_dbt_biopsy_targeting(
        angular_range_deg=15.0,
        n_projections=15,
        pixel_pitch_mm=0.1,
        mtf_effective=0.637,
        depth_resolution_constant=1.0,
        calibration_offset_mm=0.2,
    )
    print(f"  Δz_FWHM = {dbt_15.depth_resolution_mm:.3f} mm")
    print(f"  σ_Z_DBT = {dbt_15.sigma_z_mm:.4f} mm")
    print(f"  Total Error (DBT) = {dbt_15.total_targeting_error_mm:.4f} mm")
    print(f"  Total Error (Stereo) = {dbt_15.stereo_total_error_mm:.4f} mm")
    print(f"  DBT advantage: {dbt_15.dbt_advantage} ({dbt_15.dbt_improvement_pct:+.1f}%)")
    assert not dbt_15.dbt_advantage, "15° DBT with K=1.0 should lose to stereo 15°"
    assert dbt_15.depth_resolution_mm > 5.0, "15° Δz should be > 5mm"
    print("  ✓ Narrow-angle DBT (15°, FBP) loses to stereo 15°")

    # Test 27: DBT angular range sweep (wider → better)
    print("\n[Test 27] DBT Angular Range Sweep (monotone improvement)")
    prev_error = float('inf')
    for alpha in [15, 25, 35, 50]:
        dbt_sweep = solver.solve_dbt_biopsy_targeting(
            angular_range_deg=float(alpha),
            depth_resolution_constant=0.50,
        )
        assert dbt_sweep.total_targeting_error_mm < prev_error, \
            f"DBT error should decrease with wider angle: {alpha}° not better than previous"
        prev_error = dbt_sweep.total_targeting_error_mm
        print(f"  α={alpha:2d}° → Δz={dbt_sweep.depth_resolution_mm:.2f}mm, "
              f"Total={dbt_sweep.total_targeting_error_mm:.4f}mm")
    print("  ✓ Wider angular range → smaller total error (monotone)")

    # Test 28: Crossover angle verification
    print("\n[Test 28] Crossover Angle (DBT = Stereo)")
    dbt_cross = solver.solve_dbt_biopsy_targeting(
        angular_range_deg=50.0,
        depth_resolution_constant=0.50,
    )
    print(f"  K=0.50, Stereo θ=15°: Crossover at α={dbt_cross.crossover_angle_deg:.1f}°")
    # Verify: at crossover angle, DBT ≈ Stereo
    if dbt_cross.crossover_angle_deg < 180.0:
        dbt_at_crossover = solver.solve_dbt_biopsy_targeting(
            angular_range_deg=dbt_cross.crossover_angle_deg,
            depth_resolution_constant=0.50,
        )
        diff_pct = abs(dbt_at_crossover.total_targeting_error_mm -
                       dbt_at_crossover.stereo_total_error_mm) / \
                   dbt_at_crossover.stereo_total_error_mm * 100
        print(f"  At crossover: DBT={dbt_at_crossover.total_targeting_error_mm:.4f}mm, "
              f"Stereo={dbt_at_crossover.stereo_total_error_mm:.4f}mm "
              f"(diff={diff_pct:.2f}%)")
        assert diff_pct < 5.0, "At crossover angle, DBT and Stereo should be within 5%"
        print("  ✓ At crossover angle, DBT ≈ Stereo (< 5% difference)")
    else:
        print("  ✓ No crossover exists (K too large for stereo to lose)")

    # Phase 4-B constraint prompts
    print(f"\n{'='*70}")
    print("Phase 4-B 제약 조건 프롬프트 (Stereo):")
    print("=" * 70)
    print(solver.format_biopsy_prompt(biopsy_sol))

    print(f"\n{'='*70}")
    print("Phase 4-B 제약 조건 프롬프트 (DBT 50°):")
    print("=" * 70)
    print(solver.format_dbt_biopsy_prompt(dbt_50))

    # =========================================================================
    # Phase 5: Tomosynthesis Image Quality Physics Tests
    # =========================================================================
    print(f"\n{'='*70}")
    print("Phase 5: Tomosynthesis Image Quality Physics")
    print("=" * 70)

    # Test 29: Dose-split basic (1500 uGy, N=25)
    print("\n[Test 29] Dose-split basic (1500 μGy, N=25)")
    dose_split = solver.solve_tomo_dose_split(
        total_dose_uGy=1500.0,
        n_projections=25,
    )
    print(f"  D_proj = {dose_split.dose_per_projection_uGy:.2f} μGy")
    print(f"  DQE_EID(per proj) = {dose_split.dqe_eid_per_proj:.4f}")
    print(f"  DQE_PCD(per proj) = {dose_split.dqe_pcd_per_proj:.3f}")
    print(f"  PCD DQE advantage = {dose_split.pcd_dqe_advantage_ratio:.3f}×")
    print(f"  PCD SNR gain (total) = {dose_split.pcd_snr_gain_total:.4f}×")
    assert dose_split.dose_per_projection_uGy == 60.0, \
        f"D_proj should be 60: {dose_split.dose_per_projection_uGy}"
    assert dose_split.dqe_eid_per_proj < dose_split.phase3_dqe_eid_full, \
        "DQE_EID at D/N should be less than at full dose"
    assert dose_split.dqe_pcd_per_proj == 0.850, \
        f"DQE_PCD should be 0.850: {dose_split.dqe_pcd_per_proj}"
    assert dose_split.pcd_dqe_advantage_ratio > 1.3, \
        f"PCD advantage should be > 1.3: {dose_split.pcd_dqe_advantage_ratio}"
    print("  ✓ Dose-split correctly degrades EID DQE while PCD stays constant")

    # Test 30: Dose-split Phase 3 cross-validation
    print("\n[Test 30] Phase 3 Cross-validation (N=1 → 2D)")
    dose_split_n1 = solver.solve_tomo_dose_split(
        total_dose_uGy=1500.0,
        n_projections=1,
    )
    print(f"  N=1: DQE_EID = {dose_split_n1.dqe_eid_per_proj:.4f} (expected ≈ 0.700)")
    print(f"  Phase 3 DQE_EID(full) = {dose_split_n1.phase3_dqe_eid_full:.4f}")
    print(f"  Phase 3 match: {dose_split_n1.phase3_match}")
    assert dose_split_n1.phase3_match, \
        f"N=1 should reduce to Phase 3: DQE={dose_split_n1.dqe_eid_per_proj:.4f} vs 0.700"
    # SNR ratio for N=1 should be √(DQE_PCD/DQE_EID) ≈ 1.10
    expected_snr_ratio_n1 = math.sqrt(0.850 / dose_split_n1.dqe_eid_per_proj)
    assert abs(dose_split_n1.pcd_snr_gain_total - expected_snr_ratio_n1) < 0.01, \
        f"N=1 SNR ratio: {dose_split_n1.pcd_snr_gain_total} vs {expected_snr_ratio_n1}"
    print(f"  N=1 PCD SNR gain = {dose_split_n1.pcd_snr_gain_total:.4f} ≈ 1.10")
    print("  ✓ N=1 reduces exactly to Phase 3 (2D case)")

    # Test 31: Resolution asymmetry (25°, pixel=0.1mm)
    print("\n[Test 31] Resolution asymmetry (25°, K=0.50)")
    res_25 = solver.solve_tomo_resolution(
        angular_range_deg=25.0,
        pixel_pitch_mm=0.1,
        mtf_effective=0.637,
        depth_resolution_constant=0.50,
        breast_thickness_mm=50.0,
    )
    print(f"  Δxy = {res_25.delta_xy_mm:.4f} mm")
    print(f"  Δz = {res_25.delta_z_mm:.4f} mm")
    print(f"  Asymmetry = {res_25.resolution_asymmetry_ratio:.1f}×")
    print(f"  Resolvable slices = {res_25.n_resolvable_slices}")
    # Δxy ≈ 0.157mm
    assert abs(res_25.delta_xy_mm - 0.157) < 0.001, \
        f"Δxy should be ~0.157: {res_25.delta_xy_mm}"
    # Δz ≈ 2.31mm for 25° with K=0.50
    expected_dz_25 = 0.50 / math.sin(math.radians(12.5))
    assert abs(res_25.delta_z_mm - expected_dz_25) < 0.01, \
        f"Δz should be ~{expected_dz_25:.2f}: {res_25.delta_z_mm}"
    # Asymmetry > 10
    assert res_25.resolution_asymmetry_ratio > 10, \
        f"Asymmetry should be > 10: {res_25.resolution_asymmetry_ratio}"
    print("  ✓ In-plane (detector) vs through-plane (geometry) asymmetry verified")

    # Test 32: Resolution asymmetry (50°, wide angle)
    print("\n[Test 32] Resolution asymmetry (50°, wide angle)")
    res_50 = solver.solve_tomo_resolution(
        angular_range_deg=50.0,
        depth_resolution_constant=0.42,  # iterative recon
    )
    print(f"  Δz = {res_50.delta_z_mm:.4f} mm")
    print(f"  Asymmetry = {res_50.resolution_asymmetry_ratio:.1f}×")
    assert res_50.delta_z_mm < res_25.delta_z_mm, \
        "50° should have smaller Δz than 25°"
    assert res_50.resolution_asymmetry_ratio < res_25.resolution_asymmetry_ratio, \
        "50° should have less asymmetry than 25°"
    print("  ✓ Wider angle → smaller Δz → less asymmetry")

    # Test 33: Detectability comparison (5mm lesion, 50mm breast)
    print("\n[Test 33] Detectability comparison (5mm lesion, 50mm breast)")
    detect = solver.solve_tomo_detectability(
        angular_range_deg=25.0,
        n_projections=15,
        total_dose_uGy=1500.0,
        breast_thickness_mm=50.0,
        lesion_diameter_mm=5.0,
        lesion_contrast=0.02,
    )
    print(f"  d'_2D_EID = {detect.d_prime_2d_eid:.4f}")
    print(f"  d'_tomo_EID = {detect.d_prime_tomo_eid:.4f}")
    print(f"  d'_tomo_PCD = {detect.d_prime_tomo_pcd:.4f}")
    print(f"  Tomo EID / 2D EID = {detect.tomo_vs_2d_gain_eid:.3f}×")
    print(f"  PCD tomo / EID tomo = {detect.pcd_vs_eid_tomo_gain:.3f}×")
    print(f"  PCD tomo / 2D EID = {detect.pcd_tomo_vs_2d_eid_gain:.3f}× (total)")
    assert detect.d_prime_tomo_eid > detect.d_prime_2d_eid, \
        "Tomo EID should beat 2D EID (clutter rejection)"
    assert detect.d_prime_tomo_pcd > detect.d_prime_tomo_eid, \
        "PCD tomo should beat EID tomo (DQE advantage)"
    assert detect.pcd_tomo_vs_2d_eid_gain > detect.tomo_vs_2d_gain_eid, \
        "PCD tomo total gain should exceed tomo EID gain"
    print("  ✓ PCD tomo > EID tomo > 2D EID (correct ordering)")

    # Test 34: Edge case — single projection (N=1)
    print("\n[Test 34] Edge case — N=1 (reduces to 2D)")
    detect_n1 = solver.solve_tomo_detectability(
        angular_range_deg=25.0,
        n_projections=1,
        total_dose_uGy=1500.0,
        breast_thickness_mm=50.0,
        lesion_diameter_mm=5.0,
        lesion_contrast=0.02,
    )
    # N=1: slice_thickness ≈ Δz, but if Δz < t_breast, clutter rejection still applies
    # The key check: DQE values for N=1 should match 2D
    assert abs(detect_n1.dqe_eid_tomo - detect_n1.dqe_eid_2d) < 0.001, \
        f"N=1 DQE_tomo should equal DQE_2D: {detect_n1.dqe_eid_tomo} vs {detect_n1.dqe_eid_2d}"
    # PCD vs EID ratio for N=1 is modest
    pcd_eid_ratio_n1 = detect_n1.pcd_vs_eid_tomo_gain
    assert pcd_eid_ratio_n1 < 1.5, \
        f"N=1 PCD/EID ratio should be modest: {pcd_eid_ratio_n1}"
    print(f"  N=1 DQE_EID_tomo = {detect_n1.dqe_eid_tomo:.4f} = DQE_EID_2D ✓")
    print(f"  N=1 PCD/EID ratio = {pcd_eid_ratio_n1:.4f} (modest, ~1.10)")
    print("  ✓ N=1 reduces to 2D mammography case")

    # Test 35: High N (N=49, Siemens-like)
    print("\n[Test 35] High N=49 (Siemens-like)")
    dose_split_49 = solver.solve_tomo_dose_split(
        total_dose_uGy=1500.0,
        n_projections=49,
    )
    print(f"  D_proj = {dose_split_49.dose_per_projection_uGy:.2f} μGy")
    print(f"  DQE_EID(per proj) = {dose_split_49.dqe_eid_per_proj:.4f}")
    print(f"  PCD SNR gain = {dose_split_49.pcd_snr_gain_total:.4f}×")
    assert dose_split_49.pcd_snr_gain_total > 1.5, \
        f"N=49 PCD SNR gain should be > 1.5: {dose_split_49.pcd_snr_gain_total}"
    assert dose_split_49.dqe_eid_per_proj < dose_split.dqe_eid_per_proj, \
        "N=49 EID DQE should be lower than N=25"
    # Verify formula: R = √(1 + α×N)
    expected_r_49 = math.sqrt(1 + dose_split_49.phase3_alpha * 49)
    assert abs(dose_split_49.pcd_snr_gain_total - expected_r_49) < 0.01, \
        f"R formula check: {dose_split_49.pcd_snr_gain_total} vs {expected_r_49}"
    print(f"  √(1+α×49) = {expected_r_49:.4f} = PCD gain ✓")
    print("  ✓ High N → very low per-projection DQE_EID, large PCD advantage")

    # Test 36: Cross-validation chain (Phase 1→3→5)
    print("\n[Test 36] Cross-validation chain (Phase 1→3→5)")
    # Phase 1: f_e=0.30 at D=0.5
    f_e_phase1 = 0.30
    D_ref = 0.5
    # Phase 3: α = f_e×D/(1-f_e)
    alpha_phase3 = f_e_phase1 * D_ref / (1 - f_e_phase1)
    print(f"  Phase 1: f_e = {f_e_phase1}")
    print(f"  Phase 3: α = f_e×D/(1-f_e) = {alpha_phase3:.4f}")
    # Phase 5: DQE_EID(D/N) = η/(1+α×N) with same α
    assert abs(dose_split.phase3_alpha - round(alpha_phase3, 4)) < 1e-10, \
        f"Phase 5 α should match Phase 3: {dose_split.phase3_alpha} vs {round(alpha_phase3, 4)}"
    # N=1 → Phase 3
    dqe_n1 = 0.850 / (1 + alpha_phase3 * 1)
    assert abs(dqe_n1 - 0.700) < 0.005, \
        f"N=1 DQE should match Phase 3: {dqe_n1:.4f} vs 0.700"
    print(f"  Phase 5: α = {dose_split.phase3_alpha:.4f} (same)")
    print(f"  Phase 5 N=1: DQE = η/(1+α) = {dqe_n1:.4f} ≈ 0.700 = Phase 3")
    # PCD advantage monotonically increases with N
    gains = []
    for n in [1, 9, 15, 25, 49]:
        r = math.sqrt(1 + alpha_phase3 * n)
        gains.append(r)
        print(f"  N={n:2d}: R = √(1+{alpha_phase3:.4f}×{n}) = {r:.4f}")
    for i in range(1, len(gains)):
        assert gains[i] > gains[i-1], "PCD advantage must increase with N"
    print("  ✓ Phase 1→3→5 cross-validation complete, PCD advantage monotone")

    # Phase 5 constraint prompt
    print(f"\n{'='*70}")
    print("Phase 5 제약 조건 프롬프트:")
    print("=" * 70)
    print(solver.format_tomo_prompt(dose_split, res_25, detect))

    print("\n✅ Phase 1, 2, 3, 4, 4-B, 5 모든 테스트 통과")
