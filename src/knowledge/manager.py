"""
Sophia AI: Knowledge Manager
============================
동적 지식 주입을 위한 관리자

질문에서 키워드를 감지하여 관련 물리 지식만 선택적으로 로드합니다.
모든 지식을 한 번에 보내지 않아 컨텍스트 노이즈를 줄입니다.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


class KnowledgeManager:
    """
    물리 지식 동적 로더

    질문에서 키워드를 감지하여 관련 지식 모듈만 로드합니다.
    """

    def __init__(self, knowledge_base_dir: Optional[Path] = None):
        """
        Args:
            knowledge_base_dir: 지식 베이스 루트 디렉토리 경로 (하위 폴더 자동 스캔)
        """
        if knowledge_base_dir is None:
            # 기본 경로: data/knowledge/ (모든 하위 디렉토리 스캔)
            self.knowledge_base_dir = Path(__file__).parent.parent.parent / "data" / "knowledge"
        else:
            self.knowledge_base_dir = Path(knowledge_base_dir)

        self._cache: Dict[str, Dict] = {}
        self._load_all_metadata()

    def _load_all_metadata(self):
        """모든 지식 파일의 메타데이터(키워드) 로드 (하위 디렉토리 포함)"""
        self._keyword_map: Dict[str, List[str]] = {}  # keyword -> [file_id, ...]

        if not self.knowledge_base_dir.exists():
            logger.warning(f"Knowledge base directory not found: {self.knowledge_base_dir}")
            return

        # 모든 하위 디렉토리에서 JSON 파일 로드
        for json_file in self.knowledge_base_dir.glob("**/*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)

                file_id = data.get("id", json_file.stem)
                keywords = data.get("keywords", [])

                # 키워드 매핑 (다중 모듈 지원)
                for kw in keywords:
                    kw_lower = kw.lower()
                    if kw_lower not in self._keyword_map:
                        self._keyword_map[kw_lower] = []
                    if file_id not in self._keyword_map[kw_lower]:
                        self._keyword_map[kw_lower].append(file_id)

                # 캐시에 저장
                self._cache[file_id] = data

                logger.debug(f"Loaded knowledge: {file_id} with {len(keywords)} keywords")

            except Exception as e:
                logger.error(f"Failed to load {json_file}: {e}")

    def get_relevant_knowledge(
        self,
        query: str,
        max_modules: int = 3,  # Phase 7.4: 3개로 증가
        min_score_threshold: float = 0.15  # Phase 7.18: 최소 관련성 점수
    ) -> List[Dict[str, Any]]:
        """
        질문에 관련된 지식 모듈 검색 (Phase 7.4: 하드웨어 우선순위 적용)

        Args:
            query: 사용자 질문
            max_modules: 최대 반환 모듈 수
            min_score_threshold: 최소 관련성 점수 (이하면 제외)

        Returns:
            관련 지식 모듈 리스트
        """
        query_lower = query.lower()

        # Phase 7.18: 모듈별 매칭 점수 계산 (키워드 매칭 횟수 기반)
        module_scores: Dict[str, float] = {}
        query_keywords = set(query_lower.split())

        for keyword, file_ids in self._keyword_map.items():
            if keyword in query_lower:
                for file_id in file_ids:
                    if file_id not in module_scores:
                        module_scores[file_id] = 0.0
                    # 긴 키워드에 가중치 부여 (더 구체적인 매칭)
                    keyword_weight = min(len(keyword) / 10.0, 1.0)
                    module_scores[file_id] += keyword_weight

        # 점수 정규화 (최대 점수 기준)
        if module_scores:
            max_score = max(module_scores.values())
            if max_score > 0:
                for file_id in module_scores:
                    module_scores[file_id] /= max_score

        # Phase 7.18: threshold 이상인 모듈만 선택
        matched_ids = set()
        for file_id, score in module_scores.items():
            if score >= min_score_threshold:
                matched_ids.add(file_id)
            else:
                logger.debug(f"Module {file_id} excluded: score {score:.2f} < threshold {min_score_threshold}")

        # Phase 7.4: 하드웨어 키워드 감지 시 detector_physics 우선순위 부여
        hardware_keywords = [
            'dqe', 'mtf', 'a-se', 'csi', 'selenium', '셀레늄',
            '검출기', 'detector', '직접변환', '간접변환',
            'kvp', 'mas', 'filter', '필터', 'target',
            '제조', 'manufacturer', 'design', '설계', '510k'
        ]
        is_hardware_query = any(kw in query_lower for kw in hardware_keywords)

        # Phase 7.8: PCD 저선량 SNR 키워드 감지 시 pcd_low_dose_snr 우선순위 부여
        pcd_snr_keywords = [
            'pcd', 'eid', '광자 계수', 'photon counting', '전자 노이즈',
            'electronic noise', '저선량', 'low dose', '에너지 문턱치',
            'threshold', '노이즈 플로어', 'noise floor', 'd-prime', "d'",
            '미세 석회화', 'microcalcification', 'rose criterion'
        ]
        is_pcd_snr_query = any(kw in query_lower for kw in pcd_snr_keywords)

        # Phase 3: DQE/NPS 키워드 감지
        pcd_dqe_keywords = [
            'dqe', 'detective quantum efficiency', 'nps', 'noise power spectrum',
            'neq', 'noise equivalent quanta', 'dqe 선량', '정보 전달 효율',
            'dqe advantage', '저선량 dqe', 'dqe degradation', '선량 의존',
            '선량 독립', '흡수 효율', 'η_abs', 'dose dependence', 'dose independent'
        ]
        is_pcd_dqe_query = any(kw in query_lower for kw in pcd_dqe_keywords)

        # Phase 4: MTF / Spatial Resolution 키워드 감지
        pcd_mtf_keywords = [
            'mtf', 'modulation transfer function', 'spatial resolution', '공간 해상도',
            'nyquist', 'pixel size', 'pixel pitch', 'charge sharing', '전하 공유',
            'dqe frequency', 'dqe(f)', 'sinc', 'aperture', 'scintillator blur',
            '섬광체', 'direct conversion', '직접 변환', 'indirect conversion',
            'anti-charge sharing', 'acs', 'resolution limit', 'lp/mm', 'line pairs',
            '해상도 한계', '간접변환'
        ]
        is_pcd_mtf_query = any(kw in query_lower for kw in pcd_mtf_keywords)

        # Phase 5: DBT Image Quality 키워드 감지
        tomo_iq_keywords = [
            'tomosynthesis image quality', '토모합성 영상 품질',
            'dose split', '선량 분할', 'dose per projection', '투영당 선량',
            'artifact spread function', 'asf', '아티팩트 확산',
            '3d dqe', '3d nps', '3d mtf', 'tomo dqe', 'tomo snr',
            'slice thickness', '슬라이스 두께', '슬라이스',
            'anatomical clutter', '해부학적 잡음', 'structured noise', '구조적 잡음',
            'depth discrimination', '깊이 분별', 'tissue overlap', '조직 중첩',
            'projection dose', 'low dose per view',
            'resolution asymmetry', '분해능 비대칭',
            'detectability', '검출능', 'rose criterion',
            'tomo advantage', 'tomo vs mammography',
            'voxel', '복셀', 'reconstruction noise',
            'clutter rejection', '잡음 제거',
            'dbt image quality', 'dbt snr', 'dbt dose',
            'tomosynthesis dose', 'tomo resolution',
            'dbt', '토모신세시스', 'digital breast tomosynthesis',
            '재구성', 'reconstruction', '영상 재구성',
            'fbp', 'filtered back projection', '역투영',
            'ghosting', 'blurring', '번짐',
            '투영 각도', 'projection angle', '투영각',
            'missing cone', '제한 각도', 'limited angle',
            'psf', 'point spread function', 'sidelobe',
            'ramp filter', '토모합성', 'tomosynthesis',
            'iterative reconstruction', '반복 재구성',
            'through-plane', 'in-plane', 'z축 분해능'
        ]
        is_tomo_iq_query = any(kw in query_lower for kw in tomo_iq_keywords)

        # Phase 4-B: Biopsy Geometry & Calibration 키워드 감지
        biopsy_keywords = [
            'stereotactic', 'biopsy', '생검', '정위', '스테레오',
            'parallax', '시차', 'triangulation', '삼각측량',
            'targeting accuracy', '타겟팅', 'needle', '바늘',
            'depth', '깊이', 'z-axis', 'z축', 'z axis',
            'calibration', '교정', '보정',
            'acr tolerance', '허용 오차',
            'geometric amplification', '기하학적 증폭',
            'error propagation', '오차 전파',
            'stereo angle', 'stereo pair', '15도',
            'breast biopsy', '유방 생검', 'vacuum assisted',
            'coordinate', '좌표',
            'localization error', '위치 결정 오차', 'localization accuracy',
            'dbt guided', 'tomosynthesis guided', '토모합성 유도',
            'depth resolution', '깊이 해상도', 'angular range',
            'acquisition geometry', '획득 기하학'
        ]
        is_biopsy_query = any(kw in query_lower for kw in biopsy_keywords)

        # Phase 2: PCD Spectral Contrast 키워드 감지
        pcd_contrast_keywords = [
            'cesm', 'spectral', '스펙트럼', '조영증강', 'contrast enhanced',
            'contrast agent', '조영제', 'iodine', '아이오딘', '요오드',
            'k-edge', 'energy weighting', '에너지 가중', 'energy bin', '에너지 빈',
            'material decomposition', '물질 분해', 'dual energy', '이중 에너지',
            '대조도 향상', 'cnr improvement', 'spectral mammography'
        ]
        is_pcd_contrast_query = any(kw in query_lower for kw in pcd_contrast_keywords)

        # Phase 7.21: CEM Detector Physics 키워드 감지 (Ghosting/Lag/QDE in CEM context)
        # 중요: DBT의 ghosting (재구성 아티팩트)과 CEM의 ghosting (검출기 민감도)을 구분
        cem_detector_keywords = [
            'cem', '조영증강유방촬영', 'contrast enhanced mammography',
            'cem ghosting', 'cem 잔상', 'cem lag', 'cem qde',
            'dual-energy', 'dual energy', 'le he', 'le/he', '저에너지 고에너지',
            'qde 에너지', 'qde energy', 'qde(le)', 'qde(he)', 'qe(le)', 'qe(he)',
            '28kvp', '49kvp', '28 kvp', '49 kvp', 'w/rh', 'w/ti',
            'gain map', '이득 맵', '게인 맵', 'gain mismatch',
            '민감도 감소', 'sensitivity change', 'sensitivity reduction',
            '트랩 전자', 'trapped electron', '홀 재결합', 'hole recombination'
        ]
        is_cem_detector_query = any(kw in query_lower for kw in cem_detector_keywords)

        # CEM + ghosting/lag 조합 감지 (DBT ghosting과 구분)
        is_cem_ghosting_context = (
            is_cem_detector_query or
            (any(kw in query_lower for kw in ['cem', '조영증강', 'dual', 'qde', 'qe ']) and
             any(kw in query_lower for kw in ['ghosting', '잔상', 'ghost', 'lag', '지연']))
        )

        # Phase 1-New: X-ray Tube Physics 키워드 감지
        xray_tube_keywords = [
            'x-ray tube', 'x선관', '엑스선관', 'target', '타겟', '양극',
            'molybdenum', '몰리브덴', 'rhodium', '로듐', 'tungsten', '텅스텐',
            'focal spot', '초점', 'anode', 'cathode', 'heel effect', '힐 효과',
            'characteristic radiation', '특성방사선', 'bremsstrahlung', '제동방사선'
        ]
        is_xray_tube_query = any(kw in query_lower for kw in xray_tube_keywords)

        # Phase 1-New: Filtration & HVL 키워드 감지
        filtration_keywords = [
            'hvl', 'half value layer', '반가층', 'beam quality', '빔 품질',
            'beam hardening', '빔 경화', 'inherent filtration', '고유여과',
            'added filtration', '추가여과', 'total filtration',
            'target filter combination', '타겟 필터 조합', 'mo/mo', 'mo/rh', 'w/rh', 'w/ag'
        ]
        is_filtration_query = any(kw in query_lower for kw in filtration_keywords)

        # Phase 1-New: Exposure/AEC 키워드 감지
        exposure_keywords = [
            'aec', 'automatic exposure', '자동노출', 'phototimer', '포토타이머',
            'exposure index', '노출지수', 'deviation index', 'backup time', '백업타임',
            'exposure factors', '촬영조건', '노출인자', 'technique chart'
        ]
        is_exposure_query = any(kw in query_lower for kw in exposure_keywords)

        # Phase 1-New: Scatter Radiation 키워드 감지
        scatter_keywords = [
            'scatter', '산란', 'scattered radiation', '산란선',
            'compton', '콤프턴', 'scatter to primary', 'spr', '산란선비',
            'scatter fraction', '산란분율', 'contrast degradation',
            'thick breast', '두꺼운 유방', '6cm', '두께 6', '두께6',
            'cnr 저하', '대조도 손실', 'contrast loss'
        ]
        is_scatter_query = any(kw in query_lower for kw in scatter_keywords)

        # Phase 1-New: Grid/Bucky 키워드 감지
        grid_keywords = [
            'grid', '그리드', 'bucky', '버키', 'bucky factor', '버키 팩터',
            'grid ratio', '그리드 비', 'grid frequency',
            'contrast improvement factor', 'cif', '대조도 개선',
            'selectivity', '선택성', 'grid cutoff', '그리드 컷오프',
            'moving grid', '이동 그리드', 'primary transmission'
        ]
        is_grid_query = any(kw in query_lower for kw in grid_keywords)

        # Phase 1-New: Air Gap/Magnification 키워드 감지
        airgap_keywords = [
            'air gap', '에어갭', '공기갭', 'magnification', '확대',
            'mag view', '확대 촬영', 'geometric blur', '기하학적 흐림',
            'penumbra', '반영', 'oid', 'object to image',
            'contact mammography', '접촉 촬영'
        ]
        is_airgap_query = any(kw in query_lower for kw in airgap_keywords)

        # Phase 1-Thermal: X-ray Tube Thermal Capacity 키워드 감지
        thermal_capacity_keywords = [
            '열용량', 'thermal capacity', 'heat capacity', 'heat unit', 'hu',
            '열적 한계', 'thermal limit', '열부하', 'heat loading', 'heat load',
            '초점 열', 'focal spot heat', '냉각', 'cooling',
            'ma 제한', 'tube current limit', 'ma limitation',
            '양극 열', 'anode heat', '노출 시간 증가', 'exposure time increase',
            '소초점 제한', 'small focal spot limitation',
            '자글자글', 'grainy', '확대촬영 노이즈', 'magnification noise'
        ]
        is_thermal_capacity_query = any(kw in query_lower for kw in thermal_capacity_keywords)

        # ================================================================
        # Regulatory Standards 키워드 감지 (IEC, FDA, 식약처)
        # ================================================================
        regulatory_keywords = [
            'iec', 'iec 60601', '60601-2-45', '62220', '61223',
            'fda', 'mqsa', '510k', '510(k)', 'pma', 'premarket',
            '식약처', 'mfds', 'kfda', 'gmp', '허가', '인허가', '인증',
            '규제', 'regulation', 'standard', '표준', 'certification',
            '품목허가', 'ce 마크', 'ce marking', '수락 시험', 'acceptance test',
            '유방 치밀도 고지', 'density notification', 'breast density notification'
        ]
        is_regulatory_query = any(kw in query_lower for kw in regulatory_keywords)

        # AAPM QC 키워드 감지
        aapm_qc_keywords = [
            'aapm', 'tg-18', 'tg18', 'report 29', 'report 270', 'mppg 17',
            'display qc', 'display qa', 'monitor qc', '디스플레이 품질',
            'luminance', '휘도', 'gsdf', 'grayscale standard',
            'physicist survey', '의학물리사 서베이', 'annual survey',
            'acr phantom', '팬텀 점수', 'phantom score',
            'fiber score', 'speck score', 'mass score',
            'weekly qc', 'daily qc', 'monthly qc', 'qc 체크리스트'
        ]
        is_aapm_qc_query = any(kw in query_lower for kw in aapm_qc_keywords)

        # ================================================================
        # Phase 7.10: Material vs Lesion 분리 모듈 키워드 감지
        # ================================================================

        # CsI 검출기 소재 (조영제 Iodine과 구분)
        csi_detector_keywords = [
            'csi 검출기', 'csi scintillator', '섬광체', 'csi:tl',
            '간접변환', 'indirect conversion', 'eid 검출기', 'eid detector',
            '검출기 k-edge', 'detector k-edge', '검출기 흡수', 'absorption depth'
        ]
        is_csi_detector_query = any(kw in query_lower for kw in csi_detector_keywords)

        # 석회화 대조도 (Calcium, 조영제 Iodine 아님)
        calcification_keywords = [
            '석회화 대조', 'calcification contrast', 'calcium contrast',
            '미세석회화', 'microcalcification', 'hydroxyapatite',
            'fine linear', 'amorphous', 'pleomorphic', 'bi-rads 4',
            'birads 4', '4c', '4b', '석회화 형태', 'calcification morphology'
        ]
        is_calcification_query = any(kw in query_lower for kw in calcification_keywords)

        # 확대 촬영 기하학 (penumbra, 초점 크기)
        magnification_geo_keywords = [
            'penumbra', '반그림자', '기하학적 블러', 'geometric blur',
            '초점 크기', 'focal spot size', '소초점', 'small focus',
            '1.8배 확대', '1.5배 확대', '2배 확대', 'magnification factor',
            'sid', 'sod', 'oid', '기하학적 확대', 'geometric magnification'
        ]
        is_magnification_geo_query = any(kw in query_lower for kw in magnification_geo_keywords)

        # 섬광체 빛 확산 (MTF 에너지 의존성)
        light_spread_keywords = [
            '빛 확산', 'light spread', 'light diffusion', 'columnar structure',
            '주상 구조', '흡수 깊이', 'absorption depth', 'interaction depth',
            '에너지 의존 mtf', 'energy-dependent mtf', 'scintillator blur',
            '섬광체 mtf', 'csi mtf'
        ]
        is_light_spread_query = any(kw in query_lower for kw in light_spread_keywords)

        # 시스템 MTF 체인 (cascade MTF)
        mtf_chain_keywords = [
            '시스템 mtf', 'system mtf', 'mtf chain', 'cascaded mtf',
            'total mtf', 'mtf 합성', 'mtf product', 'mtf 곱',
            '초점 mtf', 'focal mtf', '검출기 mtf', 'detector mtf',
            '움직임 블러', 'motion blur', 'resolution chain'
        ]
        is_mtf_chain_query = any(kw in query_lower for kw in mtf_chain_keywords)

        # Fine Linear → Amorphous 형태 오분류 감지
        morphology_confusion_keywords = [
            'fine linear', 'amorphous', '뭉개', '형태 왜곡', 'morphology confusion',
            '오분류', 'misclassification', '4c', '4b', 'birads 4c', 'birads 4b',
            '형태 손실', 'shape degradation'
        ]
        is_morphology_query = any(kw in query_lower for kw in morphology_confusion_keywords)

        # 매칭된 모듈 정렬
        matched_list = list(matched_ids)

        # Phase 3: DQE/NPS 질문 시 pcd_dqe_nps 우선
        if is_pcd_dqe_query and 'pcd_dqe_nps' in matched_list:
            matched_list.remove('pcd_dqe_nps')
            matched_list.insert(0, 'pcd_dqe_nps')
            logger.info("Phase 3: DQE/NPS query detected - prioritizing pcd_dqe_nps")

        # Phase 2: PCD Contrast 질문 시 pcd_spectral_contrast 우선
        if is_pcd_contrast_query and 'pcd_spectral_contrast' in matched_list:
            matched_list.remove('pcd_spectral_contrast')
            matched_list.insert(0, 'pcd_spectral_contrast')
            logger.info("Phase 2: PCD Contrast query detected - prioritizing pcd_spectral_contrast")

        # PCD SNR 질문 시 pcd_low_dose_snr 우선
        if is_pcd_snr_query and 'pcd_low_dose_snr' in matched_list:
            matched_list.remove('pcd_low_dose_snr')
            matched_list.insert(0, 'pcd_low_dose_snr')
            logger.info("Phase 7.8: PCD SNR query detected - prioritizing pcd_low_dose_snr")

        # 하드웨어 질문 시 detector_physics 우선
        if is_hardware_query and 'detector_physics' in matched_list:
            matched_list.remove('detector_physics')
            matched_list.insert(0, 'detector_physics')  # 맨 앞으로
            logger.info("Phase 7.4: Hardware query detected - prioritizing detector_physics")

        # Phase 4: MTF/Resolution 질문 시 pcd_mtf_resolution 최우선
        # (하드웨어보다 후순위에 배치하여 MTF 질문 시 hardware를 override)
        if is_pcd_mtf_query and 'pcd_mtf_resolution' in matched_list:
            matched_list.remove('pcd_mtf_resolution')
            matched_list.insert(0, 'pcd_mtf_resolution')
            logger.info("Phase 4: MTF/Resolution query detected - prioritizing pcd_mtf_resolution")

        # Phase 4-B: Biopsy 질문 시 biopsy_geometry_calibration 최우선
        if is_biopsy_query and 'biopsy_geometry_calibration' in matched_list:
            matched_list.remove('biopsy_geometry_calibration')
            matched_list.insert(0, 'biopsy_geometry_calibration')
            logger.info("Phase 4-B: Biopsy query detected - prioritizing biopsy_geometry_calibration")

        # Phase 5: Tomo IQ 질문 시 dbt_image_quality 최우선
        # 단, CEM 맥락에서는 detector_physics가 더 우선 (Phase 7.21에서 override)
        if is_tomo_iq_query and 'dbt_image_quality' in matched_list and not is_cem_ghosting_context:
            matched_list.remove('dbt_image_quality')
            matched_list.insert(0, 'dbt_image_quality')
            logger.info("Phase 5: Tomo IQ query detected - prioritizing dbt_image_quality")

        # Phase 1-New: X-ray Tube 질문 시 xray_tube_physics 우선
        if is_xray_tube_query and 'xray_tube_physics' in matched_list:
            matched_list.remove('xray_tube_physics')
            matched_list.insert(0, 'xray_tube_physics')
            logger.info("X-ray Tube query detected - prioritizing xray_tube_physics")

        # Phase 1-New: Filtration/HVL 질문 시 filtration_hvl 우선
        if is_filtration_query and 'filtration_hvl' in matched_list:
            matched_list.remove('filtration_hvl')
            matched_list.insert(0, 'filtration_hvl')
            logger.info("Filtration/HVL query detected - prioritizing filtration_hvl")

        # Phase 1-New: Exposure/AEC 질문 시 exposure_factors 우선
        if is_exposure_query and 'exposure_factors' in matched_list:
            matched_list.remove('exposure_factors')
            matched_list.insert(0, 'exposure_factors')
            logger.info("Exposure/AEC query detected - prioritizing exposure_factors")

        # Phase 1-New: Scatter 질문 시 scatter_radiation 우선
        if is_scatter_query and 'scatter_radiation' in matched_list:
            matched_list.remove('scatter_radiation')
            matched_list.insert(0, 'scatter_radiation')
            logger.info("Scatter query detected - prioritizing scatter_radiation")

        # Phase 1-New: Grid/Bucky 질문 시 antiscatter_grid 최우선 (가장 마지막에 체크하여 최고 우선순위)
        if is_grid_query and 'antiscatter_grid' in matched_list:
            matched_list.remove('antiscatter_grid')
            matched_list.insert(0, 'antiscatter_grid')
            logger.info("Grid/Bucky query detected - prioritizing antiscatter_grid")

        # Phase 1-New: Air Gap/Magnification 질문 시 air_gap_technique 최우선
        if is_airgap_query and 'air_gap_technique' in matched_list:
            matched_list.remove('air_gap_technique')
            matched_list.insert(0, 'air_gap_technique')
            logger.info("Air Gap/Magnification query detected - prioritizing air_gap_technique")

        # ================================================================
        # Regulatory Standards 질문 시 해당 모듈 우선
        # ================================================================
        if is_regulatory_query:
            regulatory_modules = ['iec_mammography_standards', 'fda_mqsa', 'mfds_guidelines']
            for mod in regulatory_modules:
                if mod in matched_list:
                    matched_list.remove(mod)
                    matched_list.insert(0, mod)
            # 매칭 안 됐어도 강제 추가 (규제 질문은 해당 모듈 필수)
            for mod in regulatory_modules:
                if mod not in matched_list and mod in self._cache:
                    if 'iec' in query_lower and mod == 'iec_mammography_standards':
                        matched_list.insert(0, mod)
                    elif ('fda' in query_lower or 'mqsa' in query_lower) and mod == 'fda_mqsa':
                        matched_list.insert(0, mod)
                    elif ('식약처' in query_lower or 'mfds' in query_lower or 'gmp' in query_lower) and mod == 'mfds_guidelines':
                        matched_list.insert(0, mod)
            logger.info("Regulatory query detected - prioritizing regulatory modules")

        # ================================================================
        # AAPM QC 질문 시 해당 모듈 우선
        # ================================================================
        if is_aapm_qc_query:
            if 'aapm_mammography_qc' in matched_list:
                matched_list.remove('aapm_mammography_qc')
                matched_list.insert(0, 'aapm_mammography_qc')
            elif 'aapm_mammography_qc' in self._cache:
                matched_list.insert(0, 'aapm_mammography_qc')
            logger.info("AAPM QC query detected - prioritizing aapm_mammography_qc")

        # ================================================================
        # Phase 7.10: Material vs Lesion 분리 모듈 우선순위
        # ================================================================

        # CsI 검출기 소재 질문 시 csi_detector_physics 우선
        if is_csi_detector_query and 'csi_detector_physics' in matched_list:
            matched_list.remove('csi_detector_physics')
            matched_list.insert(0, 'csi_detector_physics')
            logger.info("CsI Detector query detected - prioritizing csi_detector_physics")

        # 석회화 대조도 질문 시 calcification_contrast_physics 우선
        if is_calcification_query and 'calcification_contrast_physics' in matched_list:
            matched_list.remove('calcification_contrast_physics')
            matched_list.insert(0, 'calcification_contrast_physics')
            logger.info("Calcification query detected - prioritizing calcification_contrast_physics")

        # 확대 촬영 기하학 질문 시 magnification_geometry 우선
        if is_magnification_geo_query and 'magnification_geometry' in matched_list:
            matched_list.remove('magnification_geometry')
            matched_list.insert(0, 'magnification_geometry')
            logger.info("Magnification geometry query detected - prioritizing magnification_geometry")

        # 섬광체 빛 확산 질문 시 scintillator_light_spread 우선
        if is_light_spread_query and 'scintillator_light_spread' in matched_list:
            matched_list.remove('scintillator_light_spread')
            matched_list.insert(0, 'scintillator_light_spread')
            logger.info("Light spread query detected - prioritizing scintillator_light_spread")

        # 시스템 MTF 체인 질문 시 system_mtf_chain 우선
        if is_mtf_chain_query and 'system_mtf_chain' in matched_list:
            matched_list.remove('system_mtf_chain')
            matched_list.insert(0, 'system_mtf_chain')
            logger.info("MTF chain query detected - prioritizing system_mtf_chain")

        # ================================================================
        # 복합 문맥 우선순위 (Fine Linear → Amorphous 오분류 케이스)
        # ================================================================
        # 석회화 + CsI + 확대 + MTF + 두꺼운 유방 문맥이 동시에 있으면 관련 모듈 모두 포함
        # 두꺼운 유방(6cm)에서는 scatter_radiation이 중요
        is_thick_breast_query = any(kw in query_lower for kw in ['6cm', '두꺼운 유방', 'thick breast', '두께 6', '두께6'])

        # 1.8배 확대 생검 질문 감지
        is_magnification_biopsy_query = any(kw in query_lower for kw in
            ['1.8배', '1.8x', '확대 생검', '확대 스테레오', 'magnification biopsy',
             'stereotactic biopsy', '스테레오 생검', '확대촬영 생검'])

        # Phase 7.11: K-edge 키워드 감지 (DQE vs Contrast 혼동 방지)
        is_kedge_query = any(kw in query_lower for kw in ['k-edge', 'k edge', 'kedge', '케이엣지', '33kev', '33 kev'])

        if is_morphology_query or (is_calcification_query and is_csi_detector_query) or is_thick_breast_query or is_magnification_biopsy_query:
            # Phase 7.11: K-edge 질문 시 csi_detector_physics를 상위로 이동
            # (DQE vs Subject Contrast 혼동 방지를 위한 misconception 경고 포함)
            if is_kedge_query or is_csi_detector_query:
                priority_modules = [
                    'calcification_contrast_physics',
                    'csi_detector_physics',  # K-edge 질문 시 2순위 (DQE vs Δμ 구분 필수)
                    'system_mtf_chain',
                    'scatter_radiation',
                    'magnification_geometry',
                    'scintillator_light_spread',
                ]
            else:
                priority_modules = [
                    'calcification_contrast_physics',
                    'system_mtf_chain',
                    'scatter_radiation',  # 두꺼운 유방에서 SPR 증가로 CNR 저하
                    'magnification_geometry',  # 1.8배 확대 시 penumbra 영향
                    'scintillator_light_spread',
                    'csi_detector_physics'
                ]
            # 두꺼운 유방 질문 시 scatter_radiation 강제 추가 (매칭 안 됐어도)
            if is_thick_breast_query and 'scatter_radiation' not in matched_list:
                if 'scatter_radiation' in self._cache:
                    matched_list.append('scatter_radiation')
                    logger.info("Thick breast query - force-added scatter_radiation")

            # 1.8배 확대 생검 질문 시 magnification_geometry 강제 추가
            if is_magnification_biopsy_query and 'magnification_geometry' not in matched_list:
                if 'magnification_geometry' in self._cache:
                    matched_list.append('magnification_geometry')
                    logger.info("Magnification biopsy query - force-added magnification_geometry")

            # Phase 7.11: K-edge + CsI/검출기 질문 시 csi_detector_physics 강제 추가
            # (DQE vs Subject Contrast 혼동 방지를 위한 misconception 경고 필요)
            if (is_kedge_query or is_csi_detector_query) and 'csi_detector_physics' not in matched_list:
                if 'csi_detector_physics' in self._cache:
                    matched_list.append('csi_detector_physics')
                    logger.info("K-edge/CsI query - force-added csi_detector_physics for misconception prevention")

            for mod in reversed(priority_modules):
                if mod in matched_list:
                    matched_list.remove(mod)
                    matched_list.insert(0, mod)
            logger.info("Complex morphology/MTF/scatter query - multi-module priority applied")

        # ================================================================
        # Phase 1-Thermal: 열용량 질문 최우선 (모든 priority 체크 후 마지막에 적용)
        # ================================================================
        if is_thermal_capacity_query:
            if 'xray_tube_thermal_capacity' in matched_list:
                matched_list.remove('xray_tube_thermal_capacity')
                matched_list.insert(0, 'xray_tube_thermal_capacity')
            elif 'xray_tube_thermal_capacity' in self._cache:
                # 매칭 안 됐어도 열용량 질문이면 강제 추가
                matched_list.insert(0, 'xray_tube_thermal_capacity')
            logger.info("Thermal capacity query detected - prioritizing xray_tube_thermal_capacity (FINAL PRIORITY)")

        # ================================================================
        # Phase 7.21: CEM Detector Physics 최종 우선순위
        # CEM 맥락에서 Ghosting/Lag/QDE 질문 시 detector_physics 최우선 배치
        # DBT ghosting(재구성 아티팩트)과 CEM ghosting(검출기 민감도)을 구분
        # ================================================================
        if is_cem_ghosting_context:
            if 'detector_physics' in matched_list:
                matched_list.remove('detector_physics')
                matched_list.insert(0, 'detector_physics')
            elif 'detector_physics' in self._cache:
                # CEM 맥락에서는 detector_physics 강제 추가
                matched_list.insert(0, 'detector_physics')
            logger.info("Phase 7.21: CEM Ghosting/QDE query detected - prioritizing detector_physics (ABSOLUTE PRIORITY)")

        # 매칭된 모듈 반환
        results = []
        for file_id in matched_list[:max_modules]:
            if file_id in self._cache:
                results.append(self._cache[file_id])

        logger.info(f"Query matched {len(results)} knowledge modules: {matched_list[:max_modules]}")
        return results

    def format_for_context(
        self,
        knowledge_modules: List[Dict[str, Any]],
        include_tables: bool = True,
        include_formulas: bool = True,
        include_common_qa: bool = False,  # Phase 7.5: 기본값 False (Context Contamination 방지)
        query: str = "",  # Phase 7.18: 질문 기반 동적 섹션 배치
    ) -> str:
        """
        지식 모듈을 LLM 컨텍스트용 문자열로 포맷팅

        Args:
            knowledge_modules: 지식 모듈 리스트
            include_tables: 테이블 포함 여부
            include_formulas: 수식 포함 여부
            include_common_qa: 자주 묻는 질문 포함 여부 (기본 False - LLM 혼란 방지)
            query: 사용자 질문 (질문 기반 동적 섹션 배치용)

        Returns:
            포맷팅된 문자열
        """
        if not knowledge_modules:
            return ""

        # Phase 7.24: 질문 키워드 기반 모듈 순서 재조정
        # DICOM 관련 질문 시 contrast_enhanced_mammography를 맨 앞에 배치
        query_lower = query.lower()
        dicom_keywords = ["dicom", "태그", "tag", "주입 시간", "injection time",
                         "(0018,1042)", "(0018,1043)", "type 3", "metadata"]

        if any(kw in query_lower for kw in dicom_keywords):
            # DICOM 질문: contrast_enhanced_mammography 모듈을 맨 앞으로 이동
            cem_module = None
            other_modules = []
            for m in knowledge_modules:
                if m.get("id") == "contrast_enhanced_mammography":
                    cem_module = m
                else:
                    other_modules.append(m)
            if cem_module:
                knowledge_modules = [cem_module] + other_modules

        parts = []

        for module in knowledge_modules:
            module_id = module.get("id", "unknown")

            # Phase 7.19: core_physics 모듈 전용 포맷팅
            if module_id == "core_physics":
                parts.append(self._format_core_physics(module, query=query))
                continue

            # Phase 7.3: detector_physics 모듈 전용 포맷팅
            if module_id == "detector_physics":
                parts.append(self._format_detector_physics(module, query=query))
                continue

            # Phase 7.8: pcd_low_dose_snr 모듈 전용 포맷팅
            if module_id == "pcd_low_dose_snr":
                parts.append(self._format_pcd_low_dose_snr(module))
                continue

            # Phase 2: pcd_spectral_contrast 모듈 전용 포맷팅
            if module_id == "pcd_spectral_contrast":
                parts.append(self._format_pcd_spectral_contrast(module))
                continue

            # Phase 3: pcd_dqe_nps 모듈 전용 포맷팅
            if module_id == "pcd_dqe_nps":
                parts.append(self._format_pcd_dqe_nps(module))
                continue

            # Phase 4: pcd_mtf_resolution 모듈 전용 포맷팅
            if module_id == "pcd_mtf_resolution":
                parts.append(self._format_pcd_mtf_resolution(module))
                continue

            # Phase 4-B: biopsy_geometry_calibration 모듈 전용 포맷팅
            if module_id == "biopsy_geometry_calibration":
                parts.append(self._format_biopsy_geometry_calibration(module))
                continue

            # Phase 5: dbt_image_quality 모듈 전용 포맷팅
            if module_id == "dbt_image_quality":
                parts.append(self._format_dbt_image_quality(module))
                continue

            # Phase 7.24: contrast_enhanced_mammography 모듈 전용 포맷팅
            if module_id == "contrast_enhanced_mammography":
                parts.append(self._format_contrast_enhanced_mammography(module, query=query))
                continue
            source = module.get("source", {})
            parts.append(f"[표준 참조 자료] {source.get('authors', 'Unknown')} ({source.get('year', '')})")
            parts.append(f"출처: {source.get('title', '')}")
            parts.append(f"저널: {source.get('journal', '')} {source.get('volume', '')}:{source.get('pages', '')}")
            parts.append(f"DOI: {source.get('doi', '')}")
            parts.append("")

            # ================================================================
            # Phase 7.15: 자연스러운 경고 텍스트로 변환
            # JSON 구조 대신 자연스러운 한국어 문장으로 표현
            # ================================================================
            misconceptions = module.get("common_misconceptions", {})
            if misconceptions:
                parts.append("")
                parts.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                parts.append("⚠️ 주의: 다음 내용은 물리적으로 틀린 설명입니다")
                parts.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                warning_msg = misconceptions.get("WARNING", "")
                if warning_msg:
                    parts.append(warning_msg)
                    parts.append("")

                for key, m in misconceptions.items():
                    if key == "WARNING":
                        continue
                    if isinstance(m, dict):
                        wrong = m.get("WRONG", m.get("wrong", ""))
                        correct = m.get("CORRECT", m.get("correct", ""))
                        proof = m.get("physics_proof", "")
                        source_ref = m.get("source", "")

                        if wrong and correct:
                            # Phase 7.15: 더 자연스러운 문장 형태로
                            parts.append(f"[오류] {wrong}")
                            parts.append(f"[정답] {correct}")
                            if proof:
                                parts.append(f"  → 근거: {proof}")
                            if source_ref:
                                parts.append(f"  → 출처: {source_ref}")
                            parts.append("")

                parts.append("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
                parts.append("")

            # ================================================================
            # Phase 7.15: 핵심 물리 내용 (자연스러운 텍스트로)
            # ================================================================
            content = module.get("content", {})
            if content:
                parts.append("핵심 물리 내용:")
                self._format_content_recursive(content, parts, depth=0)
                parts.append("")

            # Phase 7.15: 정의 (자연스러운 문장으로)
            definitions = module.get("definitions", {})
            if definitions:
                parts.append("용어 정의:")
                for key, defn in definitions.items():
                    name = defn.get('name', key)
                    desc = defn.get('description', '')
                    parts.append(f"  {name}: {desc}")
                    if defn.get('formula_unicode'):
                        parts.append(f"    공식: {defn['formula_unicode']}")
                    if defn.get('note'):
                        parts.append(f"    참고: {defn['note']}")
                parts.append("")

            # Phase 7.15: 수식 (자연스러운 나열로)
            if include_formulas:
                formulas = module.get("formulas", {})
                if formulas:
                    parts.append("주요 공식:")
                    for key, formula in formulas.items():
                        if isinstance(formula, str):
                            label = self._KEY_TO_LABEL.get(key, key.replace("_", " "))
                            parts.append(f"  {label}: {formula}")
                        elif isinstance(formula, dict):
                            name = formula.get('name', key.replace("_", " "))
                            parts.append(f"  {name}")
                            parts.append(f"    {formula.get('formula_unicode', formula.get('formula_latex', ''))}")
                            variables = formula.get("variables", {})
                            if variables:
                                var_str = ", ".join([f"{v}={d}" for v, d in variables.items()])
                                parts.append(f"    변수: {var_str}")
                    parts.append("")

            # 테이블
            if include_tables:
                tables = module.get("tables", {})
                if tables:
                    parts.append("## 테이블")
                    for key, table in tables.items():
                        parts.append(f"### {table.get('description', key)}")

                        if key == "t_factor_by_thickness":
                            # t-factor 테이블 특별 포맷팅
                            parts.append("| 유방 두께 | t(0°) | t(10°) | t(20°) | t(30°) |")
                            parts.append("|-----------|-------|--------|--------|--------|")
                            for row in table.get("data", []):
                                parts.append(
                                    f"| {row['thickness_cm']} cm | "
                                    f"{row['t_0deg']:.3f} | {row['t_10deg']:.3f} | "
                                    f"{row['t_20deg']:.3f} | {row['t_30deg']:.3f} |"
                                )
                        elif key == "T_factor_typical":
                            parts.append("| Geometry | T값 |")
                            parts.append("|----------|-----|")
                            for row in table.get("data", []):
                                t_val = row.get("T_range", row.get("T_approx", ""))
                                parts.append(f"| {row['geometry']} | {t_val} |")

                        parts.append("")

            # 자주 묻는 질문
            if include_common_qa:
                common_qa = module.get("common_questions", [])
                if common_qa:
                    parts.append("## 자주 묻는 질문")
                    for qa in common_qa:
                        parts.append(f"Q: {qa['question']}")
                        parts.append(f"A: {qa['answer']}")
                        parts.append("")

            # Phase 7.15: 임상적 의의 (자연스러운 문장으로)
            clinical = module.get("clinical_relevance", {})
            if clinical:
                parts.append("임상적 의의:")
                if isinstance(clinical, dict):
                    for key, value in clinical.items():
                        label = self._KEY_TO_LABEL.get(key, key.replace("_", " "))
                        parts.append(f"  • {label}: {value}")
                elif isinstance(clinical, str):
                    parts.append(f"  {clinical}")
                parts.append("")

        return "\n".join(parts)

    # Phase 7.15: JSON 키 → 자연스러운 한국어 레이블 매핑
    _KEY_TO_LABEL = {
        # 일반 키
        "cascade_principle": "연쇄 원리",
        "fundamental_law": "기본 법칙",
        "implication": "의미",
        "worst_component": "제한 요소",
        "mtf_components": "MTF 구성 요소",
        "focal_spot_mtf": "초점 MTF",
        "detector_mtf": "검출기 MTF",
        "motion_mtf": "움직임 MTF",
        "scatter_mtf": "산란 MTF",
        "source": "원인",
        "formula": "공식",
        "magnification_impact": "확대 영향",
        "energy_dependence": "에너지 의존성",
        "typical_values": "일반 값",
        "combined_analysis": "종합 분석",
        "resolution_requirements": "해상도 요건",
        "optimization_strategies": "최적화 전략",
        "misclassification_risk": "오분류 위험",
        "condition": "조건",
        "result": "결과",
        "scenario": "시나리오",
        "consequence": "결과",
        # 검출기 관련
        "physical_properties": "물리적 특성",
        "absorption_physics": "흡수 물리",
        "light_spread_mtf": "빛 확산과 MTF",
        "mechanism": "메커니즘",
        "impact": "영향",
        "high_energy_degradation": "고에너지 열화",
        "optimal_energy": "최적 에너지",
        "k_edges": "K-edge",
        # 기타
        "width": "폭",
        "required_frequency": "필요 주파수",
        "mtf_threshold": "MTF 임계값",
        "characteristics": "특성",
        "dominant_frequency": "주요 주파수",
    }

    def _format_content_recursive(self, obj: Any, parts: List[str], depth: int = 0, max_depth: int = 3):
        """
        Phase 7.15: content 객체를 자연스러운 텍스트로 포맷팅
        JSON 키 이름을 한국어 레이블로 변환하여 LLM이 JSON 패턴을 모방하지 않도록 함

        Args:
            obj: 포맷팅할 객체 (dict, list, or scalar)
            parts: 출력 라인 리스트
            depth: 현재 재귀 깊이
            max_depth: 최대 재귀 깊이
        """
        indent = "  " * depth

        if depth > max_depth:
            if isinstance(obj, (dict, list)):
                parts.append(f"{indent}(추가 내용 생략)")
            return

        if isinstance(obj, dict):
            for key, value in obj.items():
                if key in ["source_ref", "sources", "note"]:  # 메타데이터 스킵
                    continue

                # Phase 7.15: JSON 키를 한국어 레이블로 변환
                label = self._KEY_TO_LABEL.get(key, key.replace("_", " ").title())

                if isinstance(value, dict):
                    parts.append(f"{indent}▶ {label}")
                    self._format_content_recursive(value, parts, depth + 1, max_depth)
                elif isinstance(value, list):
                    parts.append(f"{indent}▶ {label}:")
                    for item in value:
                        if isinstance(item, str):
                            parts.append(f"{indent}  • {item}")
                        elif isinstance(item, dict):
                            self._format_content_recursive(item, parts, depth + 1, max_depth)
                else:
                    # scalar value - 자연스러운 문장 형태로
                    parts.append(f"{indent}• {label}: {value}")
        elif isinstance(obj, list):
            for item in obj:
                if isinstance(item, str):
                    parts.append(f"{indent}• {item}")
                elif isinstance(item, dict):
                    self._format_content_recursive(item, parts, depth, max_depth)
        else:
            parts.append(f"{indent}{obj}")

    def _format_detector_physics(self, module: Dict[str, Any], query: str = "") -> str:
        """
        Phase 7.3: detector_physics 모듈 전용 포맷팅
        Phase 7.18: query 기반 동적 섹션 배치

        검출기 물리학 지식을 LLM이 이해하기 쉬운 형태로 변환
        질문 키워드에 따라 관련 섹션을 앞에 배치하여 truncation 방지
        """
        parts = []
        parts.append("=" * 50)
        parts.append("[검출기 물리학 표준 참조 자료 - Phase 7.3]")
        parts.append("=" * 50)

        # Phase 7.18: 질문 키워드 기반 섹션 우선순위 결정
        query_lower = query.lower()

        # 섹션 포맷팅 함수들을 정의
        def format_detector_types():
            section_parts = []
            detector_types = module.get("detector_types", {})
            if detector_types:
                section_parts.append("\n## 검출기 유형 비교")

                # Direct conversion (a-Se)
                direct = detector_types.get("direct_conversion", {})
                a_se = direct.get("a_Se", {})
                if a_se:
                    section_parts.append("\n### 직접변환 방식 (a-Se)")
                    section_parts.append(f"- 원자번호: {a_se.get('atomic_number', 'N/A')}")
                    section_parts.append(f"- K-edge: {a_se.get('k_edge_keV', 'N/A')} keV")
                    ehp = a_se.get("electron_hole_pair_energy_eV", {})
                    if ehp:
                        section_parts.append(f"- 전자-홀 쌍 생성 에너지 (W): {ehp.get('commonly_cited', 50)} eV (일반), {ehp.get('at_mammographic_energy_17keV_10Vum', 64)} eV (맘모그래피 에너지)")
                        if ehp.get('note'):
                            section_parts.append(f"  ⚠️ {ehp.get('note')}")
                    dqe = a_se.get("typical_DQE", {})
                    mtf = a_se.get("typical_MTF", {})
                    section_parts.append(f"- DQE(0): {dqe.get('at_0_lp_mm', 'N/A')}")
                    section_parts.append(f"- DQE(5 lp/mm): {dqe.get('at_5_lp_mm', 'N/A')}")
                    section_parts.append(f"- MTF(5 lp/mm): {mtf.get('at_5_lp_mm', 'N/A')}")
                    section_parts.append(f"- 장점: {', '.join(a_se.get('advantages', [])[:3])}")
                    section_parts.append(f"- 단점: {', '.join(a_se.get('disadvantages', [])[:2])}")

                # Indirect conversion (CsI)
                indirect = detector_types.get("indirect_conversion", {})
                csi = indirect.get("CsI_aSi", {})
                if csi:
                    section_parts.append("\n### 간접변환 방식 (CsI/a-Si)")
                    section_parts.append(f"- 섬광체: {csi.get('scintillator', 'N/A')}")
                    section_parts.append(f"- 구조: {csi.get('structure', 'N/A')}")
                    dqe = csi.get("typical_DQE", {})
                    mtf = csi.get("typical_MTF", {})
                    section_parts.append(f"- DQE(0): {dqe.get('at_0_lp_mm', 'N/A')}")
                    section_parts.append(f"- DQE(5 lp/mm): {dqe.get('at_5_lp_mm', 'N/A')}")
                    section_parts.append(f"- MTF(5 lp/mm): {mtf.get('at_5_lp_mm', 'N/A')}")
                    section_parts.append(f"- 장점: {', '.join(csi.get('advantages', [])[:3])}")
                    section_parts.append(f"- 단점: {', '.join(csi.get('disadvantages', [])[:2])}")

                # Comparison table
                comp = detector_types.get("comparison_table", {})
                if comp:
                    section_parts.append("\n### 비교 요약")
                    params = comp.get("parameter", [])
                    a_se_vals = comp.get("a_Se_direct", [])
                    csi_vals = comp.get("CsI_indirect", [])
                    implications = comp.get("clinical_implication", [])
                    section_parts.append("| 항목 | a-Se (직접) | CsI (간접) | 임상적 의미 |")
                    section_parts.append("|------|-------------|------------|-------------|")
                    for i, param in enumerate(params):
                        a_val = a_se_vals[i] if i < len(a_se_vals) else ""
                        c_val = csi_vals[i] if i < len(csi_vals) else ""
                        impl = implications[i] if i < len(implications) else ""
                        section_parts.append(f"| {param} | {a_val} | {c_val} | {impl} |")
            return section_parts

        def format_dqe_physics():
            section_parts = []
            dqe_physics = module.get("DQE_physics", {})
            if dqe_physics:
                section_parts.append("\n## DQE 물리학")
                section_parts.append(f"- 정의: {dqe_physics.get('definition', '')}")
                section_parts.append(f"- 공식: {dqe_physics.get('formula', '')}")
                section_parts.append(f"- 확장 공식: {dqe_physics.get('expanded_formula', '')}")
            return section_parts

        def format_dose_optimization():
            section_parts = []
            dose_opt = module.get("dose_optimization", {})
            if dose_opt:
                section_parts.append("\n## 선량 최적화 파라미터")
                params = dose_opt.get("parameters", {})

                kvp = params.get("kVp", {})
                if kvp:
                    section_parts.append(f"\n### kVp (관전압)")
                    section_parts.append(f"- 범위: {kvp.get('typical_range_mammography', 'N/A')}")
                    section_parts.append(f"- 선량 관계: {kvp.get('dose_relationship', 'N/A')}")
                    section_parts.append(f"- 최적화 규칙: {kvp.get('optimization_rule', 'N/A')}")

                mas = params.get("mAs", {})
                if mas:
                    section_parts.append(f"\n### mAs (관전류-시간적)")
                    section_parts.append(f"- 선량 관계: {mas.get('dose_relationship', 'N/A')}")
                    section_parts.append(f"- 노이즈 효과: {mas.get('effect_on_noise', 'N/A')}")
                    section_parts.append(f"- 최적화 규칙: {mas.get('optimization_rule', 'N/A')}")

                tf = params.get("target_filter", {})
                if tf:
                    section_parts.append("\n### Target/Filter 조합")
                    combos = tf.get("combinations", {})
                    for name, data in combos.items():
                        section_parts.append(f"- {name}: {data.get('use', '')} (HVL: {data.get('HVL_mmAl', '')})")
                    reduction = tf.get("dose_reduction_by_filter_change", {})
                    if reduction:
                        section_parts.append("\n선량 감소 효과:")
                        for change, effect in reduction.items():
                            section_parts.append(f"  - {change}: {effect}")

                cnr = dose_opt.get("CNR_maintenance_formulas", {})
                if cnr:
                    section_parts.append("\n### CNR 유지 공식")
                    section_parts.append(f"- Rose Criterion: {cnr.get('Rose_Criterion', 'CNR >= 5')}")
                    section_parts.append(f"- CNR vs Dose: {cnr.get('CNR_vs_dose', '')}")
                    maintain = cnr.get("dose_reduction_with_CNR_maintenance", {})
                    if maintain:
                        section_parts.append(f"- DQE 증가 공식: {maintain.get('required_DQE_increase', '')}")
                        section_parts.append(f"- 예시 (10%): {maintain.get('example_10_percent', '')}")
                        section_parts.append(f"- 예시 (20%): {maintain.get('example_20_percent', '')}")
            return section_parts

        def format_advanced_metrics():
            section_parts = []
            advanced = module.get("advanced_metrics", {})
            if advanced:
                section_parts.append("\n## 고급 검출 메트릭")
                for metric_name, metric_data in advanced.items():
                    section_parts.append(f"\n### {metric_name}")
                    section_parts.append(f"- 정의: {metric_data.get('definition', '')}")
                    if metric_data.get('formula'):
                        section_parts.append(f"- 공식: {metric_data.get('formula', '')}")
                    if metric_data.get('clinical_threshold'):
                        section_parts.append(f"- 임상 기준: {metric_data.get('clinical_threshold', '')}")
            return section_parts

        def format_clinical_examples():
            section_parts = []
            examples = module.get("clinical_application_examples", {})
            if examples:
                section_parts.append("\n## 계산 예시")
                for ex_name, ex_data in examples.items():
                    section_parts.append(f"\n### {ex_data.get('question', ex_name)}")
                    calc = ex_data.get("calculation", {})
                    for step, value in calc.items():
                        section_parts.append(f"  - {step}: {value}")
                    section_parts.append(f"  → 결론: {ex_data.get('conclusion', '')}")
            return section_parts

        def format_ghosting_and_lag():
            section_parts = []
            ghosting = module.get("ghosting_and_lag", {})
            if ghosting:
                section_parts.append("\n" + "=" * 50)
                section_parts.append("## ⚠️ Ghosting 및 Lag (검증된 수치)")
                section_parts.append("=" * 50)

                defs = ghosting.get("definitions", {})
                lag_def = defs.get("lag", {})
                ghost_def = defs.get("ghosting", {})
                if lag_def:
                    section_parts.append(f"\n### Lag 정의")
                    section_parts.append(f"- {lag_def.get('definition', '')}")
                    clinical = lag_def.get("clinical_values", {})
                    if clinical:
                        section_parts.append(f"- **검증된 수치: 최대 {clinical.get('typical_range_percent', 'N/A')}%** (출처: {clinical.get('source', '')})")
                if ghost_def:
                    section_parts.append(f"\n### Ghosting 정의")
                    section_parts.append(f"- {ghost_def.get('definition', '')}")
                    clinical = ghost_def.get("clinical_values", {})
                    if clinical:
                        section_parts.append(f"- **검증된 수치: 최대 {clinical.get('maximum_observed_percent', 'N/A')}%** (출처: {clinical.get('source', '')})")

                # Phase 7.18: Lag vs Ghosting 혼동 방지 경고 테이블
                section_parts.append("\n### ⚠️⚠️⚠️ 중요: Lag vs Ghosting 구분 (100배 차이!) ⚠️⚠️⚠️")
                section_parts.append("| 현상 | 정의 | **검증 수치** | 혼동 금지 |")
                section_parts.append("|------|------|---------------|-----------|")
                section_parts.append("| **Lag** | 신호 잔류 (temporal) | **0.15%** | 작은 값 |")
                section_parts.append("| **Ghosting** | 민감도 감소 (sensitivity) | **15%** | 큰 값 |")
                section_parts.append("| - | **차이** | **100배** | **절대 혼동 금지** |")
                section_parts.append("")
                section_parts.append("**⚠️ Ghosting = 15%, Lag = 0.15%. 둘을 절대 혼동하지 말 것!**")

                mech = ghosting.get("mechanism", {})
                if mech:
                    section_parts.append(f"\n### 메커니즘")
                    section_parts.append(f"- 주요 원인: {mech.get('primary_cause', '')}")
                    process = mech.get("process", [])
                    for step in process:
                        section_parts.append(f"  {step}")

                quant = ghosting.get("quantitative_data", {})
                if quant:
                    section_parts.append(f"\n### 정량적 데이터 (검증됨)")
                    transport = quant.get("charge_transport_degradation", {})
                    if transport:
                        section_parts.append(f"- Hole transport 감소: **{transport.get('hole_transport_reduction_percent', 'N/A')}%**")
                        section_parts.append(f"- Electron transport 감소: **{transport.get('electron_transport_reduction_percent', 'N/A')}%**")
                        section_parts.append(f"  (조건: {transport.get('condition', '')}, 출처: {transport.get('source', '')})")
                    lag_bias = quant.get("lag_vs_bias", {})
                    if lag_bias:
                        section_parts.append(f"- Positive bias lag: {lag_bias.get('positive_bias_lag_percent', 'N/A')}%")
                        section_parts.append(f"- Negative bias lag: {lag_bias.get('negative_bias_lag_percent', 'N/A')}%")

                factors = ghosting.get("affecting_factors", {})
                if factors:
                    section_parts.append(f"\n### 영향 인자")
                    exp_dep = factors.get("exposure_dependence", {})
                    if exp_dep:
                        section_parts.append(f"- 노출 의존성: {exp_dep.get('relationship', '')}")
                    field_dep = factors.get("electric_field_dependence", {})
                    if field_dep:
                        section_parts.append(f"- 전기장 의존성: {field_dep.get('relationship', '')}")
                    recovery = factors.get("recovery", {})
                    if recovery:
                        section_parts.append(f"- 회복 메커니즘: {recovery.get('mechanism', '')}")

                # Phase 7.22: 누적 노출 노후화 섹션
                aging = ghosting.get("cumulative_exposure_aging", {})
                if aging:
                    section_parts.append("\n### ⚠️ 누적 노출 노후화 (1년+ 사용 장비)")
                    phenomenon = aging.get("phenomenon", {})
                    if phenomenon:
                        section_parts.append(f"- **현상**: {phenomenon.get('description', '')}")
                        section_parts.append(f"- **메커니즘**: {phenomenon.get('mechanism', '')}")

                    time_variant = aging.get("time_variant_gain_map", {})
                    if time_variant:
                        section_parts.append(f"\n**Time-Variant Gain Map 문제:**")
                        section_parts.append(f"- 문제: {time_variant.get('problem', '')}")
                        reason = time_variant.get("reason", {})
                        if reason:
                            section_parts.append(f"- QDE 불일치: {reason.get('qde_mismatch', '')}")
                            section_parts.append(f"- 트랩 활성화: {reason.get('aging_trap_activation', '')}")
                            section_parts.append(f"- **핵심**: {reason.get('dynamic_behavior', '')}")

                # Phase 7.22: 적응적 CEM 보정 섹션 (correction_algorithms 내)
                corr = ghosting.get("correction_algorithms", {})
                adaptive = corr.get("practical_realtime_correction", {}).get("adaptive_cem_correction", {}) if corr else {}
                if adaptive:
                    section_parts.append("\n### 💡 소프트웨어 적응적 보정 (권장)")

                    virtual_gain = adaptive.get("virtual_de_gain_with_aging_coefficient", {})
                    if virtual_gain:
                        section_parts.append(f"\n**1. Virtual DE-Gain (노후화 계수 적용)**")
                        section_parts.append(f"- 공식: `{virtual_gain.get('equation', '')}`")
                        params = virtual_gain.get("parameters", {})
                        if params:
                            section_parts.append(f"  - QDE_ratio = {params.get('QDE_ratio', '56/97 = 0.577')}")
                            section_parts.append(f"  - β = 노후화 계수 (장비별 캘리브레이션)")

                    exposure_hist = adaptive.get("exposure_history_ghosting_decay", {})
                    if exposure_hist:
                        section_parts.append(f"\n**2. 노출 이력 기반 Ghosting 보정**")
                        section_parts.append(f"- 보정 공식: `{exposure_hist.get('equation', '')}`")
                        g_eff = exposure_hist.get("ghosting_effectiveness", {})
                        if g_eff:
                            fast = g_eff.get("fast_component", {})
                            slow = g_eff.get("slow_component", {})
                            if fast and slow:
                                section_parts.append(f"  - Fast (홀 트랩): α₁={fast.get('alpha_1', 0.10)}, τ₁={fast.get('tau_1_minutes', 2)}분")
                                section_parts.append(f"  - Slow (전자 트랩): α₂={slow.get('alpha_2', 0.05)}, τ₂={slow.get('tau_2_minutes', 60)}분")

                    restoration = adaptive.get("de_sensitivity_restoration", {})
                    if restoration:
                        section_parts.append(f"\n**3. 민감도 역보정 (올바른 방법)**")
                        section_parts.append(f"- ❌ 잘못된 방법: {restoration.get('wrong_approach', '')}")
                        section_parts.append(f"- ✅ 올바른 방법: {restoration.get('correct_approach', '')}")
                        section_parts.append(f"- 핵심: 15% 감소 복원 → **17.6% 증폭** 필요 (비선형)")

                # Phase 7.22: HE 캘리브레이션 vs 소프트웨어 보정 비교 (핵심 권장사항)
                comparison = adaptive.get("comparison_hw_vs_sw_correction", {})
                if comparison:
                    warning = comparison.get("WARNING_HE_CALIBRATION", "")
                    if warning:
                        section_parts.append(f"\n### ⚠️⚠️⚠️ {warning} ⚠️⚠️⚠️")

                    # HE 캘리브레이션 문제점
                    hw_approaches = comparison.get("hardware_approaches", {})
                    he_cal = hw_approaches.get("he_calibration", {})
                    if he_cal:
                        section_parts.append(f"\n**HE 캘리브레이션 문제점 (논문 검증):**")
                        tube_evidence = he_cal.get("tube_erosion_evidence", {})
                        if tube_evidence:
                            section_parts.append(f"- ⚠️ **튜브 손상**: {tube_evidence.get('thermal_cycling_damage', '')}")
                            section_parts.append(f"- 열전도 {tube_evidence.get('thermal_conductivity_loss_percent', 'N/A')}% 감소")
                            section_parts.append(f"- 출처: {tube_evidence.get('source', '')}")
                        detector_evidence = he_cal.get("detector_aging_evidence", {})
                        if detector_evidence:
                            section_parts.append(f"- ⚠️ **디텍터 노후화**: {detector_evidence.get('mechanism', '')}")
                            section_parts.append(f"- 8R 누적 후 민감도 {detector_evidence.get('sensitivity_after_8R_percent', 'N/A')}%")

                    # 소프트웨어 보정 장점
                    sw_approaches = comparison.get("software_approaches", {})
                    if sw_approaches:
                        section_parts.append(f"\n**✅ 소프트웨어 보정 권장 (논문 검증):**")
                        per_patient = sw_approaches.get("per_patient_offset_gain", {})
                        if per_patient:
                            section_parts.append(f"- 매 환자 후 offset/gain 보정: **{per_patient.get('effectiveness_percent', '>80')}% 효과**")
                            section_parts.append(f"- 권장 근거: \"{per_patient.get('recommendation_quote', '')}\"")
                        forward = sw_approaches.get("forward_bias_software_equivalent", {})
                        if forward:
                            section_parts.append(f"- Forward bias 원리: lag ghost **{forward.get('hardware_results', '70-88% 감소')}**")

                    # 권장 전략
                    recommended = comparison.get("recommended_strategy", {})
                    if recommended:
                        section_parts.append(f"\n**📌 권장 전략:**")
                        section_parts.append(f"- 1순위: {recommended.get('priority_1', '')}")
                        section_parts.append(f"- 2순위: {recommended.get('priority_2', '')}")
                        section_parts.append(f"- ⚠️ 회피: {recommended.get('avoid', '')}")
                        section_parts.append(f"- 근거: {recommended.get('rationale', '')}")

            return section_parts

        def format_dual_energy_response():
            section_parts = []
            dual_energy = module.get("dual_energy_response", {})
            if dual_energy:
                section_parts.append("\n" + "=" * 50)
                section_parts.append("## ⚠️ Dual-Energy 에너지 응답 (CEM 핵심)")
                section_parts.append("=" * 50)

                qde_data = dual_energy.get("energy_dependent_QDE", {})
                if qde_data:
                    # Phase 7.18: QDE/QE 용어 정의 추가 (LLM이 이해할 수 있도록)
                    section_parts.append(f"\n### 용어 정의: QDE = QE = 양자검출효율")
                    section_parts.append("- **QDE** (Quantum Detection Efficiency) = **QE** (Quantum Efficiency) = **양자검출효율** = **X선 흡수 효율**")
                    section_parts.append("- 모두 같은 의미: 입사 X선 중 검출기에서 흡수되어 신호로 변환되는 비율")
                    section_parts.append("- ⚠️ 이론 공식 QE=1-e^(-μt)는 근사치. 아래 **실측값**을 사용할 것!")
                    section_parts.append(f"- 설명: {qde_data.get('description', '')}")

                    data = qde_data.get("data", {})
                    le_data = data.get("LE_28kVp_W_Rh", {})
                    he_data = data.get("HE_49kVp_W_Ti", {})
                    if le_data and he_data:
                        le_qde = le_data.get('QDE_percent', 97)
                        he_qde = he_data.get('QDE_percent', 56)
                        qde_diff = le_qde - he_qde

                        section_parts.append(f"\n### ⚠️⚠️⚠️ CEM 양자검출효율(QE) 실측값 (반드시 인용!) ⚠️⚠️⚠️")
                        section_parts.append("| 에너지 | kVp | Target/Filter | **QE (양자검출효율)** | a-Se 두께 |")
                        section_parts.append("|--------|-----|---------------|----------------------|-----------|")
                        section_parts.append(f"| **LE** | 28 kVp | W/Rh | **{le_qde}%** | 200 μm |")
                        section_parts.append(f"| **HE** | 49 kVp | W/Ti | **{he_qde}%** | 200 μm |")
                        section_parts.append(f"| **차이** | - | - | **{qde_diff}% 포인트** | - |")
                        section_parts.append("")
                        section_parts.append(f"**⚠️ 실측 양자검출효율: QE(LE) = {le_qde}%, QE(HE) = {he_qde}%**")
                        section_parts.append(f"**⚠️ HE에서 QE가 {qde_diff}%p 낮음 → 이것이 조영제 농도 저감의 핵심 원인!**")
                        section_parts.append("**⚠️ 이론 공식(QE=1-e^(-μt))으로 계산하지 말고 위 실측값을 직접 인용할 것!**")

                    iodine = qde_data.get("iodine_k_edge", {})
                    if iodine:
                        section_parts.append(f"\n요오드 K-edge ({iodine.get('energy_keV', 'N/A')} keV):")
                        section_parts.append(f"- 200 μm a-Se: η = {iodine.get('eta_200um_percent', 'N/A')}%")
                        section_parts.append(f"- 500 μm a-Se: η = {iodine.get('eta_500um_percent', 'N/A')}%")

                dqe_imp = dual_energy.get("DQE_improvement", {})
                if dqe_imp:
                    section_parts.append(f"\n### DQE 개선")
                    section_parts.append(f"- {dqe_imp.get('description', '')}")
                    proto = dqe_imp.get("prototype_vs_conventional", {})
                    if proto:
                        section_parts.append(f"- 프로토타입 개선: **{proto.get('DQE_improvement_percent', 'N/A')}%** ({proto.get('condition', '')})")

                calib = dual_energy.get("CEM_calibration_implications", {})
                if calib:
                    section_parts.append(f"\n### CEM 캘리브레이션 문제")
                    section_parts.append(f"- 문제: {calib.get('problem', '')}")
                    section_parts.append(f"- **QDE 차이: {calib.get('QDE_difference', '')}**")
                    section_parts.append(f"- 결과: {calib.get('consequence', '')}")
            return section_parts

        # Phase 7.18: 질문 키워드에 따른 동적 섹션 순서 결정
        # CEM/Ghosting/Dual-Energy 관련 키워드
        cem_keywords = ["cem", "조영", "contrast", "ghosting", "잔상", "ghost", "lag", "dual", "이중에너지",
                       "le", "he", "qde", "quantum detection", "iodine", "요오드"]
        # DQE/해상도 관련 키워드
        dqe_keywords = ["dqe", "detective quantum", "mtf", "해상도", "resolution", "snr", "노이즈"]
        # 선량 관련 키워드
        dose_keywords = ["dose", "선량", "kvp", "mas", "radiation", "exposure", "filter", "target"]

        # 키워드 매칭으로 우선 섹션 결정
        is_cem_related = any(kw in query_lower for kw in cem_keywords)
        is_dqe_related = any(kw in query_lower for kw in dqe_keywords)
        is_dose_related = any(kw in query_lower for kw in dose_keywords)

        # 동적 섹션 순서 결정
        if is_cem_related:
            # CEM 관련: Ghosting/Dual-Energy를 맨 앞에 배치
            section_order = [
                format_ghosting_and_lag,
                format_dual_energy_response,
                format_detector_types,
                format_dqe_physics,
                format_dose_optimization,
                format_advanced_metrics,
                format_clinical_examples,
            ]
        elif is_dqe_related:
            # DQE 관련: DQE 물리학을 앞에 배치
            section_order = [
                format_dqe_physics,
                format_detector_types,
                format_dual_energy_response,
                format_advanced_metrics,
                format_dose_optimization,
                format_ghosting_and_lag,
                format_clinical_examples,
            ]
        elif is_dose_related:
            # 선량 관련: 선량 최적화를 앞에 배치
            section_order = [
                format_dose_optimization,
                format_detector_types,
                format_dqe_physics,
                format_advanced_metrics,
                format_clinical_examples,
                format_ghosting_and_lag,
                format_dual_energy_response,
            ]
        else:
            # 기본 순서
            section_order = [
                format_detector_types,
                format_dqe_physics,
                format_dose_optimization,
                format_advanced_metrics,
                format_clinical_examples,
                format_ghosting_and_lag,
                format_dual_energy_response,
            ]

        # 결정된 순서대로 섹션 포맷팅
        for format_func in section_order:
            parts.extend(format_func())

        return "\n".join(parts)

    def _format_pcd_low_dose_snr(self, module: Dict[str, Any]) -> str:
        """
        Phase 7.8: pcd_low_dose_snr 모듈 전용 포맷팅

        PCD vs EID 저선량 SNR 비교 근거를 논문 인용과 함께 제공
        """
        parts = []
        parts.append("=" * 50)
        parts.append("[PCD 저선량 SNR 증거 자료 - Phase 7.8 Layer 3]")
        parts.append("=" * 50)

        # 1. Constitutional Axioms
        core = module.get("core_physics", {})
        axioms = core.get("constitutional_axioms", {})
        if axioms:
            parts.append("\n## 물리 공리 (Constitutional Laws)")
            for key, axiom in axioms.items():
                parts.append(f"- **{axiom['law']}**: {axiom['description']}")

        # 2. SNR 공식
        snr_formulas = core.get("snr_formula", {})
        if snr_formulas:
            parts.append("\n## SNR 핵심 공식")
            eid = snr_formulas.get("eid_with_electronic_noise", {})
            if eid:
                parts.append(f"- EID: {eid.get('description', '')}")
                parts.append(f"  예시: {eid.get('example', '')}")
            pcd = snr_formulas.get("pcd_noise_eliminated", {})
            if pcd:
                parts.append(f"- PCD: {pcd.get('description', '')}")
                parts.append(f"  예시: {pcd.get('example', '')}")
            recovery = snr_formulas.get("pcd_recovery", {})
            if recovery:
                parts.append(f"- 회복률: {recovery.get('description', '')}")
                parts.append(f"  예시: {recovery.get('example', '')}")

        # 3. Evidence Database (핵심)
        evidence = module.get("evidence_database", {})
        if evidence:
            parts.append("\n## 논문 근거 (Evidence Database)")
            parts.append(f"검증 기준: {evidence.get('content_verification_note', '')}")

            # 전자노이즈 제거
            elec = evidence.get("electronic_noise_elimination", {})
            if elec:
                parts.append(f"\n### 전자노이즈 제거 기전")
                parts.append(f"기전: {elec.get('mechanism', '')}")
                for ev in elec.get("key_evidence", []):
                    level = ev.get('verification_level', 'unknown')
                    parts.append(f"- [{level}] [{ev['source']}]: \"{ev['finding']}\"")
                    if ev.get('abstract_quote'):
                        parts.append(f"  > 원문: \"{ev['abstract_quote']}\"")

            # 선량 감소 근거
            dose = evidence.get("dose_reduction_evidence", {})
            if dose:
                parts.append(f"\n### 선량 감소 근거")
                parts.append(f"요약: {dose.get('summary', '')}")
                for ev in dose.get("key_evidence", []):
                    level = ev.get('verification_level', 'unknown')
                    parts.append(f"- [{level}] [{ev['source']}]: \"{ev['finding']}\"")

            # SNR 스케일링 물리
            snr_scale = evidence.get("snr_scaling_physics", {})
            if snr_scale:
                parts.append(f"\n### SNR 스케일링 물리")
                parts.append(f"요약: {snr_scale.get('summary', '')}")
                for ev in snr_scale.get("key_evidence", []):
                    level = ev.get('verification_level', 'unknown')
                    parts.append(f"- [{level}] [{ev['source']}]: \"{ev['finding']}\"")
                    if ev.get('abstract_quote'):
                        parts.append(f"  > 원문: \"{ev['abstract_quote']}\"")
                    if ev.get('implication'):
                        parts.append(f"  의미: {ev['implication']}")

            # 스펙트럼 유방영상
            spectral = evidence.get("spectral_mammography", {})
            if spectral:
                parts.append(f"\n### 스펙트럼 유방영상")
                parts.append(f"요약: {spectral.get('summary', '')}")
                for ev in spectral.get("key_evidence", []):
                    level = ev.get('verification_level', 'unknown')
                    parts.append(f"- [{level}] [{ev['source']}]: \"{ev['finding']}\"")
                    if ev.get('abstract_quote'):
                        parts.append(f"  > 원문: \"{ev['abstract_quote']}\"")

        # 4. 임상적 의의
        clinical = module.get("clinical_significance", {})
        if clinical:
            parts.append(f"\n## 임상적 의의")
            parts.append(f"- 경로: {clinical.get('dose_reduction_pathway', '')}")
            parts.append(f"- 실무: {clinical.get('practical_implication', '')}")
            parts.append(f"- 대상: {clinical.get('target_population', '')}")

        # 5. 인용 템플릿
        template = module.get("response_template", {})
        if template:
            parts.append(f"\n## 응답 시 인용 형식")
            parts.append(f"형식: {template.get('citation_format', '')}")
            parts.append(f"예시: {template.get('example_usage', '')}")
            priority = template.get("citation_priority", {})
            if priority:
                parts.append("인용 우선순위:")
                parts.append(f"  Tier 1 (abstract 직접 인용): {priority.get('tier_1_abstract_quote', [])}")
                parts.append(f"  Tier 2 (abstract 확인): {priority.get('tier_2_abstract_supported', [])}")
                parts.append(f"  Tier 3 (full text 필요): {priority.get('tier_3_full_text_needed', [])}")
            mandatory = template.get("mandatory_citations", [])
            if mandatory:
                parts.append("필수 인용:")
                for cite in mandatory:
                    parts.append(f"  - {cite}")
            if template.get("warning"):
                parts.append(f"\n⚠️ {template['warning']}")

        return "\n".join(parts)

    def _format_pcd_spectral_contrast(self, module: Dict[str, Any]) -> str:
        """
        Phase 2: pcd_spectral_contrast 모듈 전용 포맷팅

        PCD 에너지 가중, K-edge 영상, 물질 분해의 대조도 향상 근거 제공
        """
        parts = []
        parts.append("=" * 50)
        parts.append("[PCD Spectral Contrast 증거 자료 - Phase 2 Layer 3]")
        parts.append("=" * 50)

        # 1. Phase 2 Axioms
        core = module.get("core_physics", {})
        axioms = core.get("phase2_axioms", {})
        if axioms:
            parts.append("\n## Phase 2 물리 공리")
            for key, axiom in axioms.items():
                parts.append(f"- **{axiom['law']}**: {axiom['description']}")

        # 2. Energy Weighting Gain
        ewg = core.get("energy_weighting_gain", {})
        if ewg:
            parts.append("\n## 에너지 가중 이득 (Solver 검증 결과)")
            solver_result = ewg.get("solver_result", {})
            for label, val in solver_result.items():
                parts.append(f"  - {label}: {val}")
            parts.append(f"  보장: {ewg.get('cauchy_schwarz_guarantee', '')}")

        # 3. K-edge 정보
        kedge = core.get("kedge_imaging", {})
        if kedge:
            parts.append(f"\n## K-edge 영상")
            parts.append(f"  - Iodine K-edge: {kedge.get('iodine_kedge_keV', '')} keV")
            parts.append(f"  - 최적 문턱치: {kedge.get('optimal_threshold', '')}")
            parts.append(f"  - 기전: {kedge.get('mechanism', '')}")

        # 4. Evidence Database
        evidence = module.get("evidence_database", {})
        if evidence:
            parts.append("\n## 논문 근거 (Evidence Database)")
            parts.append(f"검증 기준: {evidence.get('content_verification_note', '')}")

            for section_key in ['energy_weighting', 'spectral_detectability', 'kedge_quantification', 'material_decomposition']:
                section = evidence.get(section_key, {})
                if section:
                    parts.append(f"\n### {section.get('summary', section_key)}")
                    for ev in section.get("key_evidence", []):
                        level = ev.get('verification_level', 'unknown')
                        parts.append(f"- [{level}] [{ev['source']}]:")
                        parts.append(f"  \"{ev['finding']}\"")
                        if ev.get('abstract_quote'):
                            parts.append(f"  > 원문: \"{ev['abstract_quote']}\"")
                        if ev.get('critical_insight'):
                            parts.append(f"  ⚠️ {ev['critical_insight']}")

        # 5. 임상적 의의
        clinical = module.get("clinical_significance", {})
        if clinical:
            parts.append(f"\n## 임상적 의의")
            for key, val in clinical.items():
                parts.append(f"- {key}: {val}")

        # 6. 인용 템플릿
        template = module.get("response_template", {})
        if template:
            parts.append(f"\n## 응답 시 인용 형식")
            parts.append(f"예시: {template.get('example_usage', '')}")
            mandatory = template.get("mandatory_citations", [])
            if mandatory:
                parts.append("필수 인용:")
                for cite in mandatory:
                    parts.append(f"  - {cite}")
            if template.get("warning"):
                parts.append(f"\n⚠️ {template['warning']}")

        return "\n".join(parts)

    def _format_pcd_dqe_nps(self, module: Dict[str, Any]) -> str:
        """
        Phase 3: pcd_dqe_nps 모듈 전용 포맷팅

        DQE 선량 의존성, NPS 분해, PCD vs EID DQE 비교 근거 제공
        """
        parts = []
        parts.append("=" * 50)
        parts.append("[DQE/NPS 증거 자료 - Phase 3 Layer 3]")
        parts.append("=" * 50)

        # 1. Constitutional Axioms (Laws 7-9)
        core = module.get("core_physics", {})
        axioms = core.get("constitutional_axioms", {})
        if axioms:
            parts.append("\n## Phase 3 물리 공리 (DQE Laws)")
            for key, axiom in axioms.items():
                parts.append(f"- **{axiom['law']}**: {axiom['description']}")
                if axiom.get('implication'):
                    parts.append(f"  → {axiom['implication']}")

        # 2. DQE 공식
        dqe_formulas = core.get("dqe_formulas", {})
        if dqe_formulas:
            parts.append("\n## DQE 핵심 공식 (Solver 검증 완료)")
            for key, formula in dqe_formulas.items():
                parts.append(f"- {key}: {formula.get('example', formula.get('note', ''))}")

        # 3. NPS 분해
        nps = core.get("nps_decomposition", {})
        if nps:
            parts.append("\n## NPS 분해")
            for key, item in nps.items():
                parts.append(f"- {item.get('description', '')}")
                if item.get('example'):
                    parts.append(f"  예시: {item['example']}")

        # 4. Phase 1 교차 검증
        cross_val = core.get("phase1_cross_validation", {})
        if cross_val:
            parts.append(f"\n## Phase 1 교차 검증")
            parts.append(f"  {cross_val.get('verification', '')}")
            parts.append(f"  해석: {cross_val.get('interpretation', '')}")

        # 5. Evidence Database
        evidence = module.get("evidence_database", {})
        if evidence:
            parts.append("\n## 논문 근거 (Evidence Database)")

            # DQE 실측값
            dqe_meas = evidence.get("dqe_measurements", {})
            if dqe_meas:
                parts.append(f"\n### {dqe_meas.get('summary', 'DQE 실측값')}")
                for ev in dqe_meas.get("key_evidence", []):
                    level = ev.get('verification_level', 'unknown')
                    parts.append(f"- [{level}] [{ev.get('source', '')}]:")
                    parts.append(f"  \"{ev['finding']}\"")
                    if ev.get('abstract_quote'):
                        parts.append(f"  > 원문: \"{ev['abstract_quote']}\"")
                    if ev.get('solver_connection'):
                        parts.append(f"  → Solver 연결: {ev['solver_connection']}")

            # DQE 선량 의존성
            dose_dep = evidence.get("dose_dependence_mechanism", {})
            if dose_dep:
                parts.append(f"\n### {dose_dep.get('summary', 'DQE 선량 의존성')}")
                for ev in dose_dep.get("key_evidence", []):
                    level = ev.get('verification_level', 'unknown')
                    parts.append(f"- [{level}] [{ev.get('source', '')}]:")
                    parts.append(f"  \"{ev['finding']}\"")
                    if ev.get('abstract_quote'):
                        parts.append(f"  > 원문: \"{ev['abstract_quote']}\"")
                    if ev.get('full_text_quote'):
                        parts.append(f"  > Full text: \"{ev['full_text_quote']}\"")

            # NPS 비교
            nps_comp = evidence.get("nps_comparison", {})
            if nps_comp:
                parts.append(f"\n### {nps_comp.get('summary', 'NPS 비교')}")
                for ev in nps_comp.get("key_evidence", []):
                    level = ev.get('verification_level', 'unknown')
                    parts.append(f"- [{level}] [{ev.get('source', '')}]:")
                    parts.append(f"  \"{ev['finding']}\"")
                    if ev.get('caveat'):
                        parts.append(f"  ⚠️ 주의: {ev['caveat']}")

        # 6. Solver vs Literature
        solver_lit = module.get("solver_prediction_vs_literature", {})
        if solver_lit:
            parts.append("\n## Solver 예측 vs 문헌 비교")
            for key, pred in solver_lit.items():
                parts.append(f"- Solver: {pred.get('solver_parameter', '')}")
                parts.append(f"  문헌: {pred.get('literature_value', '')} ({pred.get('agreement', '')})")

        # 7. 흔한 오류
        misconceptions = module.get("common_misconceptions", {})
        if misconceptions:
            parts.append("\n## ⚠️ 흔한 오류")
            for key, m in misconceptions.items():
                parts.append(f"- ❌ \"{m['wrong']}\"")
                parts.append(f"  ✅ \"{m['correct']}\"")

        # 8. 제약 조건
        warnings = module.get("warning_constraints", {})
        if warnings:
            impossibilities = warnings.get("physical_impossibility", [])
            if impossibilities:
                parts.append("\n## 🚫 물리적 불가능 답변")
                for imp in impossibilities:
                    parts.append(f"  - {imp}")
            verified = warnings.get("solver_verified_values", {})
            if verified:
                parts.append("\n## 검증된 정답 (1% 초과 오차 시 거부)")
                for param_key, vals in verified.items():
                    for k, v in vals.items():
                        parts.append(f"  - {k}: {v}")

        return "\n".join(parts)

    def _format_pcd_mtf_resolution(self, module: Dict[str, Any]) -> str:
        """
        Phase 4: pcd_mtf_resolution 모듈 전용 포맷팅

        MTF, 공간 해상도, DQE(f) 비교 근거 제공
        """
        parts = []
        parts.append("=" * 50)
        parts.append("[MTF/Spatial Resolution 증거 자료 - Phase 4 Layer 3]")
        parts.append("=" * 50)

        # 1. Constitutional Axioms (Laws 10-12)
        core = module.get("core_physics", {})
        axioms = core.get("constitutional_axioms", {})
        if axioms:
            parts.append("\n## Phase 4 물리 공리 (MTF/Resolution Laws)")
            for key, axiom in axioms.items():
                parts.append(f"- **{axiom['law']}**: {axiom['description']}")
                if axiom.get('implication'):
                    parts.append(f"  -> {axiom['implication']}")

        # 2. MTF 모델
        mtf_models = core.get("mtf_models", {})
        if mtf_models:
            parts.append("\n## MTF 모델 (Solver 검증 완료)")
            pcd = mtf_models.get("pcd_direct_conversion", {})
            if pcd:
                parts.append(f"- PCD (직접 변환): {pcd.get('formula', '')}")
                parts.append(f"  Charge sharing: {pcd.get('cs_factor', '')}")
                parts.append(f"  Nyquist ideal: {pcd.get('ideal_at_nyquist', '')}")
            eid = mtf_models.get("eid_indirect_conversion", {})
            if eid:
                parts.append(f"- EID (간접 변환): {eid.get('formula', '')}")
                scint = eid.get("scintillator_models", {})
                for name, data in scint.items():
                    parts.append(f"  {name}: f_c = {data.get('f_c', '')}, {data.get('structure', '')}")

        # 3. DQE(f) 모델
        dqe_f = core.get("dqe_frequency", {})
        if dqe_f:
            parts.append("\n## DQE(f) 모델")
            parts.append(f"- PCD: {dqe_f.get('pcd_model', '')}")
            parts.append(f"- EID: {dqe_f.get('eid_model', '')}")
            parts.append(f"- Phase 3 교차검증: {dqe_f.get('phase3_crosscheck', '')}")
            parts.append(f"- PCD 우위: {dqe_f.get('pcd_advantage', '')}")

        # 4. 해상도 지표
        res_metrics = core.get("resolution_metrics", {})
        if res_metrics:
            parts.append("\n## 해상도 지표")
            typical = res_metrics.get("typical_values", {})
            for det, val in typical.items():
                parts.append(f"  - {det}: {val}")

        # 5. Evidence Database
        evidence = module.get("evidence_database", {})
        if evidence:
            parts.append("\n## 논문 근거 (Evidence Database)")

            for section_key in ['mtf_direct_conversion', 'charge_sharing', 'dqe_frequency_dependence', 'si_strip_mammography']:
                section = evidence.get(section_key, {})
                if section:
                    parts.append(f"\n### {section.get('summary', section_key)}")
                    for ev in section.get("key_evidence", []):
                        level = ev.get('verification_level', 'unknown')
                        parts.append(f"- [{level}] [{ev.get('source', '')}]:")
                        parts.append(f"  \"{ev['finding']}\"")
                        if ev.get('abstract_quote'):
                            parts.append(f"  > 원문: \"{ev['abstract_quote']}\"")
                        if ev.get('solver_connection'):
                            parts.append(f"  -> Solver: {ev['solver_connection']}")

        # 6. Solver vs Literature
        solver_lit = module.get("solver_prediction_vs_literature", {})
        if solver_lit:
            parts.append("\n## Solver 예측 vs 문헌 비교")
            for key, pred in solver_lit.items():
                parts.append(f"- Solver: {pred.get('solver_parameter', '')}")
                parts.append(f"  문헌: {pred.get('literature_value', '')} ({pred.get('agreement', '')})")

        # 7. 흔한 오류
        misconceptions = module.get("common_misconceptions", {})
        if misconceptions:
            parts.append("\n## 흔한 오류")
            for key, m in misconceptions.items():
                parts.append(f"- X \"{m['wrong']}\"")
                parts.append(f"  O \"{m['correct']}\"")

        # 8. 제약 조건
        warnings = module.get("warning_constraints", {})
        if warnings:
            impossibilities = warnings.get("physical_impossibility", [])
            if impossibilities:
                parts.append("\n## 물리적 불가능 답변")
                for imp in impossibilities:
                    parts.append(f"  - {imp}")

        # 9. 인용 템플릿
        template = module.get("response_template", {})
        if template:
            parts.append(f"\n## 응답 시 인용 형식")
            parts.append(f"예시: {template.get('example_usage', '')}")
            mandatory = template.get("mandatory_citations", [])
            if mandatory:
                parts.append("필수 인용:")
                for cite in mandatory:
                    parts.append(f"  - {cite}")
            if template.get("warning"):
                parts.append(f"\n! {template['warning']}")

        return "\n".join(parts)

    def _format_biopsy_geometry_calibration(self, module: Dict[str, Any]) -> str:
        """
        Phase 4-B: biopsy_geometry_calibration 모듈 전용 포맷팅

        스테레오 정위 생검의 기하학, 오차 전파, 교정 물리 근거 제공
        """
        parts = []
        parts.append("=" * 50)
        parts.append("[Biopsy Geometry & Calibration - Phase 4-B Layer 3]")
        parts.append("=" * 50)

        # 1. Constitutional Axioms (Laws 13-14)
        core = module.get("core_physics", {})
        axioms = core.get("constitutional_axioms", {})
        if axioms:
            parts.append("\n## Phase 4-B 물리 공리 (Biopsy Geometry Laws)")
            for key, axiom in axioms.items():
                parts.append(f"- **{axiom['law']}**: {axiom['description']}")
                parts.append(f"  공식: {axiom.get('formula', '')}")
                if axiom.get('implication'):
                    parts.append(f"  -> {axiom['implication']}")

        # 2. Geometry Model
        geometry = core.get("geometry_model", {})
        if geometry:
            parts.append("\n## 스테레오 기하학 모델")
            stereo = geometry.get("stereo_pair", {})
            if stereo:
                parts.append(f"  - 표준 각도: ±{stereo.get('standard_angle_deg', 15)}°")
                parts.append(f"  - 범위: {stereo.get('angle_range_deg', '')}")
                parts.append(f"  - 참고: {stereo.get('formula_note', '')}")

            coord = geometry.get("coordinate_calculation", {})
            if coord:
                parts.append("\n  좌표 산출:")
                for axis, formula in coord.items():
                    parts.append(f"    - {axis} = {formula}")

            amp_table = geometry.get("geometric_amplification_table", {})
            if amp_table:
                parts.append("\n  기하학적 증폭 테이블:")
                parts.append("  | 각도 | G (증폭) | 비고 |")
                parts.append("  |------|----------|------|")
                for key, data in amp_table.items():
                    parts.append(f"  | {data['angle_deg']}° | {data['G']:.3f} | {data['error_note']} |")

        # 3. Error Budget
        error_budget = core.get("error_budget", {})
        if error_budget:
            parts.append(f"\n## 오차 예산 (Error Budget)")
            parts.append(f"  공식: {error_budget.get('description', '')}")
            components = error_budget.get("components", {})
            for name, desc in components.items():
                parts.append(f"  - {name}: {desc}")
            parts.append(f"  ACR 허용: ≤{error_budget.get('acr_tolerance_mm', 1.0)}mm")
            parts.append(f"  출처: {error_budget.get('acr_source', '')}")

        # 4. Phase 4-A Connection
        phase4a = core.get("phase4a_connection", {})
        if phase4a:
            parts.append(f"\n## Phase 4-A(MTF) 연결")
            parts.append(f"  {phase4a.get('description', '')}")
            pcd_adv = phase4a.get("pcd_advantage", {})
            if pcd_adv:
                parts.append(f"  - PCD MTF(Ny): {pcd_adv.get('mtf_pcd_nyquist', '')}")
                parts.append(f"  - EID MTF(Ny): {pcd_adv.get('mtf_eid_nyquist', '')}")
                parts.append(f"  - σ_Δx PCD: {pcd_adv.get('sigma_dx_pcd_mm', '')}")
                parts.append(f"  - σ_Δx EID: {pcd_adv.get('sigma_dx_eid_mm', '')}")
                parts.append(f"  - 개선율: {pcd_adv.get('targeting_improvement', '')}")
            parts.append(f"  기전: {phase4a.get('mechanism', '')}")

        # 5. Evidence Database
        evidence = module.get("evidence_database", {})
        if evidence:
            parts.append("\n## 논문 근거 (Evidence Database)")

            for section_key in ['stereotactic_geometry', 'clinical_accuracy_standards', 'detector_resolution_impact', 'calibration_physics']:
                section = evidence.get(section_key, {})
                if section:
                    parts.append(f"\n### {section.get('summary', section_key)}")
                    for ev in section.get("key_evidence", []):
                        level = ev.get('verification_level', 'unknown')
                        parts.append(f"- [{level}] [{ev.get('source', '')}]:")
                        parts.append(f"  \"{ev['finding']}\"")
                        if ev.get('abstract_quote'):
                            parts.append(f"  > 원문: \"{ev['abstract_quote']}\"")
                        if ev.get('solver_connection'):
                            parts.append(f"  -> Solver: {ev['solver_connection']}")

        # 6. Solver vs Literature
        solver_lit = module.get("solver_prediction_vs_literature", {})
        if solver_lit:
            parts.append("\n## Solver 예측 vs 문헌 비교")
            for key, pred in solver_lit.items():
                parts.append(f"- Solver: {pred.get('solver_parameter', '')}")
                parts.append(f"  문헌: {pred.get('literature_value', '')} ({pred.get('agreement', '')})")

        # 7. 흔한 오류
        misconceptions = module.get("common_misconceptions", {})
        if misconceptions:
            parts.append("\n## 흔한 오류")
            for key, m in misconceptions.items():
                parts.append(f"- X \"{m['wrong']}\"")
                parts.append(f"  O \"{m['correct']}\"")

        # 8. 제약 조건
        warnings = module.get("warning_constraints", {})
        if warnings:
            impossibilities = warnings.get("physical_impossibility", [])
            if impossibilities:
                parts.append("\n## 물리적 불가능 답변")
                for imp in impossibilities:
                    parts.append(f"  - {imp}")
            verified = warnings.get("solver_verified_values", {})
            if verified:
                parts.append("\n## 검증된 정답 (1% 초과 오차 시 거부)")
                for param_key, vals in verified.items():
                    for k, v in vals.items():
                        parts.append(f"  - {k}: {v}")

        # 9. 인용 템플릿
        template = module.get("response_template", {})
        if template:
            parts.append(f"\n## 응답 시 인용 형식")
            parts.append(f"예시: {template.get('example_usage', '')}")
            mandatory = template.get("mandatory_citations", [])
            if mandatory:
                parts.append("필수 인용:")
                for cite in mandatory:
                    parts.append(f"  - {cite}")
            if template.get("warning"):
                parts.append(f"\n! {template['warning']}")

        return "\n".join(parts)

    def _format_dbt_image_quality(self, module: Dict[str, Any]) -> str:
        """
        Phase 5: dbt_image_quality 모듈 전용 포맷팅

        토모합성 영상 품질: dose-split DQE, 분해능 비대칭, clutter rejection 근거 제공
        """
        parts = []
        parts.append("=" * 50)
        parts.append("[DBT Image Quality - Phase 5 Layer 3]")
        parts.append("=" * 50)

        # 1. Constitutional Axioms (Laws 16-18)
        core = module.get("core_physics", {})
        axioms = core.get("constitutional_axioms", {})
        if axioms:
            parts.append("\n## Phase 5 물리 공리 (Tomo Image Quality Laws)")
            for key, axiom in axioms.items():
                parts.append(f"- **{axiom['law']}**: {axiom['description']}")
                parts.append(f"  공식: {axiom.get('formula', '')}")
                if axiom.get('implication'):
                    parts.append(f"  -> {axiom['implication']}")

        # 2. Dose-Split Model
        dose_model = core.get("dose_split_model", {})
        if dose_model:
            parts.append("\n## Dose-Split DQE 모델 (Solver 검증 완료)")
            parts.append(f"  - α 유도: {dose_model.get('alpha_derivation', '')}")
            parts.append(f"  - EID: {dose_model.get('eid_dqe_formula', '')}")
            parts.append(f"  - PCD: {dose_model.get('pcd_dqe_formula', '')}")
            parts.append(f"  - SNR 비: {dose_model.get('snr_ratio', '')}")
            examples = dose_model.get("example_values", {})
            if examples:
                parts.append("  예시:")
                for n_key, data in examples.items():
                    note = f" ({data['note']})" if data.get('note') else ""
                    parts.append(f"    {n_key}: R = {data['R']}{note}")

        # 3. Clutter Model
        clutter = core.get("clutter_model", {})
        if clutter:
            parts.append(f"\n## Clutter Rejection 모델")
            parts.append(f"  - 공식: {clutter.get('gain_formula', '')}")
            parts.append(f"  - SNR boost: {clutter.get('snr_boost', '')}")
            parts.append(f"  - 예시: {clutter.get('example', '')}")

        # 4. Evidence Database
        evidence = module.get("evidence_database", {})
        if evidence:
            parts.append("\n## 논문 근거 (Evidence Database)")

            for section_key in ['dose_split_immunity', '3d_image_quality_framework',
                                'anatomical_clutter', 'clinical_dose_reduction',
                                'resolution_asymmetry']:
                section = evidence.get(section_key, {})
                if section:
                    parts.append(f"\n### {section.get('summary', section_key)}")
                    for ev in section.get("key_evidence", []):
                        level = ev.get('verification_level', 'unknown')
                        parts.append(f"- [{level}] [{ev.get('source', '')}]:")
                        parts.append(f"  \"{ev['finding']}\"")
                        if ev.get('abstract_quote'):
                            parts.append(f"  > 원문: \"{ev['abstract_quote']}\"")
                        if ev.get('solver_connection'):
                            parts.append(f"  -> Solver: {ev['solver_connection']}")

        # 5. Solver vs Literature
        solver_lit = module.get("solver_prediction_vs_literature", {})
        if solver_lit:
            parts.append("\n## Solver 예측 vs 문헌 비교")
            for key, pred in solver_lit.items():
                parts.append(f"- Solver: {pred.get('solver_parameter', '')}")
                parts.append(f"  문헌: {pred.get('literature_value', '')} ({pred.get('agreement', '')})")

        # 6. 흔한 오류
        misconceptions = module.get("common_misconceptions", {})
        if misconceptions:
            parts.append("\n## 흔한 오류")
            for key, m in misconceptions.items():
                parts.append(f"- X \"{m['wrong']}\"")
                parts.append(f"  O \"{m['correct']}\"")

        # 7. 제약 조건
        warnings = module.get("warning_constraints", {})
        if warnings:
            impossibilities = warnings.get("physical_impossibility", [])
            if impossibilities:
                parts.append("\n## 물리적 불가능 답변")
                for imp in impossibilities:
                    parts.append(f"  - {imp}")
            verified = warnings.get("solver_verified_values", {})
            if verified:
                parts.append("\n## 검증된 정답 (1% 초과 오차 시 거부)")
                for param_key, vals in verified.items():
                    parts.append(f"  [{param_key}]")
                    for k, v in vals.items():
                        parts.append(f"    - {k}: {v}")

        # 8. 인용 템플릿
        template = module.get("response_template", {})
        if template:
            parts.append(f"\n## 응답 시 인용 형식")
            parts.append(f"예시: {template.get('example_usage', '')}")
            mandatory = template.get("mandatory_citations", [])
            if mandatory:
                parts.append("필수 인용:")
                for cite in mandatory:
                    parts.append(f"  - {cite}")
            if template.get("warning"):
                parts.append(f"\n! {template['warning']}")

        return "\n".join(parts)

    def _format_core_physics(self, module: Dict[str, Any], query: str = "") -> str:
        """
        Phase 7.19: core_physics 모듈 전용 포맷팅

        Constitutional axioms와 golden formulas를 LLM이 이해하기 쉬운 형태로 변환
        """
        parts = []
        parts.append("=" * 50)
        parts.append("[Core Physics - Constitutional Axioms & Golden Formulas]")
        parts.append("=" * 50)

        query_lower = query.lower()

        # Constitutional Axioms
        axioms_data = module.get("constitutional_axioms", {})
        if axioms_data:
            # Phase별 공리 포맷팅
            for phase_key in ["phase1_fundamental", "phase2_contrast", "phase3_dqe",
                             "phase4_mtf", "phase4b_biopsy", "phase5_tomo"]:
                phase = axioms_data.get(phase_key, {})
                if not phase:
                    continue

                title = phase.get("title", phase_key)
                parts.append(f"\n### {title}")

                laws = phase.get("laws", [])
                for law in laws:
                    parts.append(f"- **{law.get('name', '')}**: {law.get('statement', law.get('formula', ''))}")
                    if law.get("warning"):
                        parts.append(f"  (Warning: {law['warning']})")

                # Key derivations
                derivations = phase.get("key_derivations", [])
                if derivations:
                    for d in derivations:
                        parts.append(f"  - {d}")

                # Warning
                if phase.get("warning"):
                    parts.append(f"Warning: {phase['warning']}")

        # Golden Formulas - 질문 관련 공식만 필터링
        golden = module.get("golden_formulas", {})
        if golden:
            parts.append("\n## Golden Formulas (검증된 공식)")

            # 질문 키워드로 관련 공식 필터링
            formula_keywords = {
                "MGD_2D": ["mgd", "mean glandular", "선량", "dose", "dance"],
                "MGD_DBT_T": ["mgd", "dbt", "tomosynthesis", "토모", "t-factor", "T-factor"],
                "MGD_DBT_t": ["mgd", "dbt", "projection", "t-factor", "토모"],
                "T_FACTOR": ["t-factor", "T-factor", "projection", "dbt"],
                "QUANTUM_NOISE": ["noise", "quantum", "노이즈", "양자", "포아송"],
                "SNR_QUANTUM": ["snr", "signal", "noise", "신호"],
                "DOSE_SNR_RELATION": ["snr", "dose", "선량"],
                "HVL": ["hvl", "half value", "반가층"],
                "ATTENUATION": ["attenuation", "감쇠", "beer", "lambert"],
                "CONTRAST": ["contrast", "대조도"],
                "CNR": ["cnr", "contrast", "noise"],
                "DQE": ["dqe", "detective", "quantum efficiency"],
                "MAS_CALCULATION": ["mas", "exposure", "노출"],
                "SENSITIVITY": ["sensitivity", "민감도", "tpr"],
                "SPECIFICITY": ["specificity", "특이도"],
                "PPV": ["ppv", "positive predictive"],
                "NPV": ["npv", "negative predictive"],
                "ACCURACY": ["accuracy", "정확도"],
            }

            # 쿼리와 매칭되는 공식 찾기
            matched_formulas = []
            for formula_id, keywords in formula_keywords.items():
                if formula_id in golden and any(kw in query_lower for kw in keywords):
                    matched_formulas.append(formula_id)

            # 매칭된 공식이 없으면 주요 공식 5개만 표시
            if not matched_formulas:
                matched_formulas = ["MGD_2D", "SNR_QUANTUM", "DQE", "CNR", "HVL"]

            for formula_id in matched_formulas:
                formula = golden.get(formula_id, {})
                if not formula:
                    continue

                parts.append(f"\n**{formula.get('name', formula_id)}**")
                parts.append(f"  {formula.get('formula_unicode', formula.get('formula_latex', ''))}")

                # 변수 설명
                variables = formula.get("variables", {})
                if variables and len(variables) <= 5:
                    var_str = ", ".join([f"{v}={d.get('description', '')}"
                                        for v, d in list(variables.items())[:3]])
                    parts.append(f"  변수: {var_str}")

                if formula.get("source"):
                    parts.append(f"  출처: {formula['source']}")

        # Knowledge Blocks - 핵심 정보만
        blocks = module.get("knowledge_blocks", {})
        if blocks and "tomo" in query_lower or "mgd" in query_lower:
            mgd_tomo = blocks.get("mgd_tomosynthesis", {})
            if mgd_tomo:
                parts.append("\n## MGD for Tomosynthesis (Dance et al. 2011)")
                concepts = mgd_tomo.get("concepts", {})

                t_factor = concepts.get("t_factor", {})
                if t_factor:
                    parts.append(f"- t-factor: {t_factor.get('definition', '')}")
                    parts.append(f"  {t_factor.get('formula', '')}")

                T_factor = concepts.get("T_factor", {})
                if T_factor:
                    parts.append(f"- T-factor: {T_factor.get('definition', '')}")
                    parts.append(f"  {T_factor.get('formula', '')}")
                    typical = T_factor.get("typical_values", {})
                    if typical:
                        parts.append(f"  Full-field: {typical.get('full_field', '')}")

                # t-factor 테이블
                t_table = mgd_tomo.get("t_factor_table_50_glandularity_W_Al", [])
                if t_table:
                    parts.append("\n  t-factor by thickness (50% glandularity, W/Al):")
                    parts.append("  | Thickness | t(0°) | t(10°) | t(20°) | t(30°) |")
                    parts.append("  |-----------|-------|--------|--------|--------|")
                    for row in t_table[:4]:
                        parts.append(f"  | {row.get('thickness_cm')} cm | {row.get('t_0', 1.0):.3f} | "
                                   f"{row.get('t_10', 0.98):.3f} | {row.get('t_20', 0.92):.3f} | "
                                   f"{row.get('t_30', 0.85):.3f} |")

        # Strict Rules
        rules_data = module.get("strict_rules", {})
        rules = rules_data.get("rules", [])
        if rules:
            parts.append("\n## Strict Rules (필수 준수)")
            for rule in rules:
                parts.append(f"{rule.get('id', '-')}. {rule.get('name', '')}: {rule.get('statement', '')}")

        return "\n".join(parts)

    def _format_contrast_enhanced_mammography(self, module: Dict[str, Any], query: str = "") -> str:
        """
        Phase 7.24: contrast_enhanced_mammography 모듈 전용 포맷팅

        DICOM 관련 질문 시 dicom_contrast_timing 섹션을 최우선 배치하여
        Lost in the Middle 현상 방지
        """
        parts = []
        parts.append("=" * 50)
        parts.append("[조영증강 유방촬영(CEM) 참조 자료 - Phase 7.24]")
        parts.append("=" * 50)

        # Phase 7.24: 질문 키워드 기반 섹션 우선순위 결정
        query_lower = query.lower()

        content = module.get("content", {})

        # 섹션 포맷팅 함수들 정의
        def format_dicom_contrast_timing():
            """DICOM 조영제 타이밍 섹션 - 핵심 정보"""
            section_parts = []
            dicom = content.get("dicom_contrast_timing", {})
            if not dicom:
                return section_parts

            section_parts.append("\n" + "=" * 50)
            section_parts.append("## ⚠️ DICOM 조영제 타이밍 정보 (검증됨)")
            section_parts.append("=" * 50)

            # WARNING 먼저 표시
            warning = dicom.get("WARNING", "")
            if warning:
                section_parts.append(f"\n**{warning}**")

            # 핵심 문제 요약
            problem = dicom.get("problem_summary", {})
            if problem:
                section_parts.append("\n### 핵심 문제 요약")
                section_parts.append(f"- **질문**: {problem.get('question', '')}")
                section_parts.append(f"- **답변**: {problem.get('answer', '')}")
                section_parts.append(f"- **출처**: {problem.get('source', '')}")

            # DICOM 태그 정보 - 평탄화하여 출력
            bolus_module = dicom.get("dicom_contrast_bolus_module", {})
            if bolus_module:
                section_parts.append("\n### DICOM Contrast/Bolus Module 태그")
                section_parts.append(f"- 표준: {bolus_module.get('standard', '')}")
                section_parts.append(f"- **태그 유형: {bolus_module.get('tag_type', '')}**")

                tags = bolus_module.get("available_tags", {})

                # Injection Timing 태그
                injection = tags.get("injection_timing", {})
                if injection:
                    section_parts.append("\n**주입 시간 태그:**")
                    section_parts.append("| 태그 | 이름 | 유형 | 설명 |")
                    section_parts.append("|------|------|------|------|")
                    for tag_id, tag_info in injection.items():
                        section_parts.append(
                            f"| {tag_id} | {tag_info.get('name', '')} | **{tag_info.get('type', '')}** | {tag_info.get('description', '')} |"
                        )

                # Volume/Dose 태그
                volume = tags.get("volume_dose", {})
                if volume:
                    section_parts.append("\n**용량 태그:**")
                    for tag_id, tag_info in volume.items():
                        section_parts.append(
                            f"- {tag_id}: {tag_info.get('name', '')} ({tag_info.get('type', '')})"
                        )

                # Flow 태그
                flow = tags.get("flow_parameters", {})
                if flow:
                    section_parts.append("\n**유속 태그:**")
                    for tag_id, tag_info in flow.items():
                        section_parts.append(
                            f"- {tag_id}: {tag_info.get('name', '')} ({tag_info.get('type', '')})"
                        )

            # 근본 원인 분석
            root_cause = dicom.get("root_cause_analysis", {})
            if root_cause:
                section_parts.append("\n### 근본 원인 분석")
                section_parts.append(f"**주요 원인**: {root_cause.get('primary_cause', '')}")

                factors = root_cause.get("contributing_factors", [])
                if factors:
                    section_parts.append("\n**기여 요인:**")
                    for f in factors:
                        section_parts.append(f"- {f.get('factor', '')}: {f.get('impact', '')}")
                        if f.get('source'):
                            section_parts.append(f"  (출처: {f.get('source')})")

                workaround = root_cause.get("current_workaround", {})
                if workaround:
                    section_parts.append(f"\n**현재 우회 방법**: {workaround.get('method', '')}")
                    section_parts.append(f"- 한계: {workaround.get('limitation', '')}")

            # 해결책/권장사항
            solutions = dicom.get("solutions_recommendations", {})
            if solutions:
                section_parts.append("\n### 해결책 및 권장사항")
                section_parts.append(f"(출처: {solutions.get('source', '')})")

                short_term = solutions.get("short_term", [])
                if short_term:
                    section_parts.append("\n**단기 해결책:**")
                    for s in short_term:
                        section_parts.append(f"- {s}")

                long_term = solutions.get("long_term", [])
                if long_term:
                    section_parts.append("\n**장기 해결책:**")
                    for s in long_term:
                        section_parts.append(f"- {s}")

            return section_parts

        def format_principles():
            """CEM 기본 원리"""
            section_parts = []
            principles = content.get("principles", {})
            if principles:
                section_parts.append("\n## CEM 기본 원리")
                section_parts.append(f"- 설명: {principles.get('description', '')}")
                section_parts.append(f"- 메커니즘: {principles.get('mechanism', '')}")
                section_parts.append(f"- 출처: {principles.get('source_ref', '')}")
            return section_parts

        def format_dual_energy():
            """이중 에너지 기법"""
            section_parts = []
            dual = content.get("dual_energy_technique", {})
            if dual:
                section_parts.append("\n## 이중 에너지 기법")
                section_parts.append(f"- 요오드 K-edge: {dual.get('iodine_k_edge_keV', '')} keV")

                le = dual.get("low_energy_image", {})
                if le:
                    section_parts.append(f"- LE 영상: {le.get('tube_voltage_kVp', '')} kVp - {le.get('description', '')}")

                he = dual.get("high_energy_image", {})
                if he:
                    section_parts.append(f"- HE 영상: {he.get('tube_voltage_kVp', '')} kVp - {he.get('description', '')}")

                recomb = dual.get("recombined_image", {})
                if recomb:
                    section_parts.append(f"- 재조합: {recomb.get('description', '')}")
                    section_parts.append(f"  공식: {recomb.get('formula', '')}")
            return section_parts

        def format_acquisition_protocol():
            """획득 프로토콜"""
            section_parts = []
            protocol = content.get("acquisition_protocol", {})
            if protocol:
                section_parts.append("\n## 획득 프로토콜")

                injection = protocol.get("contrast_injection", {})
                if injection:
                    section_parts.append(f"- 조영제: {injection.get('agent', '')}")
                    section_parts.append(f"- 용량: {injection.get('dose_ml_kg', '')} mL/kg")
                    section_parts.append(f"- 주입 속도: {injection.get('injection_rate_ml_s', '')} mL/s")
                    section_parts.append(f"- 지연 시간: {injection.get('delay_min', '')}분")

                sequence = protocol.get("imaging_sequence", {})
                if sequence:
                    section_parts.append(f"- 촬영 순서: {sequence.get('order', '')}")
                    section_parts.append(f"- 총 소요 시간: {sequence.get('total_time_min', '')}분")
            return section_parts

        def format_clinical_performance():
            """임상 성능"""
            section_parts = []
            perf = content.get("clinical_performance", {})
            if perf:
                section_parts.append("\n## 임상 성능")

                sens = perf.get("sensitivity", {})
                if sens:
                    section_parts.append(f"- CEM 민감도: {sens.get('cem', '')}")
                    section_parts.append(f"- 기존 맘모그래피: {sens.get('mammography_alone', '')}")
                    section_parts.append(f"- 비교: {sens.get('comparison', '')}")

                spec = perf.get("specificity", {})
                if spec:
                    section_parts.append(f"- CEM 특이도: {spec.get('cem', '')}")
            return section_parts

        def format_artifacts():
            """아티팩트"""
            section_parts = []
            artifacts = content.get("artifacts", {})
            if artifacts:
                section_parts.append("\n## CEM 아티팩트")

                types = artifacts.get("types", {})
                for name, data in types.items():
                    section_parts.append(f"\n### {name}")
                    section_parts.append(f"- 설명: {data.get('description', '')}")
                    section_parts.append(f"- 원인: {data.get('cause', '')}")
                    section_parts.append(f"- 발생률: {data.get('occurrence_rate_percent', '')}%")

                reduction = artifacts.get("artifact_reduction", {})
                if reduction:
                    section_parts.append(f"\n아티팩트 감소: {reduction.get('finding', '')}")
            return section_parts

        def format_magnification_physics():
            """확대 촬영 SNR 물리학"""
            section_parts = []
            mag = content.get("magnification_snr_physics", {})
            if mag:
                section_parts.append("\n## 확대 촬영 SNR 물리학")

                warning = mag.get("WARNING", "")
                if warning:
                    section_parts.append(f"**{warning}**")

                inv_square = mag.get("inverse_square_law", {})
                if inv_square:
                    section_parts.append(f"- 역제곱 법칙: {inv_square.get('principle', '')}")
                    section_parts.append(f"  공식: {inv_square.get('formula', '')}")

                rose = mag.get("rose_criterion", {})
                if rose:
                    section_parts.append(f"\n- Rose Criterion: CNR > {rose.get('threshold', 5)}")
                    section_parts.append(f"  {rose.get('violation_risk', '')}")
            return section_parts

        # DICOM 관련 키워드
        dicom_keywords = ["dicom", "태그", "tag", "주입 시간", "injection time", "acquisition time",
                        "촬영 시점", "시간 정보", "timing", "metadata", "메타데이터",
                        "(0018,1042)", "(0018,1043)", "contrast bolus", "type 3", "optional"]

        # 프로토콜 관련 키워드
        protocol_keywords = ["protocol", "프로토콜", "injection", "주입", "timing window", "시간창"]

        # 아티팩트 관련 키워드
        artifact_keywords = ["artifact", "아티팩트", "ripple", "halo", "ghost", "잔상"]

        # 확대 관련 키워드
        mag_keywords = ["magnification", "확대", "snr", "rose criterion"]

        # 키워드 매칭
        is_dicom_related = any(kw in query_lower for kw in dicom_keywords)
        is_protocol_related = any(kw in query_lower for kw in protocol_keywords)
        is_artifact_related = any(kw in query_lower for kw in artifact_keywords)
        is_mag_related = any(kw in query_lower for kw in mag_keywords)

        # 동적 섹션 순서 결정
        if is_dicom_related:
            # DICOM 관련: dicom_contrast_timing을 맨 앞에 배치
            section_order = [
                format_dicom_contrast_timing,
                format_acquisition_protocol,
                format_principles,
                format_dual_energy,
                format_clinical_performance,
                format_artifacts,
                format_magnification_physics,
            ]
        elif is_protocol_related:
            # 프로토콜 관련
            section_order = [
                format_acquisition_protocol,
                format_dicom_contrast_timing,
                format_principles,
                format_dual_energy,
                format_clinical_performance,
                format_artifacts,
                format_magnification_physics,
            ]
        elif is_artifact_related:
            # 아티팩트 관련
            section_order = [
                format_artifacts,
                format_principles,
                format_dual_energy,
                format_acquisition_protocol,
                format_clinical_performance,
                format_dicom_contrast_timing,
                format_magnification_physics,
            ]
        elif is_mag_related:
            # 확대 촬영 관련
            section_order = [
                format_magnification_physics,
                format_dual_energy,
                format_acquisition_protocol,
                format_clinical_performance,
                format_principles,
                format_artifacts,
                format_dicom_contrast_timing,
            ]
        else:
            # 기본 순서
            section_order = [
                format_principles,
                format_dual_energy,
                format_acquisition_protocol,
                format_clinical_performance,
                format_artifacts,
                format_dicom_contrast_timing,
                format_magnification_physics,
            ]

        # 결정된 순서대로 섹션 포맷팅
        for format_func in section_order:
            parts.extend(format_func())

        # 출처 정보 추가
        sources = module.get("sources", [])
        if sources:
            parts.append("\n## 참조 문헌")
            for src in sources[:5]:  # 상위 5개만
                citation = src.get("citation", "")
                if citation:
                    parts.append(f"- {citation}")
                    if src.get("verified_quotes"):
                        parts.append("  [전문 검증 완료]")

        return "\n".join(parts)

    def get_knowledge_for_query(self, query: str) -> str:
        """
        질문에 맞는 지식을 검색하고 포맷팅하여 반환

        Args:
            query: 사용자 질문

        Returns:
            LLM 컨텍스트용 포맷팅된 지식 문자열

        Phase 7.18: query 기반 동적 섹션 배치 지원
        """
        modules = self.get_relevant_knowledge(query)
        return self.format_for_context(modules, query=query)

    def get_all_knowledge_ids(self) -> List[str]:
        """등록된 모든 지식 모듈 ID 반환"""
        return list(self._cache.keys())

    def get_knowledge_by_id(self, knowledge_id: str) -> Optional[Dict]:
        """특정 ID의 지식 모듈 반환"""
        return self._cache.get(knowledge_id)


# 싱글톤 인스턴스
_manager_instance: Optional[KnowledgeManager] = None


def get_knowledge_manager() -> KnowledgeManager:
    """KnowledgeManager 싱글톤 인스턴스 반환"""
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = KnowledgeManager()
    return _manager_instance


# 편의 함수
def get_relevant_physics_knowledge(query: str) -> str:
    """
    질문에 관련된 물리 지식 반환 (편의 함수)

    Args:
        query: 사용자 질문

    Returns:
        포맷팅된 물리 지식 문자열
    """
    manager = get_knowledge_manager()
    return manager.get_knowledge_for_query(query)


# 테스트
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    manager = KnowledgeManager()

    print("=" * 60)
    print("등록된 지식 모듈:", manager.get_all_knowledge_ids())
    print("=" * 60)

    test_queries = [
        "DBT에서 MGD 계산 시 T-factor의 의미는?",
        "5cm 유방의 t-factor 값은?",
        "Dance et al. 논문에서 정의한 선량 공식",
        "SNR과 노이즈의 관계",  # 매칭 안 됨 (아직 SNR 모듈 없음)
    ]

    for query in test_queries:
        print(f"\n질문: {query}")
        print("-" * 40)
        knowledge = manager.get_knowledge_for_query(query)
        if knowledge:
            print(knowledge[:500] + "..." if len(knowledge) > 500 else knowledge)
        else:
            print("(관련 지식 없음)")
