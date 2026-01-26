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
        if is_tomo_iq_query and 'dbt_image_quality' in matched_list:
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
    ) -> str:
        """
        지식 모듈을 LLM 컨텍스트용 문자열로 포맷팅

        Args:
            knowledge_modules: 지식 모듈 리스트
            include_tables: 테이블 포함 여부
            include_formulas: 수식 포함 여부
            include_common_qa: 자주 묻는 질문 포함 여부 (기본 False - LLM 혼란 방지)

        Returns:
            포맷팅된 문자열
        """
        if not knowledge_modules:
            return ""

        parts = []

        for module in knowledge_modules:
            module_id = module.get("id", "unknown")

            # Phase 7.3: detector_physics 모듈 전용 포맷팅
            if module_id == "detector_physics":
                parts.append(self._format_detector_physics(module))
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

    def _format_detector_physics(self, module: Dict[str, Any]) -> str:
        """
        Phase 7.3: detector_physics 모듈 전용 포맷팅

        검출기 물리학 지식을 LLM이 이해하기 쉬운 형태로 변환
        """
        parts = []
        parts.append("=" * 50)
        parts.append("[검출기 물리학 표준 참조 자료 - Phase 7.3]")
        parts.append("=" * 50)

        # 1. 검출기 유형 비교
        detector_types = module.get("detector_types", {})
        if detector_types:
            parts.append("\n## 검출기 유형 비교")

            # Direct conversion (a-Se)
            direct = detector_types.get("direct_conversion", {})
            a_se = direct.get("a_Se", {})
            if a_se:
                parts.append("\n### 직접변환 방식 (a-Se)")
                parts.append(f"- 원자번호: {a_se.get('atomic_number', 'N/A')}")
                parts.append(f"- K-edge: {a_se.get('k_edge_keV', 'N/A')} keV")
                dqe = a_se.get("typical_DQE", {})
                mtf = a_se.get("typical_MTF", {})
                parts.append(f"- DQE(0): {dqe.get('at_0_lp_mm', 'N/A')}")
                parts.append(f"- DQE(5 lp/mm): {dqe.get('at_5_lp_mm', 'N/A')}")
                parts.append(f"- MTF(5 lp/mm): {mtf.get('at_5_lp_mm', 'N/A')}")
                parts.append(f"- 장점: {', '.join(a_se.get('advantages', [])[:3])}")
                parts.append(f"- 단점: {', '.join(a_se.get('disadvantages', [])[:2])}")

            # Indirect conversion (CsI)
            indirect = detector_types.get("indirect_conversion", {})
            csi = indirect.get("CsI_aSi", {})
            if csi:
                parts.append("\n### 간접변환 방식 (CsI/a-Si)")
                parts.append(f"- 섬광체: {csi.get('scintillator', 'N/A')}")
                parts.append(f"- 구조: {csi.get('structure', 'N/A')}")
                dqe = csi.get("typical_DQE", {})
                mtf = csi.get("typical_MTF", {})
                parts.append(f"- DQE(0): {dqe.get('at_0_lp_mm', 'N/A')}")
                parts.append(f"- DQE(5 lp/mm): {dqe.get('at_5_lp_mm', 'N/A')}")
                parts.append(f"- MTF(5 lp/mm): {mtf.get('at_5_lp_mm', 'N/A')}")
                parts.append(f"- 장점: {', '.join(csi.get('advantages', [])[:3])}")
                parts.append(f"- 단점: {', '.join(csi.get('disadvantages', [])[:2])}")

            # Comparison table
            comp = detector_types.get("comparison_table", {})
            if comp:
                parts.append("\n### 비교 요약")
                params = comp.get("parameter", [])
                a_se_vals = comp.get("a_Se_direct", [])
                csi_vals = comp.get("CsI_indirect", [])
                implications = comp.get("clinical_implication", [])
                parts.append("| 항목 | a-Se (직접) | CsI (간접) | 임상적 의미 |")
                parts.append("|------|-------------|------------|-------------|")
                for i, param in enumerate(params):
                    a_val = a_se_vals[i] if i < len(a_se_vals) else ""
                    c_val = csi_vals[i] if i < len(csi_vals) else ""
                    impl = implications[i] if i < len(implications) else ""
                    parts.append(f"| {param} | {a_val} | {c_val} | {impl} |")

        # 2. DQE 물리학
        dqe_physics = module.get("DQE_physics", {})
        if dqe_physics:
            parts.append("\n## DQE 물리학")
            parts.append(f"- 정의: {dqe_physics.get('definition', '')}")
            parts.append(f"- 공식: {dqe_physics.get('formula', '')}")
            parts.append(f"- 확장 공식: {dqe_physics.get('expanded_formula', '')}")

        # 3. 선량 최적화
        dose_opt = module.get("dose_optimization", {})
        if dose_opt:
            parts.append("\n## 선량 최적화 파라미터")
            params = dose_opt.get("parameters", {})

            kvp = params.get("kVp", {})
            if kvp:
                parts.append(f"\n### kVp (관전압)")
                parts.append(f"- 범위: {kvp.get('typical_range_mammography', 'N/A')}")
                parts.append(f"- 선량 관계: {kvp.get('dose_relationship', 'N/A')}")
                parts.append(f"- 최적화 규칙: {kvp.get('optimization_rule', 'N/A')}")

            mas = params.get("mAs", {})
            if mas:
                parts.append(f"\n### mAs (관전류-시간적)")
                parts.append(f"- 선량 관계: {mas.get('dose_relationship', 'N/A')}")
                parts.append(f"- 노이즈 효과: {mas.get('effect_on_noise', 'N/A')}")
                parts.append(f"- 최적화 규칙: {mas.get('optimization_rule', 'N/A')}")

            tf = params.get("target_filter", {})
            if tf:
                parts.append("\n### Target/Filter 조합")
                combos = tf.get("combinations", {})
                for name, data in combos.items():
                    parts.append(f"- {name}: {data.get('use', '')} (HVL: {data.get('HVL_mmAl', '')})")
                reduction = tf.get("dose_reduction_by_filter_change", {})
                if reduction:
                    parts.append("\n선량 감소 효과:")
                    for change, effect in reduction.items():
                        parts.append(f"  - {change}: {effect}")

            # CNR maintenance formulas
            cnr = dose_opt.get("CNR_maintenance_formulas", {})
            if cnr:
                parts.append("\n### CNR 유지 공식")
                parts.append(f"- Rose Criterion: {cnr.get('Rose_Criterion', 'CNR >= 5')}")
                parts.append(f"- CNR vs Dose: {cnr.get('CNR_vs_dose', '')}")
                maintain = cnr.get("dose_reduction_with_CNR_maintenance", {})
                if maintain:
                    parts.append(f"- DQE 증가 공식: {maintain.get('required_DQE_increase', '')}")
                    parts.append(f"- 예시 (10%): {maintain.get('example_10_percent', '')}")
                    parts.append(f"- 예시 (20%): {maintain.get('example_20_percent', '')}")

        # 4. 고급 메트릭
        advanced = module.get("advanced_metrics", {})
        if advanced:
            parts.append("\n## 고급 검출 메트릭")
            for metric_name, metric_data in advanced.items():
                parts.append(f"\n### {metric_name}")
                parts.append(f"- 정의: {metric_data.get('definition', '')}")
                if metric_data.get('formula'):
                    parts.append(f"- 공식: {metric_data.get('formula', '')}")
                if metric_data.get('clinical_threshold'):
                    parts.append(f"- 임상 기준: {metric_data.get('clinical_threshold', '')}")

        # 5. 응용 예시
        examples = module.get("clinical_application_examples", {})
        if examples:
            parts.append("\n## 계산 예시")
            for ex_name, ex_data in examples.items():
                parts.append(f"\n### {ex_data.get('question', ex_name)}")
                calc = ex_data.get("calculation", {})
                for step, value in calc.items():
                    parts.append(f"  - {step}: {value}")
                parts.append(f"  → 결론: {ex_data.get('conclusion', '')}")

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

    def get_knowledge_for_query(self, query: str) -> str:
        """
        질문에 맞는 지식을 검색하고 포맷팅하여 반환

        Args:
            query: 사용자 질문

        Returns:
            LLM 컨텍스트용 포맷팅된 지식 문자열
        """
        modules = self.get_relevant_knowledge(query)
        return self.format_for_context(modules)

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
