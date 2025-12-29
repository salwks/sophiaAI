"""
MARIA-Mammo: Paper Classifier
=============================
논문 자동 분류 (Modality, Pathology, Study Type, Population)
"""

import logging
import re
from typing import Dict, List, Optional, Set

from src.models import Paper

logger = logging.getLogger(__name__)


class PaperClassifier:
    """논문 자동 분류기"""

    # Modality 패턴
    MODALITY_PATTERNS: Dict[str, List[str]] = {
        "DBT": [
            r"\bDBT\b",
            r"tomosynthesis",
            r"3D\s+mammograph",
            r"three-?dimensional\s+mammograph",
            r"breast\s+tomosynthesis",
        ],
        "FFDM": [
            r"\bFFDM\b",
            r"full[- ]?field\s+digital",
            r"2D\s+mammograph",
            r"digital\s+mammograph(?!.*tomosynthesis)",
            r"two-?dimensional\s+mammograph",
        ],
        "CEM": [
            r"\bCEM\b",
            r"contrast[- ]?enhanced\s+mammograph",
            r"CESM",
            r"contrast\s+mammograph",
        ],
        "SM": [
            r"synthetic\s+mammograph",
            r"synthesized\s+2D",
            r"synthetic\s+2D",
            r"s2D",
            r"SM\s+image",
        ],
        "MRI": [
            r"\bMRI\b",
            r"magnetic\s+resonance",
            r"breast\s+MR\b",
        ],
        "US": [
            r"\bUS\b(?!\s+(?:population|women|patient))",
            r"ultrasound",
            r"ultrasonograph",
            r"sonograph",
            r"ABUS",
            r"HHUS",
        ],
    }

    # Pathology 패턴
    PATHOLOGY_PATTERNS: Dict[str, List[str]] = {
        "mass": [
            r"\bmass\b",
            r"\bmasses\b",
            r"\btumor\b",
            r"\btumour\b",
            r"\bnodule\b",
            r"\blesion\b",
        ],
        "calcification": [
            r"calcification",
            r"microcalcification",
            r"\bcalc\b",
        ],
        "density": [
            r"breast\s+density",
            r"dense\s+breast",
            r"BI-?RADS.*density",
            r"mammographic\s+density",
            r"parenchymal\s+density",
        ],
        "distortion": [
            r"architectural\s+distortion",
            r"\bdistortion\b",
        ],
        "asymmetry": [
            r"asymmetr",
            r"focal\s+asymmetry",
            r"developing\s+asymmetry",
            r"global\s+asymmetry",
        ],
    }

    # Study Type 패턴
    STUDY_TYPE_PATTERNS: Dict[str, List[str]] = {
        "prospective": [
            r"prospective\s+(?:study|cohort|trial)",
            r"prospectively\s+(?:collected|enrolled)",
        ],
        "retrospective": [
            r"retrospective\s+(?:study|cohort|analysis|review)",
            r"retrospectively\s+(?:reviewed|analyzed|collected)",
        ],
        "meta-analysis": [
            r"meta[- ]?analysis",
            r"systematic\s+review\s+and\s+meta",
        ],
        "review": [
            r"systematic\s+review",
            r"literature\s+review",
            r"narrative\s+review",
            r"scoping\s+review",
        ],
        "rct": [
            r"randomized",
            r"randomised",
            r"\bRCT\b",
            r"random\s+allocation",
            r"clinical\s+trial",
        ],
        "case-control": [
            r"case[- ]?control",
        ],
        "cohort": [
            r"cohort\s+study",
            r"longitudinal\s+study",
        ],
    }

    # Population 패턴
    POPULATION_PATTERNS: Dict[str, List[str]] = {
        "Asian": [
            r"\bKorean\b",
            r"\bJapanese\b",
            r"\bChinese\b",
            r"\bAsian\b",
            r"\bTaiwan",
            r"\bSingapore\b",
            r"\bHong\s+Kong\b",
            r"\bVietnam",
            r"\bThailand\b",
            r"\bMalaysia\b",
            r"\bIndia\b",
        ],
        "Western": [
            r"\bAmerican\b",
            r"\bEuropean\b",
            r"\bCaucasian\b",
            r"\bUnited\s+States\b",
            r"\bUSA\b",
            r"\bU\.S\.\b",
            r"\bUK\b",
            r"\bBritish\b",
            r"\bGerman\b",
            r"\bFrench\b",
            r"\bItalian\b",
            r"\bSpanish\b",
            r"\bDutch\b",
            r"\bSwedish\b",
            r"\bNorwegian\b",
            r"\bDanish\b",
            r"\bCanadian\b",
            r"\bAustralian\b",
        ],
    }

    def __init__(self):
        # 패턴 컴파일
        self._compiled_patterns: Dict[str, Dict[str, re.Pattern]] = {}

        for category, patterns in [
            ("modality", self.MODALITY_PATTERNS),
            ("pathology", self.PATHOLOGY_PATTERNS),
            ("study_type", self.STUDY_TYPE_PATTERNS),
            ("population", self.POPULATION_PATTERNS),
        ]:
            self._compiled_patterns[category] = {}
            for name, pattern_list in patterns.items():
                combined = "|".join(f"({p})" for p in pattern_list)
                self._compiled_patterns[category][name] = re.compile(
                    combined, re.IGNORECASE
                )

    def _get_text(self, paper: Paper) -> str:
        """검색할 텍스트 추출"""
        return f"{paper.title} {paper.abstract}"

    def classify_modality(self, paper: Paper) -> List[str]:
        """Modality 분류"""
        text = self._get_text(paper)
        modalities = []

        for modality, pattern in self._compiled_patterns["modality"].items():
            if pattern.search(text):
                modalities.append(modality)

        return modalities

    def classify_pathology(self, paper: Paper) -> List[str]:
        """Pathology 분류"""
        text = self._get_text(paper)
        pathologies = []

        for pathology, pattern in self._compiled_patterns["pathology"].items():
            if pattern.search(text):
                pathologies.append(pathology)

        return pathologies

    def classify_study_type(self, paper: Paper) -> Optional[str]:
        """Study Type 분류 (하나만)"""
        text = self._get_text(paper)

        # 우선순위: meta-analysis > rct > prospective > retrospective > others
        priority = [
            "meta-analysis",
            "rct",
            "prospective",
            "retrospective",
            "review",
            "case-control",
            "cohort",
        ]

        for study_type in priority:
            pattern = self._compiled_patterns["study_type"].get(study_type)
            if pattern and pattern.search(text):
                return study_type

        return None

    def classify_population(self, paper: Paper) -> Optional[str]:
        """Population 분류"""
        text = self._get_text(paper)

        asian_match = self._compiled_patterns["population"]["Asian"].search(text)
        western_match = self._compiled_patterns["population"]["Western"].search(text)

        if asian_match and western_match:
            return "Mixed"
        elif asian_match:
            return "Asian"
        elif western_match:
            return "Western"

        return None

    def classify(self, paper: Paper) -> Paper:
        """
        논문 전체 분류

        Args:
            paper: 원본 Paper

        Returns:
            분류된 Paper
        """
        paper.modality = self.classify_modality(paper)
        paper.pathology = self.classify_pathology(paper)
        paper.study_type = self.classify_study_type(paper)
        paper.population = self.classify_population(paper)

        return paper

    def classify_batch(self, papers: List[Paper], progress: bool = True) -> List[Paper]:
        """
        배치 분류

        Args:
            papers: 논문 리스트
            progress: 진행률 표시

        Returns:
            분류된 논문 리스트
        """
        from tqdm import tqdm

        classified = []
        iterator = tqdm(papers, desc="Classifying") if progress else papers

        for paper in iterator:
            classified.append(self.classify(paper))

        # 통계
        stats = self.get_classification_stats(classified)
        logger.info(self._format_stats(stats))

        return classified

    def get_classification_stats(self, papers: List[Paper]) -> Dict:
        """분류 통계"""
        stats = {
            "total": len(papers),
            "modality": {},
            "pathology": {},
            "study_type": {},
            "population": {},
        }

        for paper in papers:
            # Modality
            for m in paper.modality:
                stats["modality"][m] = stats["modality"].get(m, 0) + 1

            # Pathology
            for p in paper.pathology:
                stats["pathology"][p] = stats["pathology"].get(p, 0) + 1

            # Study Type
            if paper.study_type:
                stats["study_type"][paper.study_type] = (
                    stats["study_type"].get(paper.study_type, 0) + 1
                )

            # Population
            if paper.population:
                stats["population"][paper.population] = (
                    stats["population"].get(paper.population, 0) + 1
                )

        return stats

    def _format_stats(self, stats: Dict) -> str:
        """통계 포맷팅"""
        lines = [f"Classification stats ({stats['total']} papers):"]

        for category in ["modality", "pathology", "study_type", "population"]:
            items = stats.get(category, {})
            if items:
                sorted_items = sorted(items.items(), key=lambda x: -x[1])
                items_str = ", ".join(f"{k}={v}" for k, v in sorted_items[:5])
                lines.append(f"  {category}: {items_str}")

        return " | ".join(lines)
