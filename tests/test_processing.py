"""
MARIA-Mammo: Processing Tests
"""

import pytest

from src.models import Paper
from src.processing.classifier import PaperClassifier
from src.processing.cleaner import DataCleaner


class TestDataCleaner:
    """DataCleaner 테스트"""

    def setup_method(self):
        self.cleaner = DataCleaner()

    def test_clean_abstract_html_tags(self):
        """HTML 태그 제거 테스트"""
        text = "<b>Background</b>: This is a <i>test</i>."
        result = self.cleaner.clean_abstract(text)
        assert "<b>" not in result
        assert "<i>" not in result
        assert "Background: This is a test." == result

    def test_clean_abstract_entities(self):
        """HTML 엔티티 변환 테스트"""
        text = "A &amp; B &lt; C &gt; D"
        result = self.cleaner.clean_abstract(text)
        assert result == "A & B < C > D"

    def test_clean_title(self):
        """제목 정제 테스트"""
        title = "  Test Title.  "
        result = self.cleaner.clean_title(title)
        assert result == "Test Title"

    def test_normalize_authors(self):
        """저자 정규화 테스트"""
        authors = ["Kim  JH", "", "Lee SY"]
        result = self.cleaner.normalize_authors(authors)
        assert result == ["Kim JH", "Lee SY"]

    def test_is_retracted(self, sample_paper):
        """Retracted 감지 테스트"""
        sample_paper.title = "[Retracted] Original Title"
        assert self.cleaner.is_retracted(sample_paper)

        sample_paper.title = "Normal Title"
        sample_paper.publication_types = ["Retracted Publication"]
        assert self.cleaner.is_retracted(sample_paper)


class TestPaperClassifier:
    """PaperClassifier 테스트"""

    def setup_method(self):
        self.classifier = PaperClassifier()

    def test_classify_modality_dbt(self, sample_paper):
        """DBT 분류 테스트"""
        sample_paper.title = "Digital Breast Tomosynthesis Study"
        sample_paper.abstract = "We evaluated DBT performance..."

        result = self.classifier.classify_modality(sample_paper)
        assert "DBT" in result

    def test_classify_modality_ffdm(self, sample_paper):
        """FFDM 분류 테스트"""
        sample_paper.title = "Full-field digital mammography"
        sample_paper.abstract = ""

        result = self.classifier.classify_modality(sample_paper)
        assert "FFDM" in result

    def test_classify_pathology_calcification(self, sample_paper):
        """석회화 분류 테스트"""
        sample_paper.title = "Microcalcification detection"
        sample_paper.abstract = ""

        result = self.classifier.classify_pathology(sample_paper)
        assert "calcification" in result

    def test_classify_study_type_retrospective(self, sample_paper):
        """연구 유형 분류 테스트"""
        sample_paper.abstract = "This retrospective study included 500 patients..."

        result = self.classifier.classify_study_type(sample_paper)
        assert result == "retrospective"

    def test_classify_study_type_meta_analysis(self, sample_paper):
        """메타분석 분류 테스트"""
        sample_paper.title = "A meta-analysis of DBT studies"

        result = self.classifier.classify_study_type(sample_paper)
        assert result == "meta-analysis"

    def test_classify_population_asian(self, sample_paper):
        """인구 분류 테스트 - Asian"""
        sample_paper.abstract = "We studied Korean women..."

        result = self.classifier.classify_population(sample_paper)
        assert result == "Asian"

    def test_classify_population_western(self, sample_paper):
        """인구 분류 테스트 - Western"""
        sample_paper.abstract = "American women were enrolled..."

        result = self.classifier.classify_population(sample_paper)
        assert result == "Western"

    def test_classify_full(self, sample_paper):
        """전체 분류 테스트"""
        result = self.classifier.classify(sample_paper)

        assert isinstance(result.modality, list)
        assert isinstance(result.pathology, list)
