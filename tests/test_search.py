"""
MARIA-Mammo: Search Tests
"""

import pytest

from src.models import QueryFilters, SearchQuery
from src.search.query_parser import QueryParser


class TestQueryParser:
    """QueryParser 테스트"""

    def setup_method(self):
        self.parser = QueryParser()

    def test_parse_simple_query(self):
        """간단한 쿼리 파싱"""
        result = self.parser.parse("mammography screening")

        assert result.original_query == "mammography screening"
        assert "mammography" in result.keywords
        assert "screening" in result.keywords

    def test_parse_modality_dbt(self):
        """DBT 키워드 파싱"""
        result = self.parser.parse("DBT performance evaluation")

        assert "DBT" in result.filters.modality

    def test_parse_modality_tomosynthesis(self):
        """Tomosynthesis 키워드 파싱"""
        result = self.parser.parse("digital tomosynthesis study")

        assert "DBT" in result.filters.modality

    def test_parse_pathology(self):
        """Pathology 파싱"""
        result = self.parser.parse("microcalcification detection")

        assert "calcification" in result.filters.pathology

    def test_parse_study_type(self):
        """Study type 파싱"""
        result = self.parser.parse("meta-analysis of DBT studies")

        assert result.filters.study_type == "meta-analysis"

    def test_parse_population(self):
        """Population 파싱"""
        result = self.parser.parse("Korean women breast screening")

        assert result.filters.population == "Asian"

    def test_parse_year_since(self):
        """연도 파싱 - since"""
        result = self.parser.parse("studies since 2020")

        assert result.filters.year_min == 2020

    def test_parse_year_last_n(self):
        """연도 파싱 - last N years"""
        result = self.parser.parse("research in last 5 years")

        assert result.filters.year_min is not None
        assert result.filters.year_min >= 2019  # Approximate

    def test_parse_mesh_terms(self):
        """MeSH 용어 추출"""
        result = self.parser.parse("mammography breast cancer screening")

        assert "Mammography" in result.mesh_terms
        assert "Breast Neoplasms" in result.mesh_terms

    def test_parse_intent_comparison(self):
        """의도 추론 - 비교"""
        result = self.parser.parse("DBT vs FFDM comparison")

        assert "comparison" in result.intent.lower()

    def test_parse_intent_performance(self):
        """의도 추론 - 성능"""
        result = self.parser.parse("sensitivity specificity of mammography")

        assert "performance" in result.intent.lower() or "evaluation" in result.intent.lower()

    def test_parse_empty_query(self):
        """빈 쿼리 처리"""
        result = self.parser.parse("")

        assert result.original_query == ""
        assert result.keywords == []

    def test_extract_keywords_stopwords(self):
        """불용어 제거"""
        keywords = self.parser._extract_keywords("the study of mammography in the breast")

        assert "the" not in keywords
        assert "of" not in keywords
        assert "in" not in keywords
        assert "mammography" in keywords
        assert "breast" in keywords

    def test_has_filters(self):
        """필터 유무 확인"""
        result1 = self.parser.parse("general mammography query")
        result2 = self.parser.parse("DBT studies since 2020")

        # 첫 번째는 DBT 필터가 없어야 함
        assert result1.filters.modality is None or len(result1.filters.modality) == 0

        # 두 번째는 필터가 있어야 함
        assert result2.has_filters
