"""
MARIA-Mammo: Test Fixtures
"""

import pytest

from src.models import Paper, QueryFilters, SearchQuery


@pytest.fixture
def sample_paper():
    """샘플 논문"""
    return Paper(
        pmid="12345678",
        doi="10.1000/example",
        title="Digital Breast Tomosynthesis vs Full-Field Digital Mammography for Detection of Microcalcifications",
        authors=["Kim JH", "Lee SY", "Park HJ"],
        journal="Radiology",
        journal_abbrev="Radiology",
        year=2024,
        month=6,
        abstract="Purpose: To compare the diagnostic performance of digital breast tomosynthesis (DBT) and full-field digital mammography (FFDM) for microcalcification detection. Materials and Methods: Retrospective study of 500 women. Results: DBT showed sensitivity of 92% vs 85% for FFDM (p<0.01). Conclusion: DBT is superior for microcalcification detection.",
        mesh_terms=["Mammography", "Breast Neoplasms", "Calcinosis"],
        keywords=["DBT", "FFDM", "microcalcification"],
        publication_types=["Journal Article", "Comparative Study"],
        modality=["DBT", "FFDM"],
        pathology=["calcification"],
        study_type="retrospective",
        population="Asian",
        citation_count=25,
        journal_if=19.7,
        pmc_id="PMC123456",
    )


@pytest.fixture
def sample_papers():
    """샘플 논문 리스트"""
    return [
        Paper(
            pmid="11111111",
            title="DBT screening outcomes",
            journal="Radiology",
            year=2024,
            abstract="Prospective study on DBT screening...",
            authors=["Author A"],
            modality=["DBT"],
            study_type="prospective",
        ),
        Paper(
            pmid="22222222",
            title="FFDM and breast density",
            journal="European Radiology",
            year=2023,
            abstract="Breast density affects FFDM sensitivity...",
            authors=["Author B"],
            modality=["FFDM"],
            pathology=["density"],
        ),
        Paper(
            pmid="33333333",
            title="Contrast enhanced mammography",
            journal="AJR",
            year=2022,
            abstract="CEM improves detection of occult cancers...",
            authors=["Author C"],
            modality=["CEM"],
            pathology=["mass"],
        ),
    ]


@pytest.fixture
def sample_query():
    """샘플 검색 쿼리"""
    return SearchQuery(
        original_query="DBT vs FFDM for microcalcification",
        keywords=["dbt", "ffdm", "microcalcification"],
        mesh_terms=["Mammography", "Calcinosis"],
        filters=QueryFilters(modality=["DBT", "FFDM"]),
        intent="comparison study; calcification detection",
    )
