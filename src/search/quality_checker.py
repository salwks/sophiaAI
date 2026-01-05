"""
검색 품질 체크 모듈
RAG Hallucination 방지를 위한 검색 결과 관련성 검증
"""
from typing import List, Tuple
from src.models import SearchResult
import re


class SearchQualityChecker:
    """검색 결과 품질 평가기"""

    def __init__(self, min_confidence: float = 0.6):
        """
        Args:
            min_confidence: 최소 신뢰도 임계값 (0.0-1.0)
        """
        self.min_confidence = min_confidence

    def check_relevance(
        self,
        query: str,
        results: List[SearchResult],
        top_k: int = 3
    ) -> Tuple[float, bool, str]:
        """
        검색 결과의 관련성 평가

        Args:
            query: 사용자 질문
            results: 검색 결과 리스트
            top_k: 평가할 상위 결과 수

        Returns:
            (confidence_score, is_high_quality, reason)
            - confidence_score: 0.0-1.0 신뢰도 점수
            - is_high_quality: 고품질 여부
            - reason: 평가 이유
        """
        if not results:
            return 0.0, False, "검색 결과 없음"

        query_lower = query.lower()
        top_results = results[:top_k]

        # 평가 기준들
        scores = []
        reasons = []

        # 1. BI-RADS 특정 항목 질문 체크
        birads_specific_match = self._check_birads_specific(query_lower, top_results)
        if birads_specific_match:
            score, reason = birads_specific_match
            scores.append(score)
            reasons.append(reason)

        # 2. 키워드 매칭 체크
        keyword_score, keyword_reason = self._check_keywords(query_lower, top_results)
        scores.append(keyword_score)
        reasons.append(keyword_reason)

        # 3. 검색 점수 체크
        score_check, score_reason = self._check_scores(top_results)
        scores.append(score_check)
        reasons.append(score_reason)

        # 최종 점수 (평균)
        avg_score = sum(scores) / len(scores) if scores else 0.0

        # 고품질 판정
        is_high_quality = avg_score >= self.min_confidence

        # 이유 정리
        final_reason = " | ".join(reasons)

        return avg_score, is_high_quality, final_reason

    def _check_birads_specific(
        self,
        query: str,
        results: List[SearchResult]
    ) -> Tuple[float, str] | None:
        """BI-RADS 특정 항목 질문인지 체크"""

        # Category 0-6
        cat_pattern = r'category\s*[0-6]|cat\s*[0-6]|카테고리\s*[0-6]'
        if re.search(cat_pattern, query):
            cat_num = re.search(r'[0-6]', query).group()
            expected_pmid = f"CAT{cat_num}"

            for result in results:
                if expected_pmid in result.paper.pmid.upper():
                    return 1.0, f"CAT{cat_num} 정확히 매칭"

            return 0.3, f"CAT{cat_num} 질문이지만 해당 문서 검색 안 됨"

        # Mass descriptors
        mass_terms = {
            'margin': 'CHUNK_MARGIN',
            'shape': 'CHUNK_SHAPE',
            'density': 'CHUNK_DENSITY',
        }

        for term, expected_chunk in mass_terms.items():
            if term in query:
                for result in results:
                    if expected_chunk in result.paper.pmid:
                        return 1.0, f"{term} 청크 정확히 매칭"

                return 0.3, f"{term} 질문이지만 해당 청크 검색 안 됨"

        # Calcification descriptors
        calc_terms = {
            'amorphous': 'CHUNK_AMORPHOUS',
            'fine linear': 'CHUNK_FINE_LINEAR',
            'pleomorphic': 'CHUNK_FINE_PLEOMORPHIC',
            'skin': 'CHUNK_SKIN',
            'vascular': 'CHUNK_VASCULAR',
            'dystrophic': 'CHUNK_COARSE',
            'coarse': 'CHUNK_COARSE',
        }

        for term, expected_chunk in calc_terms.items():
            if term in query:
                for result in results:
                    if expected_chunk in result.paper.pmid:
                        return 1.0, f"{term} 청크 정확히 매칭"

                return 0.3, f"{term} 질문이지만 해당 청크 검색 안 됨"

        # Distribution
        dist_terms = {
            'regional': 'CHUNK_REGIONAL',
            'diffuse': 'CHUNK_DIFFUSE',
            'linear': 'CHUNK_LINEAR',
            'segmental': 'CHUNK_SEGMENTAL',
            'grouped': 'CHUNK_GROUPED',
        }

        for term, expected_chunk in dist_terms.items():
            if term in query and 'distribution' in query:
                for result in results:
                    if expected_chunk in result.paper.pmid:
                        return 1.0, f"{term} distribution 청크 정확히 매칭"

                return 0.3, f"{term} distribution 질문이지만 해당 청크 검색 안 됨"

        return None

    def _check_keywords(
        self,
        query: str,
        results: List[SearchResult]
    ) -> Tuple[float, str]:
        """키워드 매칭 체크"""

        # 쿼리에서 중요 키워드 추출 (불용어 제외)
        stopwords = {'what', 'is', 'are', 'the', 'a', 'an', 'in', 'of', 'for', 'to',
                     'how', 'when', 'where', 'why', 'does', 'mean', 'means',
                     '무엇', '이', '가', '은', '는', '을', '를', '에', '의', '과',
                     '뭐', '어떤', '어떻게', '언제', '왜'}

        query_words = set(re.findall(r'\b\w+\b', query.lower()))
        important_words = query_words - stopwords

        if not important_words:
            return 0.5, "키워드 추출 불가"

        # Top 결과들의 제목/내용에서 키워드 매칭률
        total_matches = 0
        total_possible = len(important_words) * len(results)

        for result in results:
            title = result.paper.title.lower() if result.paper.title else ""
            pmid = result.paper.pmid.lower()

            for word in important_words:
                if word in title or word in pmid:
                    total_matches += 1

        match_rate = total_matches / total_possible if total_possible > 0 else 0.0

        if match_rate >= 0.5:
            return 0.9, f"키워드 매칭률 {match_rate:.1%}"
        elif match_rate >= 0.3:
            return 0.6, f"키워드 매칭률 {match_rate:.1%} (보통)"
        else:
            return 0.3, f"키워드 매칭률 {match_rate:.1%} (낮음)"

    def _check_scores(
        self,
        results: List[SearchResult]
    ) -> Tuple[float, str]:
        """검색 점수 체크"""

        if not results:
            return 0.0, "결과 없음"

        top_score = results[0].score

        if top_score >= 0.8:
            return 1.0, f"1위 점수 {top_score:.1%} (매우 높음)"
        elif top_score >= 0.6:
            return 0.7, f"1위 점수 {top_score:.1%} (높음)"
        elif top_score >= 0.4:
            return 0.5, f"1위 점수 {top_score:.1%} (보통)"
        else:
            return 0.3, f"1위 점수 {top_score:.1%} (낮음)"

    def get_quality_level(self, confidence: float) -> str:
        """신뢰도에 따른 품질 레벨 반환"""
        if confidence >= 0.8:
            return "high"  # 고품질: LLM 답변 + 문서 전문
        elif confidence >= 0.5:
            return "medium"  # 중품질: 문서 전문만
        else:
            return "low"  # 저품질: 검색 실패 메시지


def main():
    """테스트"""
    from src.models import Paper, SearchResult

    checker = SearchQualityChecker(min_confidence=0.6)

    # 테스트 케이스 1: 정확한 매칭
    query1 = "What is BI-RADS category 3?"
    results1 = [
        SearchResult(
            paper=Paper(pmid="BIRADS_2025_SECTION_V_CAT3", title="BI-RADS Category 3"),
            score=0.9,
            rank=1
        )
    ]

    score, is_high, reason = checker.check_relevance(query1, results1)
    print(f"Query: {query1}")
    print(f"Score: {score:.2f}, High Quality: {is_high}, Reason: {reason}\n")

    # 테스트 케이스 2: 잘못된 매칭
    query2 = "What are the 4 margin classifications?"
    results2 = [
        SearchResult(
            paper=Paper(pmid="BIRADS_2025_SECTION_V_CAT1", title="BI-RADS Category 1"),
            score=0.5,
            rank=1
        )
    ]

    score, is_high, reason = checker.check_relevance(query2, results2)
    print(f"Query: {query2}")
    print(f"Score: {score:.2f}, High Quality: {is_high}, Reason: {reason}\n")


if __name__ == "__main__":
    main()
