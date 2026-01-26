"""
50개 질문 RAG 시스템 테스트
"""

import sys
sys.path.insert(0, '/Users/sinjaeho/Documents/works/sophyAI/maria-mammo')

from src.indexing.database import DatabaseManager
from src.indexing.embedder import PaperEmbedder
from src.search.query_translator import QueryTranslator
import json
from datetime import datetime

# 50개 테스트 질문
QUESTIONS = [
    # 1. 필터 및 타겟 기술 심화 (15개)
    "유방촬영용 X선 관에서 고유 여과(Inherent Filtration)와 부가 여과(Added Filtration)를 조합하여 사용하는 물리적 목적은 무엇인가?",
    "유방촬영술에서 낮은 에너지의 광자를 걸러내어 환자의 피부 선량을 줄이는 필터의 역할을 기술적으로 설명하시오.",
    "2~5cm 두께의 얇은 유방 촬영 시 Molybdenum(Mo) 타겟과 Mo 필터 조합이 가장 효율적인 이유는?",
    "유방 두께가 5cm 이상이거나 치밀도가 높을 때, Mo 필터 대신 Rhodium(Rh) 필터를 사용하는 물리적 근거는 무엇인가?",
    "K-흡수단(K-edge) 필터링 기법이 유방촬영 영상의 대조도(Contrast) 향상에 기여하는 원리는 무엇인가?",
    "텅스텐(W) 타겟 시스템에서 은(Ag) 또는 알루미늄(Al) 필터를 조합하여 사용할 때의 장점은 무엇인가?",
    "조영증강 유방촬영술(CEM)의 고에너지(High-energy) 영상 획득 시 티타늄(Ti)이나 구리(Cu) 필터를 사용하는 기술적 이유는?",
    "CEM의 저에너지(Low-energy) 영상과 고에너지 영상 획득 시 각각 사용되는 전형적인 관전압(kVp)과 필터 재질은?",
    "디지털 유방 검출기의 특성을 고려할 때, 높은 에너지의 X선 빔과 적절한 필터 선택이 평균 유선 선량(MGD) 감소에 미치는 영향은?",
    "필터의 재질(Mo, Rh, Ag, Cu)에 따라 X선 빔의 반가층(HVL) 수치가 변화하는 양상을 기술하시오.",
    "필터 두께의 변화가 영상의 대조도 대 잡음비(CNR)와 신호 대 잡음비(SNR)에 미치는 상관관계는 무엇인가?",
    "자동 노출 제어(AEC) 시스템이 유방의 두께와 밀도를 감지하여 타겟/필터 조합을 결정하는 기술적 알고리즘의 원리는?",
    "유방 토모신테시스(DBT) 시스템에서 필터 선택이 투사 영상별 선량 분배와 최종 재구성 영상 품질에 미치는 영향은?",
    "필터 물질의 원자 번호가 특정 에너지 대역의 광자를 선택적으로 통과시키는 '창(Window)' 효과와 갖는 관계는?",
    "장비의 품질 관리(QC) 과정에서 필터의 결함이나 이탈 여부를 확인하기 위해 수행하는 물리적 테스트 항목은?",

    # 2. 영상 획득 및 물리적 원리 (10개)
    "공간 해상도(Spatial Resolution) 저하를 방지하기 위해 유방 압박(Compression) 시 고려해야 할 기하학적 요인은?",
    "확대 촬영(Magnification) 시 작은 초점(Small Focal Spot) 사용이 기하학적 흐림(Geometric Blurring)을 줄이는 원리는?",
    "유방 압박 장치가 산란선(Scattered Radiation) 감소와 해부학적 조직 겹침 제거에 기여하는 방식은?",
    "그리드(Grid) 사용 시 영상 대조도 향상과 그에 따른 방사선량 증가 사이의 기술적 트레이드오프(Trade-off)는?",
    "DBT 촬영 시 튜브의 이동 각도(Angular Range)가 영상의 Z축 해상도와 아티팩트 발생에 미치는 영향은?",
    "고대조도 병변(예: 금속 마커) 주변에서 발생하는 'Ghosting' 또는 'Shadowing' 아티팩트의 기술적 원인은?",
    "'For Processing' 원본 데이터가 'For Presentation' 전시용 데이터로 변환될 때 적용되는 영상 처리 알고리즘의 종류는?",
    "노이즈 감소(Noise Reduction) 알고리즘이 미세 석회화의 선명도(Sharpness)에 미칠 수 있는 부정적 영향은?",
    "유방촬영용 모니터의 휘도(Luminance) 응답 지침인 DICOM GSDF 표준이 판독 정확도에 미치는 영향은?",
    "판독실의 주변 조명(Ambient Light) 조건이 연조직 대조도 인지에 미치는 물리적 영향은 무엇인가?",

    # 3. BI-RADS 어휘 및 진단 지표 (10개)
    "BI-RADS 용어 체계에서 '가려진 경계(Obscured Margin)'를 판단하는 구체적인 백분율(%) 기준은 무엇인가?",
    "종괴가 지방 조직과 인접해 있을 때 경계가 명확하지 않다면 '가려진(Obscured)' 대신 어떤 용어를 사용해야 하는가?",
    "'구조적 왜곡(Architectural Distortion)'이 중심부 종괴 없이 방사형 선들로 나타나는 병리학적 의미는 무엇인가?",
    "선별 검사(Screening)에서 '비대칭(Asymmetry)' 소견이 발견되었을 때 진단 검사로 리콜하는 판단 기준은?",
    "'Focal Asymmetry'와 'Global Asymmetry'를 구분하는 면적 및 투사 방향(View)의 기준은?",
    "과거 영상과 비교하여 새로 나타나거나 커진 'Developing Asymmetry'가 갖는 높은 악성 예측도는?",
    "미세 석회화의 '선상 분지형(Fine Linear Branching)' 형태가 유선관 내 암종(DCIS)과 강력히 연관되는 이유는?",
    "석회화의 '구상(Grouped)' 분포를 정의하기 위한 면적 및 최소 개수 기준은 무엇인가?",
    "'구역(Segmental)' 분포의 석회화가 다중심성 유방암(Multifocal cancer)을 시사하는 기하학적 형태는?",
    "유방 치밀도 Category D(극도로 치밀) 환자의 판독문 기술 시 반드시 언급해야 하는 진단적 한계점은?",

    # 4. 임상 지침, 성과 분석 및 최신 트렌드 (15개)
    "고위험군 여성이 연령 30세부터 선별 검사를 시작해야 하는 구체적인 유전적/임상적 조건은?",
    "트랜스젠더 환자의 유방암 선별 검사 지침에서 호르몬 사용 기간이 갖는 중요성은?",
    "BI-RADS 0 단계를 부여한 후 최종 카테고리 판정을 위해 수행하는 추가 영상 검사의 종류는?",
    "'아마도 양성(Probably Benign, Category 3)' 판정을 내리기 위한 3대 전형적 영상 소견은?",
    "카테고리 3 병변의 6개월 추적 관찰 검사에서 안정성이 확인되었을 때의 다음 관리 지침은?",
    "유방 토모신테시스(DBT)가 기존 2D 촬영 대비 재검사율(Recall rate)을 낮추는 기술적 기전은?",
    "합성 2D(Synthetic 2D) 영상이 환자의 방사선 피폭량을 줄이면서도 2D 촬영을 대체할 수 있는 근거는?",
    "CEM 검사 시 조영제 주입 후 2분에서 10분 사이에 촬영을 완료해야 하는 생리학적 이유는?",
    "조영제 알레르기나 신장 기능 저하 환자에게 CEM 대신 권고되는 대안 영상 검사는?",
    "조직 검사 후 '영상의학-병리학적 불일치(Discordance)'가 발생했을 때 수행해야 하는 추가 절차는?",
    "의료 성과 감사(Medical Outcomes Audit)에서 암 발견율(CDR)과 위음성율을 추적 관리해야 하는 법적 근거는?",
    "남성 유방의 '불꽃 모양(Flame-shaped)' 밀도 소견이 여성형 유방증을 시사하는 임상적 특징은?",
    "유방 보형물 삽입 환자의 선별 검사 시 보형물에 의해 가려지는 조직을 최소화하기 위한 촬영 기법은?",
    "RAG 시스템에서 BI-RADS 가이드라인의 복잡한 계층 구조를 보존하기 위한 벡터 데이터베이스 설계 전략은?",
    "인공지능 기반 판독 보조 소프트웨어가 BI-RADS 4A 이상의 마킹을 했을 때 리포트 자동 생성 로직에 포함되어야 하는 필수 요소는?",
]

# 질문 카테고리
CATEGORIES = {
    "filter_target": (0, 15, "필터 및 타겟 기술"),
    "acquisition_physics": (15, 25, "영상 획득 및 물리적 원리"),
    "birads_lexicon": (25, 35, "BI-RADS 어휘 및 진단"),
    "clinical_guidelines": (35, 50, "임상 지침 및 트렌드"),
}

def is_guideline(pmid: str) -> bool:
    """가이드라인 문서 여부 확인 (ACR, BIRADS, PHYSICS, CLINICAL)"""
    prefixes = ("ACR_", "BIRADS_", "PHYSICS_", "CLINICAL_")
    return pmid.startswith(prefixes)


def test_all_questions():
    """50개 질문 테스트 실행"""
    print("=" * 80)
    print("50개 질문 RAG 시스템 테스트")
    print("=" * 80)

    # 초기화
    db = DatabaseManager()
    embedder = PaperEmbedder()
    translator = QueryTranslator()

    results = []
    acr_hit_count = 0
    total_acr_in_top5 = 0

    for i, question in enumerate(QUESTIONS):
        print(f"\n[{i+1}/50] {question[:60]}...")

        # 번역
        translated = translator.translate(question)
        print(f"   번역: {translated[:80]}")

        # 임베딩 및 검색
        query_emb = embedder.embed_query(translated)
        search_results = db.search_vector(query_emb, k=10)

        # 결과 분석
        top5_pmids = [pmid for pmid, _ in search_results[:5]]
        acr_in_top5 = [pmid for pmid in top5_pmids if is_guideline(pmid)]

        result = {
            "question_id": i + 1,
            "question": question,
            "translated": translated,
            "top5_pmids": top5_pmids,
            "acr_in_top5": acr_in_top5,
            "acr_count": len(acr_in_top5),
            "top1_is_acr": len(top5_pmids) > 0 and is_guideline(top5_pmids[0]),
        }
        results.append(result)

        if result["acr_count"] > 0:
            acr_hit_count += 1
            total_acr_in_top5 += result["acr_count"]

        # Top 5 출력
        print(f"   Top 5: {top5_pmids}")
        if acr_in_top5:
            print(f"   ✓ 가이드라인: {acr_in_top5}")
        else:
            print(f"   ✗ 가이드라인 없음")

    # 카테고리별 통계
    print("\n" + "=" * 80)
    print("카테고리별 결과")
    print("=" * 80)

    category_stats = {}
    for cat_name, (start, end, label) in CATEGORIES.items():
        cat_results = results[start:end]
        acr_hits = sum(1 for r in cat_results if r["acr_count"] > 0)
        top1_acr = sum(1 for r in cat_results if r["top1_is_acr"])
        total_acr = sum(r["acr_count"] for r in cat_results)

        category_stats[cat_name] = {
            "label": label,
            "total": end - start,
            "acr_hits": acr_hits,
            "top1_acr": top1_acr,
            "total_acr_in_top5": total_acr,
        }

        print(f"\n{label} ({end-start}개 질문):")
        print(f"  - Top5에 가이드라인 포함: {acr_hits}/{end-start} ({100*acr_hits/(end-start):.1f}%)")
        print(f"  - Top1이 가이드라인: {top1_acr}/{end-start} ({100*top1_acr/(end-start):.1f}%)")
        print(f"  - Top5 내 총 가이드라인 수: {total_acr}")

    # 전체 통계
    print("\n" + "=" * 80)
    print("전체 결과 요약")
    print("=" * 80)
    print(f"총 질문 수: 50")
    print(f"Top5에 가이드라인 포함: {acr_hit_count}/50 ({100*acr_hit_count/50:.1f}%)")
    top1_acr_total = sum(1 for r in results if r["top1_is_acr"])
    print(f"Top1이 가이드라인: {top1_acr_total}/50 ({100*top1_acr_total/50:.1f}%)")
    print(f"Top5 내 총 가이드라인 수: {total_acr_in_top5}")

    # 결과 저장
    output = {
        "test_date": datetime.now().isoformat(),
        "total_questions": 50,
        "summary": {
            "acr_in_top5_count": acr_hit_count,
            "acr_in_top5_rate": acr_hit_count / 50,
            "top1_acr_count": top1_acr_total,
            "top1_acr_rate": top1_acr_total / 50,
            "total_acr_in_top5": total_acr_in_top5,
        },
        "category_stats": category_stats,
        "results": results,
    }

    with open("/Users/sinjaeho/Documents/works/sophyAI/maria-mammo/tests/test_50_results.json", "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"\n결과 저장: tests/test_50_results.json")

    return output

if __name__ == "__main__":
    test_all_questions()
