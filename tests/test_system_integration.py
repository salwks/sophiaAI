"""
System Integration Test
========================
각 Phase별 동작 검증 및 Knowledge 연결 테스트
"""

import sys
import json
from pathlib import Path

# 프로젝트 루트 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.reasoning.physics_triage import (
    PhysicsClassifier, SolverRouter, FrameworkInjector, PostVerifier,
    PhysicsDomain, ClassificationResult
)
from src.knowledge.manager import KnowledgeManager


def test_classifier():
    """Phase별 분류 테스트"""
    print("\n" + "="*60)
    print("1. PhysicsClassifier 테스트")
    print("="*60)

    classifier = PhysicsClassifier()

    test_cases = [
        # Phase 1: SNR + 전자노이즈
        ("선량 50% 감소 시 PCD의 SNR 감소율은?", PhysicsDomain.PHASE1_SNR),
        ("전자노이즈 30%일 때 EID vs PCD 비교", PhysicsDomain.PHASE1_SNR),

        # Phase 2: 스펙트럼
        ("에너지 가중치 적용 시 대조도 향상은?", PhysicsDomain.PHASE2_SPECTRAL),
        ("스펙트럴 영상의 CNR 개선", PhysicsDomain.PHASE2_SPECTRAL),

        # Phase 3: DQE
        ("저선량에서 DQE 변화율은?", PhysicsDomain.PHASE3_DQE),
        ("DQE의 선량 의존성 분석", PhysicsDomain.PHASE3_DQE),

        # Phase 4: MTF
        ("MTF at Nyquist frequency 비교", PhysicsDomain.PHASE4_MTF),
        ("공간 해상도와 픽셀 피치 관계", PhysicsDomain.PHASE4_MTF),

        # Phase 4B: 깊이 분해능
        ("토모합성 깊이 분해능은?", PhysicsDomain.PHASE4B_DEPTH),
        ("50도 각도에서 slice thickness는?", PhysicsDomain.PHASE4B_DEPTH),

        # Phase 5: 토모 영상품질
        ("25 projection DBT에서 dose-split DQE 저하", PhysicsDomain.PHASE5_TOMO_IQ),
        ("토모합성 detectability index 계산", PhysicsDomain.PHASE5_TOMO_IQ),
    ]

    results = []
    for query, expected_domain in test_cases:
        result = classifier.classify(query)
        match = "✅" if result.primary_domain == expected_domain else "❌"
        results.append({
            "query": query[:40] + "..." if len(query) > 40 else query,
            "expected": expected_domain.value,
            "actual": result.primary_domain.value,
            "confidence": f"{result.confidence:.2f}",
            "match": match
        })
        print(f"{match} [{result.confidence:.2f}] {query[:50]}")
        print(f"   Expected: {expected_domain.value}, Got: {result.primary_domain.value}")

    success_count = sum(1 for r in results if r["match"] == "✅")
    print(f"\n분류 정확도: {success_count}/{len(results)} ({100*success_count/len(results):.0f}%)")

    return results


def test_solver_router():
    """SolverRouter 각 Phase 테스트"""
    print("\n" + "="*60)
    print("2. SolverRouter 테스트")
    print("="*60)

    router = SolverRouter()

    test_cases = [
        # Phase 1
        {
            "domain": PhysicsDomain.PHASE1_SNR,
            "params": {"dose_ratio": 0.5, "electronic_noise_fraction": 0.3},
            "expected_label": "PCD SNR advantage"
        },
        # Phase 3
        {
            "domain": PhysicsDomain.PHASE3_DQE,
            "params": {"dose_ratio": 0.5, "electronic_noise_fraction": 0.3, "eta_abs": 0.85},
            "expected_label": "DQE"
        },
        # Phase 4
        {
            "domain": PhysicsDomain.PHASE4_MTF,
            "params": {"pixel_pitch_mm": 0.1},
            "expected_label": "MTF"
        },
        # Phase 4B
        {
            "domain": PhysicsDomain.PHASE4B_DEPTH,
            "params": {"angular_range_deg": 50, "depth_resolution_constant": 0.5},
            "expected_label": "depth"
        },
        # Phase 5
        {
            "domain": PhysicsDomain.PHASE5_TOMO_IQ,
            "params": {"n_projections": 25, "electronic_noise_fraction": 0.3, "eta_abs": 0.85},
            "expected_label": "SNR"
        },
        # Phase 2 (미구현 예상)
        {
            "domain": PhysicsDomain.PHASE2_SPECTRAL,
            "params": {},
            "expected_label": None  # 실패 예상
        },
    ]

    results = []
    for case in test_cases:
        # ClassificationResult 생성
        classification = ClassificationResult(
            primary_domain=case["domain"],
            confidence=0.9,
            keyword_path=case["domain"],
            semantic_path=case["domain"],
            paths_agree=True,
            extracted_params=case["params"]
        )

        try:
            result = router.route_and_solve(classification)
            if result:
                status = "✅"
                value = f"{result.primary_value:.4f}"
                label = result.primary_label
            else:
                status = "⚠️"
                value = "None"
                label = "No result"
        except Exception as e:
            status = "❌"
            value = "Error"
            label = str(e)[:50]

        results.append({
            "domain": case["domain"].value,
            "status": status,
            "value": value,
            "label": label
        })

        print(f"{status} {case['domain'].value}: {value} ({label})")

    return results


def test_knowledge_manager():
    """Knowledge Manager 테스트"""
    print("\n" + "="*60)
    print("3. KnowledgeManager 테스트")
    print("="*60)

    km = KnowledgeManager()

    test_queries = [
        ("PCD SNR 전자노이즈", ["pcd_low_dose_snr"]),
        ("DQE 선량 의존성", ["pcd_dqe_nps", "detector_physics"]),
        ("MTF 공간해상도", ["pcd_mtf_resolution", "spatial_resolution_mtf"]),
        ("토모합성 dose-split", ["dbt_image_quality"]),
        ("MGD 평균유선선량", ["mgd_dosimetry"]),
        ("유방 밀도 BI-RADS", ["breast_density", "birads_categories"]),
        ("압박력 compression", ["compression_physics"]),
        ("포지셔닝 MLO CC", ["breast_positioning"]),
        ("조영증강 CEM", ["contrast_enhanced_mammography"]),
        ("QC 품질관리 ACR", ["qc_protocols"]),
    ]

    results = []
    for query, expected_modules in test_queries:
        modules = km.get_relevant_knowledge(query, max_modules=3)
        found_ids = [m.get("id", "unknown") for m in modules]

        # 예상 모듈 중 하나라도 포함되면 성공
        match = any(exp in found_ids for exp in expected_modules)
        status = "✅" if match else "❌"

        results.append({
            "query": query,
            "expected": expected_modules,
            "found": found_ids,
            "match": status
        })

        print(f"{status} '{query}'")
        print(f"   Expected: {expected_modules}")
        print(f"   Found: {found_ids}")

    success_count = sum(1 for r in results if r["match"] == "✅")
    print(f"\nKnowledge 검색 정확도: {success_count}/{len(results)} ({100*success_count/len(results):.0f}%)")

    return results


def test_framework_injector():
    """FrameworkInjector 테스트"""
    print("\n" + "="*60)
    print("4. FrameworkInjector 테스트")
    print("="*60)

    from src.reasoning.physics_triage import SolverResult

    injector = FrameworkInjector()

    # 샘플 SolverResult 생성
    sample_result = SolverResult(
        domain=PhysicsDomain.PHASE1_SNR,
        primary_value=1.10,
        primary_label="PCD SNR advantage",
        all_values={"snr_pcd": 0.707, "snr_eid": 0.642},
        formula_used="SNR_ratio = √[(1-f_e)/(1-f_e×D')]",
        physical_principle="전자노이즈 제거로 PCD가 저선량에서 SNR 우위",
        parameters={"dose_ratio": 0.5, "f_e": 0.3},
        derivation_summary=["Step 1: 전자노이즈 비율 확인", "Step 2: SNR 공식 적용"]
    )

    try:
        framework = injector.generate_framework(PhysicsDomain.PHASE1_SNR, sample_result)
        print("✅ FrameworkPrompt 생성 성공")
        print(f"   - physics_principle: {framework.physics_principle[:50]}...")
        print(f"   - formula_guide: {framework.formula_guide[:50]}...")
        print(f"   - parameter_values: {framework.parameter_values[:50]}...")
        return {"status": "✅", "message": "Framework generated successfully"}
    except Exception as e:
        print(f"❌ FrameworkPrompt 생성 실패: {e}")
        return {"status": "❌", "message": str(e)}


def test_knowledge_module_completeness():
    """Knowledge 모듈 완전성 검사"""
    print("\n" + "="*60)
    print("5. Knowledge 모듈 완전성 검사")
    print("="*60)

    knowledge_dir = project_root / "data" / "knowledge" / "physics"

    required_fields = ["id", "keywords", "sources"]
    recommended_fields = ["content", "clinical_relevance"]

    results = []
    for json_file in sorted(knowledge_dir.glob("*.json")):
        try:
            with open(json_file, "r", encoding="utf-8") as f:
                data = json.load(f)

            # 필수 필드 확인
            missing_required = [f for f in required_fields if f not in data]
            missing_recommended = [f for f in recommended_fields if f not in data]

            # sources 배열 확인
            sources = data.get("sources", [])
            has_pmid = any("pmid" in s or "pmc" in s for s in sources) if sources else False

            if missing_required:
                status = "❌"
                note = f"Missing: {missing_required}"
            elif not sources:
                status = "⚠️"
                note = "No sources array"
            elif not has_pmid:
                status = "⚠️"
                note = "No PMID/PMC in sources"
            else:
                status = "✅"
                note = f"{len(sources)} sources"

            results.append({
                "file": json_file.stem,
                "status": status,
                "sources_count": len(sources),
                "has_pmid": has_pmid,
                "note": note
            })

            print(f"{status} {json_file.stem}: {note}")

        except Exception as e:
            results.append({
                "file": json_file.stem,
                "status": "❌",
                "note": f"Error: {e}"
            })
            print(f"❌ {json_file.stem}: Error - {e}")

    complete_count = sum(1 for r in results if r["status"] == "✅")
    print(f"\n완전한 모듈: {complete_count}/{len(results)}")

    return results


def test_end_to_end_flow():
    """End-to-End 흐름 테스트 (LLM 제외)"""
    print("\n" + "="*60)
    print("6. End-to-End 흐름 테스트 (LLM 제외)")
    print("="*60)

    classifier = PhysicsClassifier()
    router = SolverRouter()
    injector = FrameworkInjector()
    km = KnowledgeManager()

    test_query = "PCD 맘모그래피에서 선량을 50% 줄였을 때 SNR 감소율은? 전자노이즈 비율 30%"

    print(f"Query: {test_query}\n")

    # Step 1: 분류
    print("Step 1: Classification")
    classification = classifier.classify(test_query)
    print(f"   Domain: {classification.primary_domain.value}")
    print(f"   Confidence: {classification.confidence:.2f}")
    print(f"   Params: {classification.extracted_params}")

    # Step 2: Solver
    print("\nStep 2: Solver")
    solver_result = router.route_and_solve(classification)
    if solver_result:
        print(f"   Primary Value: {solver_result.primary_value:.4f}")
        print(f"   Label: {solver_result.primary_label}")
        print(f"   Formula: {solver_result.formula_used[:60]}...")
    else:
        print("   ❌ Solver failed")
        return {"status": "❌", "step": "solver"}

    # Step 3: Framework
    print("\nStep 3: Framework Injection")
    framework = injector.generate_framework(classification.primary_domain, solver_result)
    print(f"   Principle: {framework.physics_principle[:60]}...")

    # Step 4: Knowledge
    print("\nStep 4: Knowledge Retrieval")
    modules = km.get_relevant_knowledge(test_query, max_modules=2)
    print(f"   Modules: {[m.get('id') for m in modules]}")

    # Step 5: Context 생성
    print("\nStep 5: LLM Context Assembly")
    context_parts = []

    # Framework 추가
    framework_text = injector.format_as_prompt(framework)
    context_parts.append(f"[Physics Framework]\n{framework_text[:200]}...")

    # Knowledge 추가
    for module in modules[:1]:
        module_text = km.format_for_context([module])
        context_parts.append(f"[Knowledge: {module.get('id')}]\n{module_text[:200]}...")

    print("   Context assembled successfully")
    print(f"   Total context length: ~{sum(len(p) for p in context_parts)} chars")

    print("\n✅ End-to-End flow completed (LLM call skipped)")
    return {"status": "✅", "message": "Flow complete"}


def main():
    """메인 테스트 실행"""
    print("="*60)
    print("SOPHIA AI - System Integration Test")
    print("="*60)

    results = {}

    # 1. Classifier 테스트
    results["classifier"] = test_classifier()

    # 2. SolverRouter 테스트
    results["solver"] = test_solver_router()

    # 3. KnowledgeManager 테스트
    results["knowledge"] = test_knowledge_manager()

    # 4. FrameworkInjector 테스트
    results["framework"] = test_framework_injector()

    # 5. Knowledge 모듈 완전성
    results["completeness"] = test_knowledge_module_completeness()

    # 6. End-to-End
    results["e2e"] = test_end_to_end_flow()

    # 최종 요약
    print("\n" + "="*60)
    print("테스트 요약")
    print("="*60)

    # Classifier 정확도
    classifier_success = sum(1 for r in results["classifier"] if r["match"] == "✅")
    print(f"1. Classifier: {classifier_success}/{len(results['classifier'])}")

    # Solver 성공률
    solver_success = sum(1 for r in results["solver"] if r["status"] == "✅")
    print(f"2. Solver: {solver_success}/{len(results['solver'])}")

    # Knowledge 검색
    knowledge_success = sum(1 for r in results["knowledge"] if r["match"] == "✅")
    print(f"3. Knowledge: {knowledge_success}/{len(results['knowledge'])}")

    # Framework
    print(f"4. Framework: {results['framework']['status']}")

    # 모듈 완전성
    complete_modules = sum(1 for r in results["completeness"] if r["status"] == "✅")
    print(f"5. Module Completeness: {complete_modules}/{len(results['completeness'])}")

    # E2E
    print(f"6. End-to-End: {results['e2e']['status']}")


if __name__ == "__main__":
    main()
