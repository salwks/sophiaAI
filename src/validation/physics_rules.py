"""
물리 개념 검증 규칙 엔진

규칙을 YAML/JSON 형식으로 정의하고 동적으로 적용합니다.
새로운 규칙 추가 시 코드 수정 없이 RULES 딕셔너리에만 추가하면 됩니다.
"""

import re
from typing import List, Dict, Any

# ============================================
# 물리 규칙 정의 (확장 가능)
# ============================================
PHYSICS_RULES: List[Dict[str, Any]] = [
    # 규칙 1: 가우시안 필터 + MTF 유지 = 오류
    {
        "id": "gaussian_mtf",
        "name": "가우시안 필터-MTF 관계",
        "description": "가우시안 필터는 MTF를 손실시킵니다 (에지 블러링)",
        "error_patterns": [
            # 가우시안이 MTF/해상도를 유지한다고 주장하면 오류
            r"가우시안.*(?:필터|filter).*(?:mtf|해상도).*(?:유지|보존|preserv|maintain)",
            r"(?:mtf|해상도).*(?:유지|보존).*가우시안",
            r"gaussian.*filter.*(?:mtf|resolution).*(?:preserv|maintain)",
            r"gaussian.*(?:preserv|maintain).*(?:mtf|resolution)",
        ],
        "exception_keywords": ["손실", "저하", "감소", "불가", "loss", "degrad", "reduc", "blur"],
        "correct_statement": "가우시안 필터는 에지 블러링으로 인해 MTF(공간 해상도)를 손실시킵니다.",
    },

    # 규칙 2: 산란선 ≠ 양자노이즈
    {
        "id": "scatter_quantum",
        "name": "산란선-양자노이즈 구분",
        "description": "산란선(Scatter)과 양자노이즈(Quantum Mottle)는 다른 물리 현상입니다",
        "error_patterns": [
            r"산란.*(?:양자|quantum).*(?:같|유사|동일|identical|similar)",
            r"(?:양자|quantum).*산란.*(?:같|유사|동일)",
            r"scatter.*quantum.*(?:same|similar|identical)",
        ],
        "exception_keywords": ["다른", "구분", "차이", "differ", "distinct"],
        "correct_statement": "산란선은 X선의 방향 변화, 양자노이즈는 광자 수의 통계적 변동입니다.",
    },

    # 규칙 3: 선량 증가 ≠ 노이즈 증가
    {
        "id": "dose_noise",
        "name": "선량-노이즈 관계",
        "description": "선량 증가는 노이즈를 감소시킵니다 (SNR 향상)",
        "error_patterns": [
            r"(?:선량|dose).*(?:증가|높|increase).*(?:노이즈|noise).*(?:증가|높|increase)",
            r"(?:노이즈|noise).*(?:증가|높).*(?:선량|dose).*(?:증가|높)",
        ],
        "exception_keywords": ["감소", "낮", "줄", "decreas", "reduc", "lower"],
        "correct_statement": "선량 증가 → 광자 수 증가 → SNR 향상 → 노이즈 감소",
    },

    # 규칙 4: kVp 증가 = 투과력 증가
    {
        "id": "kvp_penetration",
        "name": "kVp-투과력 관계",
        "description": "kVp 증가는 X선 에너지와 투과력을 증가시킵니다",
        "error_patterns": [
            r"(?:kvp|kv).*(?:증가|높|increase).*(?:투과|penetr).*(?:감소|낮|decrease)",
            r"(?:kvp|kv).*(?:감소|낮|decrease).*(?:투과|penetr).*(?:증가|높|increase)",
        ],
        "exception_keywords": [],
        "correct_statement": "kVp↑ → X선 에너지↑ → 투과력↑ (두꺼운 조직 촬영에 적합)",
    },

    # 규칙 5: 미세석회화 검출 - 고해상도 필요
    {
        "id": "calcification_resolution",
        "name": "미세석회화-해상도 관계",
        "description": "미세석회화 검출에는 고해상도(높은 MTF)가 필수입니다",
        "error_patterns": [
            r"미세.*석회화.*(?:저해상도|low.*resolution|blur|흐림)",
            r"(?:저해상도|low.*resolution).*미세.*석회화.*(?:검출|detect)",
        ],
        "exception_keywords": ["어려", "불가", "difficul", "cannot", "unable"],
        "correct_statement": "미세석회화(100-500μm)는 고해상도 영상에서만 정확히 검출됩니다.",
    },

    # 규칙 6: 딥러닝 디노이징 - 학습 데이터 중요
    {
        "id": "dl_denoising_training",
        "name": "딥러닝 디노이징-학습 데이터",
        "description": "딥러닝 디노이징은 학습 데이터의 품질에 크게 의존합니다",
        "error_patterns": [
            r"딥러닝.*(?:디노이징|denoising).*(?:학습|train).*(?:불필요|필요.*없)",
            r"(?:cnn|deep.*learning).*(?:denois).*(?:no.*train|without.*train)",
        ],
        "exception_keywords": [],
        "correct_statement": "딥러닝 디노이징의 성능은 학습 데이터 품질과 양에 직접적으로 의존합니다.",
    },
]


def check_physics_errors(response: str) -> List[Dict[str, str]]:
    """
    응답에서 물리 개념 오류를 검사합니다.

    Args:
        response: LLM 응답 텍스트

    Returns:
        발견된 오류 목록 [{id, name, description, correct_statement}, ...]
    """
    errors = []
    response_lower = response.lower()

    for rule in PHYSICS_RULES:
        rule_triggered = False
        matched_text = ""

        # 오류 패턴 검사
        for pattern in rule["error_patterns"]:
            match = re.search(pattern, response_lower)
            if match:
                matched_text = match.group(0)
                rule_triggered = True
                break

        if not rule_triggered:
            continue

        # 예외 키워드 확인 (올바른 진술인 경우 제외)
        exception_keywords = rule.get("exception_keywords", [])
        if exception_keywords:
            # 매칭된 문장 주변 컨텍스트 확인 (앞뒤 50자)
            match_pos = response_lower.find(matched_text)
            context_start = max(0, match_pos - 50)
            context_end = min(len(response_lower), match_pos + len(matched_text) + 50)
            context = response_lower[context_start:context_end]

            # 예외 키워드가 컨텍스트에 있으면 올바른 진술
            if any(exc in context for exc in exception_keywords):
                continue

        # 오류로 확정
        errors.append({
            "id": rule["id"],
            "name": rule["name"],
            "description": rule["description"],
            "correct_statement": rule["correct_statement"],
        })

    return errors


def get_physics_rules_summary() -> str:
    """물리 규칙 요약을 반환합니다 (프롬프트에 포함용)"""
    summary_parts = ["# 물리 개념 정의 (엄격히 준수)"]

    for rule in PHYSICS_RULES:
        summary_parts.append(f"- {rule['name']}: {rule['correct_statement']}")

    return "\n".join(summary_parts)


def add_rule(rule: Dict[str, Any]) -> None:
    """새 규칙을 동적으로 추가합니다"""
    required_fields = ["id", "name", "description", "error_patterns", "correct_statement"]
    for field in required_fields:
        if field not in rule:
            raise ValueError(f"Missing required field: {field}")

    # 중복 ID 체크
    if any(r["id"] == rule["id"] for r in PHYSICS_RULES):
        raise ValueError(f"Rule with id '{rule['id']}' already exists")

    PHYSICS_RULES.append(rule)


# 테스트용
if __name__ == "__main__":
    # 테스트 케이스
    test_cases = [
        ("가우시안 필터는 MTF를 유지하면서 노이즈를 제거합니다.", True),  # 오류
        ("가우시안 필터는 MTF를 손실시키므로 사용하지 마세요.", False),  # 정상
        ("산란선과 양자노이즈는 유사한 현상입니다.", True),  # 오류
        ("산란선과 양자노이즈는 다른 현상입니다.", False),  # 정상
    ]

    print("물리 규칙 엔진 테스트")
    print("=" * 60)

    for text, should_error in test_cases:
        errors = check_physics_errors(text)
        has_error = len(errors) > 0
        status = "✅" if has_error == should_error else "❌"
        print(f"{status} '{text[:40]}...'")
        print(f"   예상: {'오류' if should_error else '정상'}, 실제: {'오류' if has_error else '정상'}")
        if errors:
            print(f"   발견된 오류: {[e['name'] for e in errors]}")
        print()
