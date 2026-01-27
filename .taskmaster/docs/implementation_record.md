# Implementation Record (구현 기록)

> 이 파일은 논문 기반 구현 내역과 검증 상태를 기록합니다.
> Claude는 작업 전 이 파일을 반드시 확인해야 합니다.

---

## 전체 파이프라인 아키텍처

```
┌─────────────────────────────────────────────────────────────────────────┐
│                         Maria-Mammo 파이프라인                            │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  [사용자 질문]                                                           │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────┐                           │
│  │ 1. Query Decomposition (Phase 7.20)     │                           │
│  │    - 질문 분석 → 필요 값 식별            │                           │
│  │    - Grounded Values 테이블 생성         │                           │
│  │    - 파일: query_decomposer.py           │                           │
│  └─────────────────────────────────────────┘                           │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────┐                           │
│  │ 2. Complexity Classification (MDAgents) │                           │
│  │    - LOW: 단일 에이전트 (14.7초)         │                           │
│  │    - MODERATE: MDT 협업 (95.5초)         │                           │
│  │    - HIGH: ICT 순차 (226초)              │                           │
│  │    - 파일: physics_triage.py             │                           │
│  └─────────────────────────────────────────┘                           │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────┐                           │
│  │ 3. Query Expansion (HALO P7)            │                           │
│  │    - 원본 질문 → 확장된 질문들           │                           │
│  │    - 파일: query_expander.py             │                           │
│  └─────────────────────────────────────────┘                           │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────┐                           │
│  │ 4. UnifiedPromptBuilder (Phase 7.19)    │                           │
│  │    - CQO/QOCO 템플릿 선택                │                           │
│  │    - Grounded Values 주입                │                           │
│  │    - 파일: unified_builder.py            │                           │
│  └─────────────────────────────────────────┘                           │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────┐                           │
│  │ 5. LLM 답변 생성                         │                           │
│  └─────────────────────────────────────────┘                           │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────┐                           │
│  │ 6. Agent-as-a-Judge (Phase 7.6)         │                           │
│  │    ┌───────────────────────────────┐    │                           │
│  │    │ 6.1 Planning Agent            │    │                           │
│  │    │     - 평가 작업 분해           │    │                           │
│  │    └───────────────────────────────┘    │                           │
│  │              │                          │                           │
│  │              ▼                          │                           │
│  │    ┌───────────────────────────────┐    │                           │
│  │    │ 6.2 Tool Verification (HERMES)│    │                           │
│  │    │     - SympyFormulaVerifier    │    │                           │
│  │    │     - DimensionalAnalyzer     │    │                           │
│  │    │     - KBCrossReferencer       │    │                           │
│  │    │     - VerificationMemory      │    │                           │
│  │    └───────────────────────────────┘    │                           │
│  │              │                          │                           │
│  │              ▼                          │                           │
│  │    ┌───────────────────────────────┐    │                           │
│  │    │ 6.3 Multi-Agent Debate        │    │                           │
│  │    │     (ChatEval)                │    │                           │
│  │    │     - PhysicsExpert           │    │                           │
│  │    │     - ClinicalExpert          │    │                           │
│  │    │     - EquipmentExpert         │    │                           │
│  │    │     - 2라운드 토론 + 합의      │    │                           │
│  │    └───────────────────────────────┘    │                           │
│  │              │                          │                           │
│  │              ▼                          │                           │
│  │    ┌───────────────────────────────┐    │                           │
│  │    │ 6.4 Correction Feedback       │    │                           │
│  │    │     - 수정 가이드 생성         │    │                           │
│  │    │     - 재생성 프롬프트          │    │                           │
│  │    └───────────────────────────────┘    │                           │
│  └─────────────────────────────────────────┘                           │
│       │                                                                 │
│       ▼                                                                 │
│  ┌─────────────────────────────────────────┐                           │
│  │ 7. 최종 답변 또는 재생성                  │                           │
│  └─────────────────────────────────────────┘                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 논문별 적용 방법 상세

### P1: Lost in Prompt Order (arXiv:2601.14152)
**적용 파일**: `src/retrieval/dynamic_evidence.py`

```
논문 핵심:
- 프롬프트 순서가 정확도에 영향 (+14.7%)
- QOCO가 가장 효과적 (Options 반복이 핵심!)

적용 방법:
┌─────────────────────────────────────────┐
│ CQO (Context-Question-Options)          │
│ ----------------------------------------│
│ 1. [C] 검증된 물리학 참조                │
│ 2. [Q] 질문                              │
│ 3. [O] 응답 요구사항                     │
└─────────────────────────────────────────┘

┌─────────────────────────────────────────┐
│ QOCO (Question-Options-Context-Options) │
│ ----------------------------------------│
│ 1. [Q] 질문                              │
│ 2. [O] 응답 요구사항 (첫 번째)           │
│ 3. [C] 검증된 물리학 참조                │
│ 4. [O] 응답 요구사항 (반복!) ← 핵심!     │
└─────────────────────────────────────────┘

코드 위치:
- _build_cqo_prompt() : 344행
- _build_qoco_prompt() : 380행 (Options 반복 포함)
```

### P3: MDAgents (arXiv:2404.15155)
**적용 파일**: `src/reasoning/physics_triage.py`

```
논문 핵심:
- 복잡도에 따른 적응형 라우팅
- 3단계: LOW / MODERATE / HIGH

적용 방법:
┌─────────────────────────────────────────────────────────────┐
│ ComplexityLevel                                             │
│ ------------------------------------------------------------│
│ LOW (14.7초)                                                │
│   - 조건: 간단한 정의/사실 질문                              │
│   - 키워드: "정의", "무엇", "차이"                           │
│   - 라우팅: 단일 에이전트                                    │
│                                                             │
│ MODERATE (95.5초)                                           │
│   - 조건: 설명/비교/임상적 분석 필요                         │
│   - 키워드: "설명", "영향", "비교", "분석"                   │
│   - 라우팅: MDT (Multi-Disciplinary Team)                   │
│                                                             │
│ HIGH (226초)                                                │
│   - 조건: 복잡한 계산, 다중 요소, 진단 결정                  │
│   - 키워드: "계산", "최적화", "진단", 수식 포함              │
│   - 라우팅: ICT (Individual → Collaborative → Team)         │
└─────────────────────────────────────────────────────────────┘

코드 위치:
- ComplexityLevel enum
- classify_complexity() 메서드
- triage_full() 메서드
```

### P5: Agent-as-a-Judge (arXiv:2601.05111)
**적용 파일**: `src/evaluation/agent_judge.py`

```
논문 핵심 (Survey):
- LLM-as-a-Judge의 한계 극복
- 4가지 핵심 요소: Planning, Tool, Multi-Agent, Memory

적용 방법:
┌─────────────────────────────────────────────────────────────┐
│ 1. Planning Agent (평가 작업 분해)                          │
│    - 답변 내용 분석                                         │
│    - 필요한 검증 도구 결정                                   │
│    - subtasks 리스트 생성                                    │
│                                                             │
│ 2. Tool Verification (도구 기반 검증)                       │
│    → HERMES 스타일 (arXiv:2511.18760)                       │
│                                                             │
│ 3. Multi-Agent Debate (다중 에이전트 토론)                  │
│    → ChatEval 스타일 (arXiv:2308.07201)                     │
│                                                             │
│ 4. Correction Feedback (수정 가이드)                        │
│    - 발견된 문제 → 구체적 수정 방향                         │
│    - 재생성용 프롬프트 생성                                  │
└─────────────────────────────────────────────────────────────┘
```

### P5a: ChatEval (arXiv:2308.07201)
**적용 파일**: `src/evaluation/agent_judge.py` - MultiAgentDebate 클래스

```
논문 핵심:
- Multi-Agent Debate로 편향 감소
- 다양한 페르소나 필수 (같은 역할 = 성능 저하)
- One-by-One 통신 전략

적용 방법:
┌─────────────────────────────────────────────────────────────┐
│ Expert Agents (3명)                                         │
│ ------------------------------------------------------------│
│ PhysicsExpert                                               │
│   - 페르소나: "의료영상물리학 전문가"                        │
│   - 평가 초점: 수치 정확성, 물리 법칙                        │
│   - 키워드: snr, cnr, dqe, mtf, 노이즈, 선량                │
│                                                             │
│ ClinicalExpert                                              │
│   - 페르소나: "영상의학과 전문의"                            │
│   - 평가 초점: 진단적 가치, 임상적 의미                      │
│   - 키워드: 진단, 환자, FN, FP, 검출                        │
│                                                             │
│ EquipmentExpert                                             │
│   - 페르소나: "의료기기 엔지니어"                            │
│   - 평가 초점: 장비 특성, 기술 사양                          │
│   - 키워드: 검출기, CEM, a-Se, 픽셀                         │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Debate Protocol                                             │
│ ------------------------------------------------------------│
│ 1. Round 1                                                  │
│    - PhysicsExpert 평가 (이전 의견 없음)                    │
│    - ClinicalExpert 평가 (Physics 의견 참고)                │
│    - EquipmentExpert 평가 (Physics, Clinical 참고)          │
│    - 합의 체크 (2/3 이상 동의?)                             │
│                                                             │
│ 2. Round 2 (합의 안 된 경우)                                │
│    - 모든 에이전트가 이전 라운드 의견 참고                   │
│    - 재평가 후 최종 다수결                                   │
│                                                             │
│ 3. 최종 판정                                                │
│    - 다수결로 APPROVED/REVISION_REQUIRED/REJECTED           │
│    - 평균 점수 계산                                         │
└─────────────────────────────────────────────────────────────┘
```

### P5b: HERMES (arXiv:2511.18760)
**적용 파일**: `src/evaluation/agent_judge.py` - Tool Verifiers

```
논문 핵심:
- 비공식 추론 + 공식 검증 인터리빙
- Memory 모듈로 증명 연속성 유지
- 각 단계마다 도구 호출로 검증

적용 방법:
┌─────────────────────────────────────────────────────────────┐
│ VerificationMemory (메모리 모듈)                            │
│ ------------------------------------------------------------│
│ - steps: List[VerificationStep]  # 검증 단계 기록          │
│ - intermediate_states: Dict      # 중간 상태 저장           │
│ - get_all_issues()               # 모든 이슈 수집           │
│ - get_all_corrections()          # 모든 수정 가이드 수집    │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ VerificationStep (검증 단계)                                │
│ ------------------------------------------------------------│
│ - step_id: int                   # 단계 번호                │
│ - tool_used: str                 # 사용 도구                │
│ - input_claim: str               # 검증 대상                │
│ - verified: bool                 # 검증 통과 여부           │
│ - issue: Optional[str]           # 발견된 문제              │
│ - correction: Optional[str]      # 수정 가이드 ← 핵심!      │
└─────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────┐
│ Tool Verifiers                                              │
│ ------------------------------------------------------------│
│ SympyFormulaVerifier                                        │
│   - Ghosting/Lag 혼동 감지 (100배 차이)                     │
│   - W값 오류 감지 (Si 3.6eV vs CEM 50-64eV)                 │
│   - Sympy로 수식 검증 (optional)                            │
│   - 수정 가이드 생성                                        │
│                                                             │
│ DimensionalAnalyzer                                         │
│   - 단위 일관성 검증 (mGy vs Gy)                            │
│   - MGD 범위 검증 (0.5-3 mGy)                               │
│                                                             │
│ KBCrossReferencer                                           │
│   - KB 값과 대조                                            │
│   - 출처 포함 수정 가이드 생성                               │
└─────────────────────────────────────────────────────────────┘
```

### P7: HALO Framework (arXiv:2409.10011)
**적용 파일**: `src/search/query_expander.py`

```
논문 핵심:
- Query Expansion으로 검색 품질 향상
- MMR로 관련성 + 다양성 균형

적용 방법 (Query Expansion - 완료):
┌─────────────────────────────────────────────────────────────┐
│ 원본 질문                                                    │
│ "CEM 검출기의 Ghosting이란?"                                │
│                                                             │
│           ▼ Query Expansion                                 │
│                                                             │
│ 확장된 질문들                                                │
│ - "CEM 검출기 Ghosting 정의"                                │
│ - "CEM Ghosting artifact 원인"                              │
│ - "CEM 잔류 신호 비율"                                       │
│ - "Ghosting vs Lag 차이"                                    │
└─────────────────────────────────────────────────────────────┘

MMR Ranking (Pending - Task #3):
┌─────────────────────────────────────────────────────────────┐
│ score = α × relevance(doc, query)                           │
│        - (1-α) × max_similarity(doc, selected_docs)         │
│                                                             │
│ α = 0.7 (관련성 70%, 다양성 30%)                            │
└─────────────────────────────────────────────────────────────┘
```

---

## 완료된 구현 (Verified)

### Phase 7.19: 지식 파이프라인 통합
- **상태**: ✅ 완료 (2026-01-27)
- **파일**:
  - `data/knowledge/physics/core_physics.json`
  - `src/prompts/unified_builder.py`
- **내용**: 4개 분산 경로 → UnifiedPromptBuilder 단일화

### Phase 7.20: Query Decomposition
- **상태**: ✅ 완료 (2026-01-27)
- **파일**: `src/retrieval/query_decomposer.py`
- **내용**: Ghosting(15%)/Lag(0.15%) 혼동 방지를 위한 Grounded Values 주입

### Phase 7.21: CEM Detector Physics Priority & Smart Truncation
- **상태**: ✅ 완료 (2026-01-27)
- **파일**:
  - `src/knowledge/manager.py` - CEM ghosting 컨텍스트 감지 + detector_physics 최우선 배치
  - `src/prompts/unified_builder.py` - `_smart_truncate()` priority markers에 Ghosting/Lag/QDE 추가
  - `src/retrieval/dynamic_evidence.py` - raw truncation → `_smart_truncate()` 사용
- **내용**:
  - **문제**: CEM ghosting 질문에서 `dbt_image_quality`가 `detector_physics`보다 먼저 매칭됨
    - "ghosting"이 `tomo_iq_keywords`에 있어서 DBT가 우선순위를 가짐
    - Ghosting/Lag 경고 테이블이 4831자 위치에 있어 4000자 truncation에서 잘림
  - **해결**:
    1. `is_cem_ghosting_context` 감지: CEM+ghosting/lag/QDE 조합 감지
    2. `detector_physics` 최종 우선순위 배치 (ABSOLUTE PRIORITY)
    3. `_smart_truncate()`가 "Ghosting", "Lag", "QDE", "100배 차이" 섹션 보존
  - **결과**: Ghosting 경고가 188자에서 시작 (4831자 → 188자)

### Phase 7.22: 누적 노출 노후화 + 적응적 CEM 보정 알고리즘
- **상태**: ✅ 완료 (2026-01-27)
- **논문 근거**:
  - Starman 2012 NLCSC (PMC3465354) - 노출 의존적 비선형 lag 보정
  - Kabir & Kasap 2012 (IEEE) - Ghosting 회복 메커니즘, 바이어스 반전
  - Zhao 2008 (PMC2673645) - 트랩 채움 효과, 민감도 변화
  - Bloomquist 2006 - 장비별 lag/ghosting 변동
- **파일**:
  - `data/knowledge/physics/detector_physics.json` - 새 섹션 추가
  - `src/knowledge/manager.py` - 포맷팅 함수 업데이트
- **추가된 지식**:
  1. **누적 노출 노후화 (cumulative_exposure_aging)**:
     - 트랩 사이트 누적 현상 (X선 유도 심층 트랩 센터 생성)
     - Time-Variant Gain Map 문제 (G_HE가 시변 변수)
     - 가역적 vs 비가역적 손상 구분
  2. **적응적 CEM 보정 (adaptive_cem_correction)**:
     - Virtual DE-Gain: `G_HE = G_LE × (56/97) × (1 - β × D_cumulative)`
     - 노출 이력 Ghosting 감쇄: `G_eff = Σ αᵢ × mAsᵢ × exp(-tᵢ/τᵢ)`
       - Fast (홀 트랩): α₁=0.10, τ₁=2분
       - Slow (전자 트랩): α₂=0.05, τ₂=60분
     - 민감도 역보정: `S_corrected = S_measured / (1 - 0.15) = S × 1.176`
       - ❌ 틀린 방법: 15% 더하기
       - ✅ 맞는 방법: (1-0.15)로 나누기 = 17.6% 증폭
  3. **Self-Diagnostic 노후화 추적**:
     - SNR 저하율 모니터링 → β 자동 조정
     - "장비 새로 산 것 같다" 효과

### Task #1: CQO/QOCO 프롬프트 템플릿 (P1)
- **논문**: Lost in Prompt Order (arXiv:2601.14152)
- **상태**: ✅ 완료 + 논문 검증됨
- **파일**: `src/retrieval/dynamic_evidence.py`
- **검증 내용**:
  - CQO: Context → Question → Options ✅
  - QOCO: Question → Options → Context → **Options 반복** ✅ (논문 핵심!)
  - 수정: Options 반복 누락 → 추가 완료

### Task #2: Query Expansion (P7)
- **논문**: HALO Framework (arXiv:2409.10011)
- **상태**: ✅ 완료
- **파일**: `src/search/query_expander.py`
- **내용**: 질문 확장으로 검색 품질 향상

### Task #3: MMR Ranking (P7)
- **논문**: HALO Framework (arXiv:2409.10011)
- **상태**: ⏳ Pending
- **내용**: 관련성 + 다양성 균형 (Maximum Marginal Relevance)

### Task #4: 복잡도 기반 적응형 라우팅 (P3)
- **논문**: MDAgents (arXiv:2404.15155)
- **상태**: ✅ 완료 + 논문 검증됨
- **파일**: `src/reasoning/physics_triage.py`
- **검증 내용**:
  - ComplexityLevel enum: LOW/MODERATE/HIGH ✅
  - 라우팅: LOW → 단일 에이전트, MODERATE → MDT, HIGH → ICT ✅
  - classify_complexity() 메서드 ✅
  - triage_full() 메서드 ✅

### Task #5: MedAgents 5-Step Multi-Agent
- **논문**: MedAgents (arXiv:2311.10537)
- **상태**: ⏳ Pending
- **내용**: Expert Gathering → Analysis → Report → Voting → Decision

### Task #6: Agent-as-a-Judge 도구 기반 검증 (P5)
- **논문**:
  - Agent-as-a-Judge Survey (arXiv:2601.05111)
  - ChatEval (arXiv:2308.07201)
  - HERMES (arXiv:2511.18760)
- **상태**: ✅ 완료 + 논문 검증됨
- **파일**: `src/evaluation/agent_judge.py`
- **검증 내용**:
  - **ChatEval 스타일 Multi-Agent Debate**:
    - 3 Expert Personas (Physics, Clinical, Equipment) ✅
    - One-by-One 통신 전략 ✅
    - 2라운드 토론 + 다수결 합의 ✅
  - **HERMES 스타일 Tool Verification**:
    - VerificationMemory 모듈 ✅
    - Step-by-step 검증 ✅
    - Sympy 통합 (optional) ✅
  - **Planning Agent**: 평가 작업 분해 ✅
  - **Correction Feedback**: 수정 가이드 생성 ✅
  - **수정 프롬프트**: 재생성용 지침 생성 ✅

### Task #7: O1-style 적응형 추론 (P6)
- **논문**: O1-style Medical Reasoning (arXiv:2501.06458)
- **상태**: ⏳ Pending

### Task #8: Ghosting/Lag 혼동 수정 (긴급)
- **상태**: ✅ 완료
- **파일**: `src/retrieval/query_decomposer.py`
- **내용**: Query Decomposition으로 검증된 값 강제 주입

### Task #9: Query Decomposition + UnifiedPromptBuilder 통합
- **상태**: ✅ 완료
- **파일**: `src/prompts/unified_builder.py`
- **내용**: Grounded values를 프롬프트 최상단에 배치

### Task #10: 지식 파이프라인 통합 (Phase 7.19)
- **상태**: ✅ 완료

---

## 논문 참조 목록

| ID | 논문 | arXiv | 적용 위치 | 검증 |
|----|------|-------|-----------|------|
| P1 | Lost in Prompt Order | 2601.14152 | dynamic_evidence.py | ✅ |
| P2 | MedAgents | 2311.10537 | (pending) | - |
| P3 | MDAgents | 2404.15155 | physics_triage.py | ✅ |
| P5 | Agent-as-a-Judge | 2601.05111 | agent_judge.py | ✅ |
| P5a | ChatEval | 2308.07201 | agent_judge.py | ✅ |
| P5b | HERMES | 2511.18760 | agent_judge.py | ✅ |
| P6 | O1-style Reasoning | 2501.06458 | (pending) | - |
| P7 | HALO Framework | 2409.10011 | query_expander.py | ✅ |

---

## 핵심 수치 (Verified Constants)

| 항목 | 값 | 주의사항 | 출처 |
|------|-----|---------|------|
| Ghosting | 15% | Lag(0.15%)와 100배 차이! | Medical Physics 39(12), 2012 |
| Lag | 0.15% | Ghosting(15%)과 혼동 금지 | Medical Physics 39(12), 2012 |
| QDE(LE) | 97% | Low Energy | Cho et al. 2008 |
| QDE(HE) | 56% | High Energy | Cho et al. 2008 |
| W (CEM) | 50-64 eV | Si(3.6eV)와 다름! | Kasap & Rowlands 2000 |
| W (Si) | 3.6 eV | CEM에 적용 금지 | Standard |

---

## 통합 상태 (Integration Status)

### Agent-as-a-Judge → app.py
- **상태**: ✅ 통합 완료 (2026-01-27)
- **파일**: `src/ui/app.py`
- **위치**: LLM 응답 생성 후 (~1503행)
- **코드**:
  ```python
  # Phase 7.6: Agent-as-a-Judge 평가
  if options.get("enable_judge", True):
      judge = get_agent_judge()
      judge_result = judge.evaluate(
          question=prompt,
          answer=full_response,
          reference_knowledge=relevant_knowledge,
          context=context
      )
  ```
- **기능**:
  - 답변 품질 검증 (Planning + Tool + Debate)
  - 수정 가이드 표시 (문제 발견 시)
  - 재생성 프롬프트 제공

### MDAgents 복잡도 분류 → orchestrator.py
- **상태**: ✅ 통합 완료 (2026-01-27)
- **파일**: `src/reasoning/orchestrator.py`
- **위치**: `process()` 메서드 (~143행)
- **코드**:
  ```python
  # Phase 7.6: MDAgents 복잡도 분류
  complexity_result = None
  if self.config.enable_physics_triage and self.physics_triage:
      complexity_result = self.physics_triage.classify_complexity(question)
      logger.info(f"[MDAgents] Complexity: {complexity_result.level.value}")
  ```
- **기능**:
  - 질문 복잡도 분류 (LOW/MODERATE/HIGH)
  - 적응형 라우팅 결정
  - OrchestrationResult에 complexity_level, complexity_confidence 포함

---

## 변경 이력

| 날짜 | 내용 | 담당 |
|------|------|------|
| 2026-01-27 | Task #4 (MDAgents) 완료 | Claude |
| 2026-01-27 | Task #6 (Agent-as-a-Judge) 논문 기반 재구현 | Claude |
| 2026-01-27 | ChatEval/HERMES 논문 검증 및 적용 | Claude |
| 2026-01-27 | 수정 프롬프트 생성 기능 추가 | Claude |
| 2026-01-27 | Agent-as-a-Judge → app.py 통합 완료 | Claude |
| 2026-01-27 | MDAgents 복잡도 분류 → orchestrator.py 통합 완료 | Claude |
| 2026-01-27 | Agent-as-a-Judge 자동 재생성 기능 추가 | Claude |
| 2026-01-27 | Ghosting/Lag 보정 알고리즘 논문 검증 및 KB 추가 | Claude |
| 2026-01-27 | Lag 보정 알고리즘 논문 상세 추가 (Mail 2007, Starman NLCSC/Forward Bias) | Claude |
| 2026-01-27 | CEM Dual-Energy Calibration 논문 전문 검증 (PMC2858626, PMC3188980) | Claude |
| 2026-01-27 | Phase 7.22: HE 캘리브레이션 vs 소프트웨어 보정 논문 근거 추가 | Claude |
| 2026-01-27 | PMC11788233 - X-ray 튜브 열사이클링 손상 전문 검증 | Claude |
| 2026-01-27 | PMC5722609 - 소프트웨어 lag 보정 >80% 효과 전문 검증 | Claude |
| 2026-01-27 | PMC3257750 - Forward bias 70-88% lag 감소 전문 검증 | Claude |
| 2026-01-27 | PMC2826385 - LE/HE 4분 대기 ghosting 최소화 전문 검증 | Claude |
| 2026-01-27 | 지식베이스: HE 캘리브레이션 비권장, 소프트웨어 보정 우선 권장 반영 | Claude |
| 2026-01-27 | Phase 7.23: CEM 확대촬영 SNR 물리학 논문 검증 및 추가 | Claude |
| 2026-01-27 | PMC4562003 (Lalji 2015) - LE 석회화 가시성 우수 전문 검증 | Claude |
| 2026-01-27 | PMC3446357 (Dromain 2012) - CEM 타이밍 프로토콜 전문 검증 | Claude |
| 2026-01-27 | 1/M² 광자 감소 법칙, Rose Criterion, PK-Gain 노이즈 증폭 추가 | Claude |

---

## Knowledge Base 논문 검증 상태

### detector_physics.json - Lag/Ghosting 보정 알고리즘

| 논문 | PMID | PMC | 핵심 내용 | 상태 |
|------|------|-----|---------|------|
| Mail 2007 | 17712306 | PMC5722609 | Two-exponential decay model: Lₙ = C₀ + C₁exp(-nτP₁) + C₂exp(-nτP₂), 피팅 계수 포함 | ✅ 검증/추가 완료 |
| Starman 2012 NLCSC | 23039642 | PMC3465354 | 노출 의존적 비선형 lag 보정: a₂,ₙ(x) = c₁(1 - e^(-c₂x)), 잔류 lag <0.29% | ✅ 검증/추가 완료 |
| Starman 2012 Forward Bias | 22225300 | PMC3257750 | 하드웨어 방식: 4V@100kHz, 70-88% lag 감소 | ✅ 검증/추가 완료 |
| Zhao 2005 Ghosting | 15789596 | - | Langevin 재결합, 트랩 에너지 (holes 0.9eV, electrons 1.2eV) | ✅ 기존 검증됨 |
| Bloomquist 2006 | 16964878 | - | Ghosting=15%, Lag=0.15% 임상 측정값 | ✅ 기존 검증됨 |

### CEM Dual-Energy Calibration 논문 (전문 검증 완료)

| 논문 | PMC | 핵심 내용 | 상태 |
|------|-----|---------|------|
| Chen 2010 | PMC2858626 | Cubic inverse-mapping (tc=c₀+c₁Dₗ+...), KNR optimal scale S=0.00145, 노이즈 감소 90-95% | ✅ 전문 검증 |
| Photon-counting DBT | PMC3188980 | w_t = 0.46-0.72, SI_DE = ln[SI_Cu] - w_t·ln[SI_Sn], lag-free detector | ✅ 전문 검증 |

### 실무적 보정 알고리즘 근거

| 개념 | 논문 근거 | 상태 |
|------|----------|------|
| Dose-weighted ghosting decay | NLCSC (Starman 2012) - 노출 의존적 lag 모델 + Zhao 2005 회복 메커니즘 | ✅ 이론적 근거 확보 |
| LUT-based Gain Map | Hajdok 2006 QDE 에너지 의존성 (LE 97% vs HE 56%) | ⚠️ 개념적 근거만 (벤더 proprietary) |
| CEM Dual-Energy Calibration | PMC2858626, PMC3188980 | ✅ 전문 검증 완료 - 수식/파라미터 추출 |

### Phase 7.22: HE 캘리브레이션 vs 소프트웨어 보정 논문 근거 (전문 검증 완료)

| 논문 | PMC | 핵심 발견 | 상태 |
|------|-----|---------|------|
| Behling 2025 | PMC11788233 | X-ray 튜브 열 사이클링 → 피팅/크래킹/용융, 열전도 44% 감소, 30kV 91% 제한 권장 | ✅ 전문 검증 |
| Mail 2007 | PMC5722609 | 소프트웨어 lag 보정 >80% 효과, "매 환자 후 offset/gain 보정" 권장 | ✅ 전문 검증 |
| Starman 2012 Forward Bias | PMC3257750 | lag ghost 70-88% 감소, CBCT 아티팩트 48-81% 감소, 프레임 레이트 페널티 없음 | ✅ 전문 검증 |
| Laidevant 2010 | PMC2826385 | LE/HE 4분 대기로 ghosting 최소화, 최신 디텍터 현저히 낮은 ghosting, 6개월 안정성 | ✅ 전문 검증 |

#### ⚠️ 결론: HE 캘리브레이션 대신 소프트웨어 보정 권장

**HE 캘리브레이션의 문제점 (논문 근거)**:
1. **튜브 열 사이클링 손상** (PMC11788233): 반복 노출 → 표면 크래킹, 피팅, 입자 손실 → 튜브 수명 단축
2. **디텍터 누적 노출** (PMC2847939): 반복 HE 노출 → X선 유도 새로운 심층 트랩 영구 생성 → ghosting 악화
3. **열전도 저하**: 침식된 표면 열전도 44% 감소 → thermal runaway 위험

**소프트웨어 보정의 장점 (논문 근거)**:
1. **높은 효과**: 소프트웨어 lag 보정 >80% 효과 (PMC5722609)
2. **하드웨어 보호**: 추가 방사선 노출 없음 → 튜브/디텍터 수명 연장
3. **동적 보정**: 촬영 이력 기반 적응적 보정 가능 (PMC3465354)
4. **프레임 레이트 유지**: hardware flush 대비 속도 페널티 없음 (PMC3257750)
5. **대기 시간 최적화**: 4분 → 0.5초 가능 (최신 디텍터 + 소프트웨어 보정)

---

## 다음 작업 (Pending)

1. **Task #3**: MMR Ranking (HALO P7)
2. **Task #5**: MedAgents 5-Step Multi-Agent
3. **Task #7**: O1-style 적응형 추론
