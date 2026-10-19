# 팀원 v11 모델 연결

`source/`는 팀원 저장소 `C:\Users\Playdata\Desktop\port-vessel-matching`의
`solution/` 코드를 가져온 사본이다. 확인 시점의 팀원 커밋은 `4d8640b`였다.
`match_lib.py`와 `script.py`는 그대로 복사했고, `train.py`의 프로젝트·데이터·소스
경로만 현재 저장소 구조에 맞게 변경했다. 원본 팀원 폴더는 수정하지 않았다.

팀원 README의 v11 Public `0.992444`와 vessel 단위 CV `0.9951`은 해당 저장소의
기록이다. 현재 폴더에는 모델 산출물이나 리더보드 화면이 없어 이 수치를 직접
재확인하지 않았다.

## 로컬 모델 생성

현재 프로젝트의 공식 `data/open/train.csv`, `data/open/vessel_catalog.csv`만 사용한다.
Python 3.11 환경에서 아래 명령으로 v11 설정(`--no-dict --callsign
--name-lookup`)을 학습한다.

```powershell
.\.venv_competition_py311\Scripts\python.exe agent\team_v11\source\train.py --out team_v11_artifacts\v11\src --no-dict --callsign --name-lookup
```

`team_v11_artifacts/`는 Git에서 제외한다. `resources.pkl`에는 공식 train의
정답에서 만들어진 조회표가 들어 있으므로 공유 저장소에 올리지 않는다.

## 에이전트 연결

`agent.integrated_service.IntegratedMatchingService`는 v11을 기본 추천으로,
기존 EXP-014 매칭을 비교용으로 실행한다. `agent/graph.py`의 구조화된 상태는
`primary_result`와 `legacy_result`를 분리한다. 기존 API의
`baseline_result`는 EXP-014 비교 결과를 담는다. 추가 검색 및 후보 비교
도구는 기존 EXP-014 점수를 사용하므로 v11 원점수와 서로 비교하지 않는다.

```powershell
.\.venv_competition_py311\Scripts\python.exe scripts\run_integrated_agent.py --sample-id TRAIN_00000
```

위 명령은 두 모델의 단일 행 결과만 표시한다. `--mode agent`는 LangGraph가
설치된 개발 환경에서 검토 그래프를 실행한다. `--details`를 지정하면 후보별
특징을 포함한 전체 상태를 표시한다. 에이전트나 LLM은 제출 패키지에
포함하지 않으며, 이 연결은 새 리더보드 제출을 수행하지 않는다.

별도 개발 환경은 `.venv_agent_py311/`이다. 과거 설명에서 참조한
`requirements-agent-py311.txt`는 현재 저장소에 없으므로 해당 파일을 이용한 새 환경 설치는 재현할 수 없다. 이 환경은 대회용
`.venv_competition_py311/`과 분리되어 있다. 현재 저장소의 `pyproject.toml`은
Python 3.12 이상을 요구하므로 이 Python 3.11 환경은 별도로 관리한다.

```powershell
.\.venv_agent_py311\Scripts\python.exe scripts\run_integrated_agent.py --sample-id TRAIN_00000 --mode agent
```

이 명령은 공식 train의 입력 열만 에이전트에 전달하며 정답 열은 제거한다.
학습 행 실행은 흐름 확인용이지 독립 성능 평가가 아니다.

## 자동 후보 검색 실험 그래프

개별 기록 검토 그래프와 별개로 `agent/experiment_graph.py`가 오프라인 실험을
관리한다. 결정적 허용 목록에서 v11 기준 설정과 카탈로그 단어장 확장 1개·3개
설정을 생성하고, 고정된 seed42 선박 그룹 분할과 행 분할에서 각 설정의 랭커를
독립적으로 다시 학습한다. 검증 예측에서는 정답 열을 제거하고 후보 포함률,
전체·한글·비한글 정확도, 가중 원점수, 학습에서 본 선박과 처음 보는 선박의
정확도를 함께 기록한다. 그룹 겹침, 조회표 학습 범위, 일부 행의 단독·부분집합
예측 일치도 검사한다.

```powershell
.\.venv_agent_py311\Scripts\python.exe scripts\run_v11_experiment_agent.py
```

그래프는 대회용 Python 3.11 환경의 `scripts/v11_experiment_worker.py`에 학습과
평가를 위임한다. 실행별 집계 JSON·Markdown은 `experiments/results/`에 저장하고
요약은 `experiments/LOG.md`에 덧붙인다. 기본 실행의 설정 생성은 규칙 기반이다.
`--planner llm`을 선택하면 LLM이 1~2개의 **선언형 검색 정책**을 제안한다.
정책은 한글/비한글 조건, 이름 표기, 카탈로그 문자열 필드, 유사도 함수,
후보 수·유사도·톤수 필터만 조합한다. 제안된 코드를 실행하지 않는다.
구조·값·후보 예산을 검사하고, 두 고정 분할에서 기준 v11 랭커를 분할마다
한 번 학습해 고정한 채 후보 포함률과 최종 예측 점수를 비교한다. 이 선별 점수는
새 정책으로 랭커를 다시 학습한 결과가 아니다.

```powershell
$env:OPENAI_EXPERIMENT_MODEL = "사용할_모델_ID"
$env:OPENAI_API_KEY = "API_키"
.\.venv_agent_py311\Scripts\python.exe scripts\run_v11_experiment_agent.py --planner llm
```

LLM 요청에는 정적인 과제 설명과 정책 스키마만 넣는다. 공식 데이터 행·정답·집계는
전송하지 않는다. 모델 이름과 토큰 사용량은 실험 계획에 기록하며 API 키는 기록하지
않는다. 외부 API 호출은 `--planner llm`을 지정할 때만 발생한다. 이 경로는 실제 API
호출을 포함한 실행을 완료했으며, 아래 2026-10-19 수정 항목에 결과와 구현 불일치를 기록했다. 기존 target 그룹은 과거 실험에 노출되어
두 분할의 점수는 탐색용이다. 그래프는 제출 ZIP이나 리더보드를 변경하지 않는다.

### 2026-10-19 수정: `english_variant` 검색 단계 불일치

실행 `agent_v11_search_20261019_011012_4bed4e`의 `variant_skeleton_fallback` 정책은
비한글 행에 `english_variant`를 지정했지만, 비한글 질의에는 로마자 변형이 없어 실제로는
기본 `compact` 이름으로 되돌아갔다. 또 `token_set_ratio`가 공백을 뺀 문자열에서 계산되어
토큰 순서 가설을 검증하지 못했다(예: `PIONEER OCEAN 2` 대 `OCEAN PIONEER 2`가 공백 제거 시 61.5,
토큰 비교 시 100). `agent/search_policy.py`에서 다음을 바로잡았다.

- `english_variant`·`learned`는 한글 행에서만 값을 만들고, 비한글 행에서는 단계를 건너뛴다.
  `when=non_korean`과 함께 쓰면 검증 단계에서 거부한다.
- 공백 토큰 질의 `tokens`와 카탈로그 필드 `tokens`를 추가했다. `token_set_ratio`는 두 필드가
  모두 `tokens`일 때만 허용한다.
- 작업자 결과에 단계별 적용 행 수와 기본 이름과 다른 질의 수(`policy_behavior_probe`)를 기록한다.

위 실행의 기존 결과 파일은 그대로 보존했다. 수정된 정책으로 그 선별 평가를 다시 실행하지는 않았다.

## 진단 기반 단일 변경 재학습 루프

`agent/improvement_graph.py`는 다음 닫힌 흐름을 LangGraph로 실행한다.

```text
분할·데이터 감사 → 분할별 기준 v11 재학습·오류 진단 → 계획(LLM 또는 결정적)
  → 로컬 검사(guard) → 단일 변경 재학습·비교 → 중단 판단 → (다음 계획 | 기록)
```

- **분할:** 기존 `group_split_seed42/123/2026`, `row_split_seed42`. 모두 과거 검증·오류 분석에
  노출된 재사용 분할이며 분할 사이에 행이 겹친다. 결과는 탐색용이고 독립 검증이 아니다.
- **진단:** 예측이 끝난 뒤 평가 단계에서만 정답을 붙인다. 오답 전부와, 정답 행 중 모델 1·2위 점수
  차이가 가장 작은 50행(차이 오름차순, sample_id 오름차순)을 사례로 고른다. 정답 후보 순위, 1위와의
  점수 차, 이름 유사도·음역·숫자·톤수·결측 특징, 호출부호·이름 조회 override 전후 예측, seen/unseen
  target을 기록한다. 선택 규칙은 `agent/error_diagnostics.py::DIAGNOSTIC_PROTOCOL`에 고정했다.
  행 ID가 있는 사례 파일은 `experiments/results/<run>/local_only/`(Git 제외)에만 둔다.
- **허용 설정:** `agent/retrain_config.py::KNOBS`의 10개 항목(고정 랭커 구성, 어려운 음성 후보 수, 후보 밖 정답 처리,
  톤수 결측 표현, 번호 특징, 국적·선종 특징, 단조 제약, num_leaves, n_estimators, override 정책)에서
  한 번에 한 항목만 바꾼다. 같은 분할·행·seed 42·후보 생성 조건으로 기준과 비교한다. override 정책만
  랭커를 재학습하지 않고 같은 분할의 기준 모델을 재사용한다.
- **LLM 입력:** 식별 정보를 뺀 집계 진단, 구현된 특징·후처리 목록, 이전 실험 결과·비용, 허용 설정,
  남은 예산만 보낸다. `assert_llm_safe`가 ID 형식·한글·비유한 숫자를 막고, 실제 요청 본문(키 제외)은
  `<run>/llm/round_N_request.json`에 남긴다. 응답은 엄격한 JSON 스키마로 받는다.
- **로컬 검사:** 범위 밖 값, 기준과 같은 값, 선언된 중복, 근거 유형 미관측·불일치, 정답·다른 행·외부
  데이터 요구, 코드·경로·식별자가 든 문장, 이전 실행 중복(`experiments/results/agent_improvement_ledger.json`)
  을 거부하고 사유를 기록한다. 실행 후에는 설정·해시·이름 일치와 설정별 동작 검사(예: 음성 후보 수 상한,
  NaN 셀 수, 비활성 특징의 미사용)를 대조한다.
- **판정·예산:** `DECISION_RULE`을 사전에 고정했다(평균 가중 원점수 +0.0005 이상, 2개 이상 분할에서 개선,
  모든 분할에서 언어별 정확도와 가중 점수 비하락). 기본 예산은 계획 3회, 학습 실험 3회, 회차당 제안 3개,
  개선 없는 회차 2회에서 중단이다.

```powershell
# 결정적 계획(외부 API 없음)
.\.venv_agent_py311\Scripts\python.exe scripts\run_v11_improvement_agent.py
# LLM 계획(.env 또는 환경변수의 OPENAI_API_KEY, OPENAI_EXPERIMENT_MODEL 또는 OPENAI_CHAT_MODEL)
.\.venv_agent_py311\Scripts\python.exe scripts\run_v11_improvement_agent.py --planner llm
# 설정을 직접 고정한 단일 실험(LLM 없음, 중복·근거 검사는 동일)
.\.venv_agent_py311\Scripts\python.exe scripts\run_v11_improvement_agent.py --planner deterministic --knob ranker_family --value catboost_v1_cpu
# 기준 설정이 팀 train/predict를 그대로 재현하는지 확인
.\.venv_competition_py311\Scripts\python.exe scripts\v11_improvement_worker.py parity --split group_seed42 --rows 1500 --output <파일>
```

이 루프는 제출 ZIP·제출 코드·리더보드를 바꾸지 않는다. LLM·에이전트 의존성은 제출 패키지에 들어가지 않는다.

### 번호 비대칭 특징 제거

`agent_v11_improve_20261019_062733_04f948`에서 결정적 계획으로 `number_features=no_asymmetry` 한 설정을 네 분할에서 비교했다(LLM 호출 0회). `num_q_only`·`num_c_only`만 학습·추론에서 0으로 만들었다. seed42 비한글 1건 퇴보, seed2026 비한글 1건 개선, seed123·행 분할은 예측 변화가 없었다. 평균 가중 원점수 변화는 약 `-0.0000014`, 사전 판정은 `regressed`로 미채택했다. 네 기준 모델 해시는 이전 실행과 같았으며, 특징 비활성화·표본 행 독립성 검사도 통과했다. 상세 점수와 사용법은 루트 `README.md` 및 `experiments/LOG.md`를 따른다.

### 최신 재학습 실험: 고정 CatBoost 랭커 구성

`agent_v11_improve_20261019_072054_425152`에서 `ranker_family=catboost_v1_cpu`를 네 분할에서 평가했다(LLM 0회). 학습 행렬·검증 후보·특징 해시가 기준과 같고, 고정 프로필·600회 학습·표본 행 독립성·CBM 저장 및 재로딩 검사를 통과했다. seed42 가중 Δ −0.000142789, seed123·seed2026 변화 없음, row seed42 −0.000135685로 평균 −0.000069619, 판정 `regressed`이며 미채택이다. 한글 정확도는 전부 같고 새로 맞힘/틀림은 분할별 합계 1/3, 고유 행 1/2다. CatBoost 중요도는 PredictionValuesChange로 기록하고 LightGBM gain과 직접 차감하지 않는다. 오류 상보성은 조회표 전후로 집계했으며 앙상블은 평가하지 않았다. 입력·코드·버전·사전 가설·확인 결과는 해당 실행 폴더에 있다.
