# Port-Problem-Solving

항만 문제해결 AI 기술 공모전의 선박명 매칭 실험 프로젝트입니다. 한글·비한글 선박명을 카탈로그의 `vessel_id`와 연결하며, 후보 검색과 재랭킹을 단계별로 비교합니다.

## 현재 상태 (2026-10-19)

MLP OOF의 GPU 실행 경로는 별도 CUDA 프로토콜로 준비되어 있습니다. CUDA 프로토콜 파일은 사전 고정된 CPU 프로토콜의 fold를 그대로 복사해 만들며(`--from-cpu-split`), fold digest가 CPU와 같은지 생성 시점에 확인합니다. 그래도 CPU fold 0~3 결과가 알려진 뒤 만든 파일이므로 CPU OOF가 주 분석이고 CUDA는 보조 장치 확인입니다. CUDA 점수가 더 좋아도 둘 중 유리한 결과를 선택하지 않습니다. 작업자 환경에 CUDA 지원 PyTorch와 GPU가 준비된 경우 저장소 루트에서 아래 명령을 사용합니다. 첫 명령은 새 환경에서 분할 파일이 없을 때만 실행하며 기존 파일을 덮어쓰지 않습니다. 현재 로컬 PyTorch는 CPU 빌드라 GPU 학습·점수는 검증되지 않았습니다. 기준 v11 LightGBM 학습과 후보/특징 생성은 CPU 경로를 유지합니다.

```powershell
.\.venv_competition_py311\Scripts\python.exe scripts\freeze_v11_mlp_oof_split.py --gpu
.\.venv_competition_py311\Scripts\python.exe scripts\run_v11_mlp_oof_experiment.py --gpu
```

실행기는 학습에 사용하는 바로 그 `.venv_competition_py311` Python에서 CUDA 가능 여부를 결과 폴더 생성 전에 확인합니다. 개선 에이전트의 재사용 4분할 계획에서는 CPU·CUDA MLP를 모두 제안하지 않습니다.

주 분석인 CPU OOF의 원본 고정 분할 파일은 복구되어 `experiments/splits/agent_v11_mlp_listwise_group5_seed20261019.json`에 있습니다(SHA-256 `04d92f91874aae3340cb530eb2ca3c53ee1d6b68a94aa0e23a523a3a7b4e2292`). 이 환경에서 같은 seed로 분할을 다시 만들면 **다른 fold가 나오므로**, 이 파일을 재생성하지 말고 그대로 사용하고 CUDA 프로토콜도 이 파일에서 파생시킵니다.

```powershell
.\.venv_competition_py311\Scripts\python.exe scripts\freeze_v11_mlp_oof_split.py --compare-to experiments\splits\agent_v11_mlp_listwise_group5_seed20261019.json
.\.venv_competition_py311\Scripts\python.exe scripts\freeze_v11_mlp_oof_split.py --refrozen-after-exposure
.\.venv_competition_py311\Scripts\python.exe scripts\run_v11_mlp_oof_experiment.py
```

`--compare-to`는 파일을 쓰지 않고 현재 환경에서 다시 만든 fold digest와 기존 파일의 fold digest를 비교합니다. `--refrozen-after-exposure`로 고정한 파일은 `analysis_role`이 `primary_cpu_oof_refrozen_after_exposure`가 되고, 그 OOF 결과는 사전 고정이 아닌 탐색 결과로 해석합니다. 고정 파일은 덮어쓰지 않으므로, 기존 파일이 있으면 freeze는 실패합니다. GPU 결정성(같은 학습 2회 반복, 배치 크기별 점수 일치)과 행 독립성은 아직 검증되지 않았고 첫 GPU 실행 때 기록해야 합니다.

- 기본 추천 모델은 팀원 코드와 병합한 **v11 LightGBM 랭커**입니다. 기존 EXP-014는 비교용으로 보존합니다.
- 2026-10-19 대회 평가 대상이 바뀌고 리더보드가 초기화됐습니다. 초기화 후 `submissions/team_v11_merged.zip`의 사용자 보고 Public 점수는 **0.88**입니다. 리더보드 화면·업로드 파일 해시·언어별 점수는 이 기록에서 직접 확인하지 않았습니다. 초기화 이전 점수는 현재와 비교할 수 없어 문서에서 삭제했습니다. 변경된 운영 규칙은 `CLAUDE.md`와 `GPT.md` 15절을 따릅니다.
- 보관된 로컬 v11 ZIP의 기록된 SHA-256은 `33dc878b5f4987011eb217f85527d960630225bd80c5d7c2ab60e64448420ab2`입니다. 로컬 파일 동일성과 서버 업로드 파일 동일성은 별도로 확인해야 합니다.
- 단일 기록 검토 에이전트와 모델 개선 실험 에이전트가 있습니다. 실험 결과를 제출 모델에 자동 적용하지 않습니다.
- 모든 기존 target 그룹은 과거 검증·오류 분석에 노출됐습니다. 아래 로컬 점수는 **탐색 결과**이며 독립 일반화 성능이나 리더보드 점수의 추정치가 아닙니다.

실험별 설정, 점수, 해석 및 채택 여부는 [`experiments/LOG.md`](experiments/LOG.md)에 기록합니다. 실험 산출물은 `experiments/results/` 아래 별도 폴더에 보존합니다.

## 모델과 실행 환경

공식 입력은 `data/open/train.csv`와 `data/open/vessel_catalog.csv`입니다. 원본 CSV는 수정하지 않습니다. 입력 데이터, 모델 산출물, 제출 ZIP, 실험별 결과는 Git에서 제외합니다.

| 환경 | 용도 |
|---|---|
| `.venv_agent_py311/` | LangGraph와 실험 기획·검토 에이전트가 설치된 로컬 Python 3.11 환경 |
| `.venv_competition_py311/` | Python 3.11 대회 라이브러리로 모델 학습·평가 |
| 프로젝트 기본 `uv` 환경 | `pyproject.toml`의 Python 3.12 이상 개발 환경; 위 두 환경과 구분 |

실험 그래프는 별도 프로세스로 대회용 Python을 호출합니다. 아래 명령은 저장소 루트에서 이미 준비된 환경을 사용합니다. 상세 모델 학습·연결 방법은 [`agent/team_v11/README.md`](agent/team_v11/README.md)를 따릅니다.

고정 분할 중 sample_id 목록만 담은 `experiments/splits/group_split_seed42.json`·`group_split_seed123.json`·`group_split_seed2026.json`·`row_split_seed42.json`은 저장소에 포함합니다. 실험 프로토콜(`experiments/protocols/`)이 참조하는 MLP 5-fold 분할 `agent_v11_mlp_listwise_group5_seed20261019.json`(SHA-256 `04d92f91…`)과 `agent_v11_mlp_listwise_cuda_group5_seed20261019.json`(`d7702028…`)은 fold별 target `vessel_id` 목록을 담고 있어 Git에서 제외하며 별도로 전달합니다. 이 분할은 scikit-learn 버전에 따라 fold가 달라져 다시 만들 수 없으므로, 받은 파일의 해시를 프로토콜 기록과 대조해 사용합니다. 그 밖에 Git 제외 로컬 파일인 공식 CSV와 `team_v11_artifacts/v11/training_metadata.json`도 필요합니다. 작업자는 이 메타데이터와 현재 입력·소스 해시를 대조합니다. 저장소 복제만으로는 이 파일들과 두 가상환경이 준비되지 않습니다. 검토 에이전트의 패키지 목록은 `agent/team_v11/requirements-agent-py311.txt`에 있습니다.

## 에이전트 사용

### 단일 기록 검토

```powershell
.\.venv_agent_py311\Scripts\python.exe scripts\run_integrated_agent.py --sample-id TRAIN_00000 --mode agent
```

v11 추천과 EXP-014 비교 결과를 반환합니다. 공식 train 행 실행은 기능 확인용입니다. 사람 검토의 실제 중단·재개와 전체 수용 조건 검증은 남아 있습니다.

### 진단 기반 성능 개선 실험

```text
입력·고정 분할 감사 → 기준 v11 재학습·오류 진단
  → 가설·단일 변경 설정 제안 → 허용 범위·중복 검사
  → 변경 모델 재학습·기준 비교 → 결과를 다음 계획에 반영 또는 중단 → 기록
```

외부 API 없이 집계 진단을 근거로 아직 실행하지 않은 설정 하나를 선택하려면:

```powershell
.\.venv_agent_py311\Scripts\python.exe scripts\run_v11_improvement_agent.py --planner deterministic --max-llm-calls 1 --max-training-experiments 1 --max-proposals 1 --patience 1
```

이 명령의 선택은 현재 실험 원장에 따라 달라집니다. `--max-llm-calls`는 결정적 모드에서도 최대 계획 회차를 제한하며, 결정적 모드의 실제 LLM 호출은 0회입니다.

허용 설정을 직접 지정할 수도 있습니다. 이 경우 예산을 계획·설정·제안 각 1회로 제한하고, 동일한 근거·허용 값·원장 중복 검사를 적용합니다.

```powershell
.\.venv_agent_py311\Scripts\python.exe scripts\run_v11_improvement_agent.py --planner deterministic --knob ranker_family --value catboost_v1_cpu
```

`ranker_family` 비교는 전체 v11 후보와 특징을 유지하고 고정된 랭커 구성을 바꿉니다. CatBoost v1은 YetiRank·600회·depth 6·학습률 0.08·seed 42·CPU 8스레드이며 단조 제약을 사용하지 않습니다. 학습 행렬(X/정답/group 크기)과 검증 후보·특징을 기준 모델과 해시로 대조합니다. 이전 LightGBM 설정의 식별 해시는 보존해 과거 실험 중복을 다시 실행하지 않습니다.

단일 변경 검사를 위한 공통 설정에는 LightGBM 전용 `num_leaves`, `n_estimators`, `monotone_constraints`의 기준값도 남아 있습니다. CatBoost에서는 이 값을 적용하지 않고 고정 프로필을 사용합니다. 실제 학습에 적용된 값은 결과 JSON의 `training_stats.ranker_params`에 기록합니다. 따라서 이 비교는 두 고정 랭커 구성의 비교이며, 손실 함수 하나만 분리한 실험은 아닙니다.

LLM이 집계 진단과 이전 실패 결과를 보고 다음 가설을 제안하도록 실행하려면:

```powershell
.\.venv_agent_py311\Scripts\python.exe scripts\run_v11_improvement_agent.py --planner llm --max-llm-calls 2 --max-training-experiments 2 --max-proposals 1 --patience 2
```

LLM 모드는 환경변수 또는 Git 제외 `.env`의 `OPENAI_API_KEY`와 `OPENAI_EXPERIMENT_MODEL`(대체: `OPENAI_CHAT_MODEL`)을 사용합니다. 외부 API에 집계 진단·특징 설명·이전 실험 결과·예산만 전송하며, 원본 행·선박명·식별자·정답은 보내지 않도록 검사합니다. 생성 코드 실행은 허용하지 않습니다. 호출 모델과 토큰 사용량, 키를 제외한 요청 본문은 실행 폴더에 기록합니다.

한 실험은 허용된 설정 10개 중 하나를 네 고정 분할에서 비교합니다. 기준과 변경 모델은 같은 학습·검증 행, seed 42, 후보 생성 조건을 사용합니다. override 정책 비교만 기존 기준 랭커를 재사용하고 다른 설정은 재학습합니다. 특징 변환은 학습과 추론에 동일하게 적용합니다.

- 기본 분할: 선박 그룹 seed 42·123·2026, 행 seed 42. 그룹 분할에서는 학습·검증 target이 겹치지 않습니다.
- 진단: 오답 전부와, 정답 중 모델 1·2위 점수 차이가 가장 작은 50행을 선택합니다. 예측 완료 후 평가 단계에서만 정답을 결합합니다.
- 비교: 전체·한글·비한글 정확도, 가중 원점수, 후보 포함률, 새로 맞힘·틀림, seen/unseen target, 학습·예측 시간.
- 개선 판정: 모든 분할의 한글·비한글·가중 점수 비하락, 평균 가중 원점수 증가 `0.0005` 이상, 두 개 이상 분할 개선. 실행 중 판정 기준을 바꾸지 않습니다.
- 중단: 학습·계획 예산 소진, 유효한 제안 없음, 개선 없는 회차 한도 도달. 호출 수와 학습 수는 CLI로 제한합니다.
- 중복 차단: 로컬 `experiments/results/agent_improvement_ledger.json`에 이전 실행 설정·결과를 기록합니다.

결과는 새 `experiments/results/agent_v11_improve_<실행시각>_<ID>/` 폴더에 생성됩니다.

| 파일 | 내용 |
|---|---|
| `summary.md`, `summary.json` | 분할별 기준·변경 점수, 판정, 비용, 중단 이유 |
| `round_N_plan.json` | 제안 가설, 검사 결과, 실행·거부된 설정 |
| `diagnostics_report.json` | 분할별·분할 간 오류 집계 |
| `retrain_<split>_<setting>.json` | 상세 지표, 동작 검사, 예측 변화 건수 |
| `local_only/` | 모델과 행 ID가 포함된 사례·변화 기록; 로컬 전용 |
| `llm/` | LLM 모드의 요청·응답과 사용량 기록 |

랭커 교체 결과에는 조회표 적용 전·후의 `complementarity`도 기록합니다. 두 모델 모두 정답·기준만 정답·대안만 정답·둘 다 오답을 언어별·seen/unseen별로 집계합니다. `oracle_accuracy`는 정답을 알고 두 모델의 top-1 중 맞힌 쪽을 선택할 때의 진단 상한으로, 실제 실행 가능한 예측기나 앙상블 점수가 아닙니다. 전체 후보 점수를 결합하는 앙상블의 상한을 의미하지도 않습니다. CatBoost 중요도는 PredictionValuesChange, LightGBM은 gain 기준이어서 서로의 값을 직접 차감하지 않습니다.

### 검색 정책 선별 실험

```powershell
.\.venv_agent_py311\Scripts\python.exe scripts\run_v11_experiment_agent.py --planner llm
```

이 경로는 LLM이 만든 검색 정책을 **기준 랭커를 고정한 상태에서** 비교합니다. 진단 기반 재학습 실험과 평가 목적이 다릅니다. 기존 검색 정책 두 개는 후보 포함률·정확도 개선 없이 후보 수와 시간이 늘었습니다. 비한글 `english_variant`와 공백 없는 토큰 유사도 계산의 불일치는 수정됐으며, 과거 결과는 보존했습니다.

## 성능 개선 실험 기록

첫 두 진단 실행에서 재현된 기준 v11의 점수입니다. 정답 수와 반올림 전 가중 점수를 함께 기록하며, 네 분할 모두 후보 포함률은 100%였습니다.

| 분할 | 전체 정답 | 한글 정답 | 비한글 정답 | 가중 원점수 |
|---|---:|---:|---:|---:|
| group seed42 | 2,341/2,344 | 243/243 | 2,098/2,101 | 0.999572 |
| group seed123 | 2,430/2,433 | 261/261 | 2,169/2,172 | 0.999586 |
| group seed2026 | 2,400/2,402 | 214/215 | 2,186/2,187 | 0.996607 |
| row seed42 | 2,445/2,445 | 234/234 | 2,211/2,211 | 1.000000 |

첫 두 LLM 실행에서 설정 네 개를 재학습·비교했습니다. 아래 변화는 재사용 분할 네 개의 가중 원점수 변화 평균입니다. 분할 간 검증 행은 겹칠 수 있습니다.

| 변경 설정 | 평균 Δ가중 | 새로 맞힘 / 틀림 합계 | 판정 |
|---|---:|---:|---|
| `tonnage_missing_encoding=nan` | -0.000070 | 0 / 2 | 퇴보 |
| `hard_negative_top_n=50` | -0.002751 | 4 / 10 | 퇴보 |
| `num_leaves=63` | 0.000000 | 0 / 0 | 효과 없음 |
| `monotone_constraints=off` | -0.000035 | 1 / 2 | 퇴보 |

### 번호 비대칭 특징 제거 결과 (2026-10-19)

- 설정: `number_features=no_asymmetry`. `num_q_only`, `num_c_only`만 0으로 만들고 번호 일치·충돌 등 다른 특징은 유지합니다.
- 근거: 네 분할의 저점수 차이 정답 사례에서 2위 후보의 번호 불일치가 총 25회 관찰됐습니다. 중복 행이 있을 수 있으며, 이 관찰만으로 해당 특징이 오류 원인임을 단정하지 않습니다.
- 실행: 같은 네 분할에서 기준·변경 모델을 재학습했습니다. 결정적 계획 1회·학습 설정 1개, 외부 LLM 호출 0회로 완료했습니다. 네 기준 모델은 앞선 실행과 해시가 같았습니다.
- 실행 ID: `agent_v11_improve_20261019_062733_04f948`. 실행 전 가설은 해당 폴더의 `pre_run_intent.json`에 기록했습니다.

| 분할 | 전체 정답 기준 → 변경 | 변경 모델 한글 정답 | 변경 모델 비한글 정답 | 변경 가중 원점수 | Δ가중 |
|---|---:|---:|---:|---:|---:|
| group seed42 | 2,341 → 2,340 / 2,344 | 243/243 | 2,097/2,101 | 0.999429 | -0.000143 |
| group seed123 | 2,430 → 2,430 / 2,433 | 261/261 | 2,169/2,172 | 0.999586 | 0.000000 |
| group seed2026 | 2,400 → 2,401 / 2,402 | 214/215 | 2,187/2,187 | 0.996744 | +0.000137 |
| row seed42 | 2,445 → 2,445 / 2,445 | 234/234 | 2,211/2,211 | 1.000000 | 0.000000 |

새로 맞힌 행과 틀린 행은 각 1건입니다. seed2026의 톤수 관련 랭킹 오류 1건을 고쳤지만 seed42에서 동일 이름 후보 간 오류가 1건 생겼습니다. 한글 정확도·후보 포함률·후보 수는 같았습니다. 평균 Δ가중은 약 `-0.0000014`이며 사전 판정은 `regressed`, **미채택**입니다. 이 미세한 평균 변화만으로 일반 성능 저하를 주장하지 않으며, 일관된 개선 근거가 없다는 판단입니다.

네 분할 모두 특징 0 처리·랭커 내 비사용 검사와 표본 행 독립성 검사를 통과했습니다. 기준 학습·변경 학습은 분할마다 별도로 수행했습니다. 결과는 [요약 Markdown](experiments/results/agent_v11_improve_20261019_062733_04f948/summary.md)과 [상세 JSON](experiments/results/agent_v11_improve_20261019_062733_04f948/summary.json)에 있습니다. 결과 폴더는 Git 제외 로컬 산출물입니다. 기준 v11과 기존 제출 ZIP을 유지했습니다.

### v11 특징 고정 CatBoost 랭커 비교 결과 (2026-10-19)

실행 ID는 `agent_v11_improve_20261019_072054_425152`입니다. 위의 고정 CatBoost v1 설정을 같은 네 분할에서 실제 학습·평가했습니다. 네 기준 LightGBM 모델은 앞선 실행과 모델 해시가 같았습니다. 학습 행렬·검증 후보·검증 특징은 두 랭커에서 모두 해시가 일치했고, CatBoost 600회 학습·저장 모델 재로딩·표본 행 독립성 검사도 통과했습니다. LLM 호출은 0회입니다.

| 분할 | CatBoost 전체 정답 | 한글 정답 | 비한글 정답 | 가중 원점수 | Δ가중 | 새로 맞힘 / 틀림 |
|---|---:|---:|---:|---:|---:|---:|
| group seed42 | 2,340/2,344 | 243/243 | 2,097/2,101 | 0.999429 | -0.000143 | 0 / 1 |
| group seed123 | 2,430/2,433 | 261/261 | 2,169/2,172 | 0.999586 | 0.000000 | 1 / 1 |
| group seed2026 | 2,400/2,402 | 214/215 | 2,186/2,187 | 0.996607 | 0.000000 | 0 / 0 |
| row seed42 | 2,444/2,445 | 234/234 | 2,210/2,211 | 0.999864 | -0.000136 | 0 / 1 |

분할별 합계로 새로 맞힘 1건·틀림 3건이며, 중복을 제외한 고유 행은 각각 1개·2개입니다. 한글 정확도·후보 포함률·후보 수는 모두 같았습니다. 평균 Δ가중은 약 `-0.00006962`로 사전 판정 `regressed`, **CatBoost 단독 구성은 미채택**입니다. 행 분할의 처음 보는 선박 정확도도 `674/674 → 673/674`로 낮아졌습니다.

| 분할 | LightGBM만 정답 | CatBoost만 정답 | 둘 다 오답 |
|---|---:|---:|---:|
| group seed42 | 1 | 0 | 3 |
| group seed123 | 1 | 1 | 2 |
| group seed2026 | 0 | 0 | 2 |
| row seed42 | 1 | 0 | 0 |

CatBoost가 기준의 일부 오류를 고쳤지만 다른 오류가 늘었습니다. 이 실행에서 앙상블은 평가하지 않았으며, 두 모델이 다르게 맞힌다는 사실만으로 앙상블 개선을 주장하지 않습니다. 네 분할 학습 시간 합계는 LightGBM 약 416초·CatBoost 약 1,397초, 검증 예측 합계는 약 89.8초·88.1초였습니다. 학습 시간에는 특징 생성이 포함되고, 예측 시간은 모델 로딩·조회표 준비를 제외한 검증 예측 구간이므로 전체 제출 실행 시간과 구분합니다.

자세한 결과는 [실험 요약](experiments/results/agent_v11_improve_20261019_072054_425152/summary.md)과 [상세 JSON](experiments/results/agent_v11_improve_20261019_072054_425152/summary.json)에 있습니다. 모두 재사용 분할의 탐색 결과이며 기준 모델과 보관 ZIP을 유지했습니다.

## 기존 EXP 노트북 실행 순서

아래는 v11 병합 이전 EXP-014 기준 재현부터 EXP-021까지의 보존된 실험입니다.

| 노트북 | 내용 |
|---|---|
| `notebooks/01_EXP014_재현과_오류분석.ipynb` | EXP-014 재현 및 오류 분석 |
| `notebooks/02_EXP014_비한글_fuzzy_후보검색.ipynb` | 비한글 입력의 fuzzy 후보 검색 비교 |
| `notebooks/03_EXP014_한글_후보검색폭_실험.ipynb` | 한글 미포함 사례와 후보 검색 폭 비교 |
| `notebooks/04_EXP017_LightGBM_LambdaRank.ipynb` | 후보·특징을 고정한 LambdaRank 재랭커 비교 |
| `notebooks/05_EXP018_CatBoost_Ranker.ipynb` | 같은 후보·특징으로 CatBoost YetiRankPairwise 재랭커 비교 |
| `notebooks/06_EXP020_LGBM_CatBoost_Ensemble.ipynb` | query 내부 순위 점수로 두 랭커를 결합하는 세 가지 비율 비교 |
| `notebooks/07_EXP021_LightGBM_num_leaves_단일파라미터_실험.ipynb` | LightGBM `num_leaves` 하나를 직접 바꿔 세 seed 비교 |

프로젝트 루트를 작업 폴더로 열고 저장소에 준비된 Python 환경에서 노트북을 위 순서로 실행합니다. 각 노트북의 첫 셀에 필요한 모듈과 실행 조건이 안내되어 있습니다. 결과 파일을 생성하는 셀은 기존 결과를 덮어쓰지 않고 실행별 결과 폴더에 기록하도록 구성되어 있습니다.

## EXP-017 결과

EXP-017은 EXP-016의 후보 생성(`limit=50`, `cutoff=50`, `max_candidates=70`)과 기존 7개 특징을 그대로 사용해 규칙 기반 재랭커와 LightGBM LambdaRank를 비교했습니다.

| seed | EXP-016 weighted | LambdaRank weighted | weighted 변화 | 판정 |
|---:|---:|---:|---:|---|
| 42 | 0.725581 | 0.783337 | +0.057756 | 개선 |
| 123 | 0.738423 | 0.727557 | -0.010866 | 하락 |
| 2026 | 0.719228 | 0.761554 | +0.042326 | 개선 |

seed 간 결과가 일관되지 않아 LambdaRank는 채택하지 않았고 새 제출 파일도 만들지 않았습니다. 세부 지표와 검증 산출물은 [`experiments/results/exp017_lambdarank_20261015_204555/`](experiments/results/exp017_lambdarank_20261015_204555/)에 있습니다.

## EXP-018 CatBoost 결과

EXP-016 후보·특징을 고정한 CatBoost `YetiRankPairwise` 비교에서 weighted 점수는 세 seed 모두 올랐고, 반복 학습 예측도 일치했습니다. 다만 비한글 정확도가 seed 42에서 `0.001904`, seed 123에서 `0.002763` 하락해 사전 통과 조건을 충족하지 못했습니다. 따라서 채택하지 않았고 제출 zip도 만들지 않았습니다. 결과 표와 판단 근거는 [`experiments/LOG.md`](experiments/LOG.md)에 기록했습니다.

## 과거 앙상블 검토 기록

현재 가상환경에 XGBoost가 없어 설치 없이 해당 실험은 건너뛰었습니다. LightGBM·CatBoost 후보 순위 앙상블 세 비율을 평가했지만, 모두 seed 123의 비한글 정확도 하락으로 세 seed 통과 기준을 만족하지 못해 미채택했습니다. seed별 점수와 행·후보 진단은 [`experiments/results/exp020_lgbm_catboost_ensemble_20261015_234914/`](experiments/results/exp020_lgbm_catboost_ensemble_20261015_234914/)에 있습니다.

## 주요 폴더

- `agent/`: 단일 기록 검토, 실험 그래프, 허용 설정과 학습·추론 연결
- `agent/team_v11/source/`: 팀원 v11 코드 사본
- `data/open/`: 공식 데이터, Git 제외·원본 보존
- `team_v11_artifacts/`: 학습된 로컬 모델·조회표, Git 제외
- `notebooks/`: 실험 및 검증 노트북
- `scripts/`: 노트북에서 사용하는 재사용 로직
- `experiments/LOG.md`: 실험 이력과 채택 판단
- `experiments/results/`: 실행별 비교표와 진단 결과
- `submissions/`: 버전별 제출 패키지 보관

실험 기록은 로컬 작업 기준이며, 커밋·푸시 및 실제 제출 여부와 구분해 관리합니다.

## 검증과 제출 경계

```powershell
.\.venv_agent_py311\Scripts\python.exe -m unittest discover -s tests -v
# 실제 CatBoost 학습·저장·재로딩 검사는 대회용 환경에서 실행
.\.venv_competition_py311\Scripts\python.exe -m unittest tests.test_catboost_ranker_experiment -v
```

단위 검사는 예산·중복 차단·진단 집계·설정 검사 등을 확인합니다. 실제 모델 점수는 별도 실험 산출물로 확인합니다. 실험 작업자는 검증 정답 제거, 학습 범위의 조회표, 일부 행의 단독·역순 부분집합 예측 일치를 검사하며, 이는 전체 행 독립성 전수 검사를 의미하지 않습니다.

제출 추론은 Python 3.11의 행별 오프라인 실행을 따릅니다. LLM·LangGraph와 실험용 코드·원본 train은 제출 패키지에 포함하지 않습니다. 새 패키지 설치·데이터 다운로드·리더보드 제출은 사용자 승인 범위에 따라 수행합니다. 실제 제출 전에는 ZIP 구조·행 독립성·오프라인 실행·실행 시간을 별도로 확인해야 합니다.
