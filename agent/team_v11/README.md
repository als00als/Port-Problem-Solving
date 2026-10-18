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

별도 개발 환경은 `.venv_agent_py311/`이며, 직접 필요한 패키지는
`requirements-agent-py311.txt`에 고정했다. 이 환경은 대회용
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
요약은 `experiments/LOG.md`에 덧붙인다. 설정 생성은 규칙 기반이며 LLM을
사용하지 않는다. 기존 target 그룹은 과거 실험에 노출되어 두 분할의 점수는
탐색용이다. 그래프는 제출 ZIP이나 리더보드를 변경하지 않는다.
