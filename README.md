# Port-Problem-Solving

## 팀원 v11 모델과 검토 에이전트

팀원 실험의 v11 LightGBM 모델을 기본 추천으로 연결하고 기존 EXP-014 매칭을
비교용으로 보존하는 로컬 통합 코드는 [`agent/team_v11/README.md`](agent/team_v11/README.md)에
기록했다. 모델은 공식 데이터로 별도 학습하며 학습 산출물은 Git에서 제외한다.
이 통합은 보관 중인 제출 ZIP이나 리더보드 제출을 변경하지 않는다.

항만 문제해결 AI 기술 공모전의 선박명 매칭 실험 프로젝트입니다. 한글·비한글 선박명을 카탈로그의 `vessel_id`와 연결하며, 후보 검색과 재랭킹을 단계별로 비교합니다.

## 현재 기준 및 제출 상태

- 기준 실험: EXP-014 `fixed_40_50_60`
- 최고 확인 리더보드 점수: EXP-014 `0.6763333333`
- EXP-016 `final_exp016_korean_limit50.zip` 리더보드 점수: 사용자 보고 `0.674` (EXP-014보다 낮음)
- 최신 재랭커 실험: EXP-020 LGBM·CatBoost 순위 앙상블 (세 seed 평가 완료, 기준 게이트 미통과)
- EXP-017은 seed 123에서 기준보다 성능이 하락해 채택하지 않았습니다. 후보 생성과 특징은 고정했으며, 재현성은 확인했지만 세 seed 통과 조건을 만족하지 못했습니다.
- 현재 최고점 기준 제출 버전은 EXP-014입니다. EXP-016은 로컬 검증 후 별도 제출했지만 리더보드 점수가 더 낮았습니다. 실제 제출 관련 결정은 사용자 확인을 따릅니다.

실험별 설정, 점수, 해석 및 채택 여부는 [`experiments/LOG.md`](experiments/LOG.md)에 기록합니다. 실험 산출물은 `experiments/results/` 아래 별도 폴더에 보존합니다.

## 노트북 실행 순서

아래 노트북은 EXP-014 기준 재현에서 후보 검색 실험과 최신 재랭커 비교까지 이어집니다.

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

## 다음 모델 검토

현재 가상환경에 XGBoost가 없어 설치 없이 해당 실험은 건너뛰었습니다. LightGBM·CatBoost 후보 순위 앙상블 세 비율을 평가했지만, 모두 seed 123의 비한글 정확도 하락으로 세 seed 통과 기준을 만족하지 못해 미채택했습니다. seed별 점수와 행·후보 진단은 [`experiments/results/exp020_lgbm_catboost_ensemble_20261015_234914/`](experiments/results/exp020_lgbm_catboost_ensemble_20261015_234914/)에 있습니다.

## 주요 폴더

- `notebooks/`: 실험 및 검증 노트북
- `scripts/`: 노트북에서 사용하는 재사용 로직
- `experiments/LOG.md`: 실험 이력과 채택 판단
- `experiments/results/`: 실행별 비교표와 진단 결과
- `submissions/`: 버전별 제출 패키지 보관

실험 기록은 로컬 작업 기준이며, 커밋·푸시 및 실제 제출 여부와 구분해 관리합니다.
