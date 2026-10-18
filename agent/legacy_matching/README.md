# 기존 매칭 서비스 사본

`submit/`은 Git 제외 대상이므로 기존 에이전트가 `submit/model/`을 직접
참조하면 새 체크아웃에서 실행할 수 없다. 이 폴더는 현재 로컬
`submit/model/{adaptive_lib,matching_service,record_validation}.py`를
추적 가능한 위치로 복사한 것이다. `adaptive_lib.py`와
`record_validation.py`는 원본과 바이트 단위로 같고,
`matching_service.py`에는 결과에 모델 이름과 예측 출처를 표시하는
메타데이터 필드만 추가했다. 기존 제출 ZIP과 `submit/`은 수정하지 않았다.

통합 서비스에서는 이 모델을 비교·추가 검색·후보 비교에만 사용한다.
기본 추천은 `agent/team_v11/`의 팀원 v11 모델이 만든다.
