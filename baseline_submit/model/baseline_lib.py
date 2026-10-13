"""Small, row-independent candidate retrieval and RandomForest features.

Only the official vessel catalog is indexed. Every query row is processed alone;
no feature depends on another evaluation row or on the evaluation distribution.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from heapq import nlargest
import unicodedata

import numpy as np
import pandas as pd


FEATURE_NAMES = (
    "name_exact",
    "name_bigram_jaccard",
    "name_trigram_jaccard",
    "name_prefix_ratio",
    "name_length_ratio",
    "gross_tonnage_gap",
    "international_gross_tonnage_gap",
    "best_tonnage_gap",
    "query_tonnage_missing",
    "catalog_tonnage_missing",
)
QUERY_COLUMNS = (
    "sample_id",
    "vessel_name",
    "gross_tonnage",
    "international_gross_tonnage",
)

# 이름 정규화 
# 선박 이름에 공백, 특수문자, 대소문자 차이가 있더라도 같은 이름으로 비교할 수 있도록 전처리하는 역할
def normalize_name(value: object) -> str:
    """Normalize spelling and spacing without converting Korean to English."""
    # 문자열인지 검사
    if not isinstance(value, str):
        return ""
    # 유니코드 정규화 + 대문자 변환
    # NFKC : 공백을 제거할 뿐만 아니라 일반적인 형태로 변환 -> 일부 문자 간 차이가 사라질 수 있음
    value = unicodedata.normalize("NFKC", value).upper()
    
    # 공백, 특수문자 제거
    # ch for ch in value in ch.isalnum() : 문자열에서 문자를 꺼내면서 영문자, 한글 등 문자 또는 숫자에 해당하는 것만 남김
    return "".join(ch for ch in value if ch.isalnum())

# 문자열을 일정한 길이로 잘라서 중복 없이 저장하는 함수
# 선박 이름의 유사도를 비교하기 위해 사용
# value : 분리할 문자열, width : 문자열을 몇 글자로 자를지
def grams(value: str, width: int) -> frozenset[str]:
    # 기준 글자수가 문자열보다 크면
    if len(value) < width:
        # frozenset() : 잘라낸 문자열들의 중복을 제거하고 수정할 수 없는 집합으로 만드는 역할
        return frozenset((value,)) if value else frozenset()
    # value[i : i + width] : 문자열을 width 길이만큼 잘라내는 역할
    return frozenset(value[i : i + width] for i in range(len(value) - width + 1))


def positive_float(value: object) -> float | None:
    # 자료형 변환 시도
    try:
        number = float(value)
    # 오류 나면 None 반환
    except (TypeError, ValueError):
        return None
    # np.isfinite() : NumPy에서 제공하는 함수로, 숫자가 유한한 값인지 확인하는 함수
    # 숫자가 무한대나 숫자가 아닌지 검사 (음수와 0도 유한한 숫자이므로 True 반환)
    return number if np.isfinite(number) and number > 0 else None

# left 또는 right가 None이면 2.0 반환
# 둘 다 값이 있으면 left에서 right을 뺀 절대값 / left, right 중 큰 값(같으면 1.0) 계산
# 계산된 값 중 2.0보다 작으면 계산된 값 반환, 2.0이 더 작으면 2.0 반환
def relative_gap(left: float | None, right: float | None) -> float:
    if left is None or right is None:
        return 2.0
    return min(abs(left - right) / max(left, right, 1.0), 2.0)


# 중복 제거 후 수정 못하는 집합으로 만들어진 left, right
def jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    if not left or not right:
        return 0.0
    # left와 right의 합집합의 길이 / left와 right의 교집합의 길이
    return len(left & right) / len(left | right)


class CatalogIndex:
    """Immutable lookup structures built from the official public catalog."""

    def __init__(self, catalog: pd.DataFrame) -> None:
        needed = {"vessel_id", "vessel_name", "gross_tonnage"}
        # issubset : 집합에서 사용하는 메서드, 한 집합의 모든 요소가 다른 집합에 포함되어 있는지 확인하는 함수
        # needed에 있는 요소들이 catalog에도 있는지 확인
        if not needed.issubset(catalog.columns):
            # 없으면 ValueError
            raise ValueError(f"Catalog is missing: {sorted(needed - set(catalog.columns))}")
        # catalog의 인덱스를 다시 등록
        self.catalog = catalog.reset_index(drop=True)
        # vessel_id가 유일하지 않으면
        if not self.catalog["vessel_id"].is_unique:
            # ValueError 발생
            raise ValueError("vessel_id must be unique in the catalog")

        # vessel_id의 자료형을 str로 바꾸로 리스트로 만듦
        self.ids = self.catalog["vessel_id"].astype(str).tolist()
        # 리스트를 반복하면서 요소의 인덱스 = i, 와 값 = vessel_id를 가져옴
        # vessel_id를 key로 i를 value로 딕셔너리 저장
        self.id_to_index = {vessel_id: i for i, vessel_id in enumerate(self.ids)}
        # catalog의 vessel_name을 정규화해서 리스트로 저장
        self.names = [normalize_name(v) for v in self.catalog["vessel_name"]]
        # bigram : 문자열을 연속된 2글자 단위로 나누는 방법
        # 정규화된 vessel_name을 연속된 2개의 문자로 나누기
        self.bigram_sets = [grams(v, 2) for v in self.names]
        # trigram : 문자열을 연속된 3글자 단위로 나누는 방법
        # 정규화된 vessel_name을 연속된 3개의 문자로 나누기
        self.trigram_sets = [grams(v, 3) for v in self.names]
        # catalog의 gross_tonnage를 숫자형 자료형(int, float 등)으로 변환
        # 숫자로 변환할 수 없다면 coerece로 변환
        # 숫자형 자료형으로 변환됐다면 float 타입으로 통일
        # to_numeric을 사용하는 이유는 숫자로 변환할 수 없는 데이터가 있어도 오류 없이 처리하기 위함
        self.gt = pd.to_numeric(self.catalog["gross_tonnage"], errors="coerce").to_numpy(dtype=float)

        # dict[str, list[int]] : 딕셔너리의 자료형을 설명하는 타입 힌트
            # str : 딕셔너리의 key값
            # list[int] : 딕셔너리의 Value는 정수로 이루어진 리스트
            # : 는 변수에 어떤 자료형이 들어갈 예정인지 표시하는 문법
        # defalutdict : 존재하지 않는 Key에 접근하면 기본값을 자동으로 생성하는 딕셔너리
            # defaultdict[list] : 존재하지 않는 Key에 접근하면 빈 리스트를 자동으로 생성
            # ex) dict["BLUESTAR"].append(0)일 때 BLUESTAR가 존재하지 않는다면 [] 생성

        # 선박 이름을 Key로, 해당 선박의 인덱스 목록을 Value로 저장
        # 이름이 같은 선박이 여러 개 있을 수 있으므로 list[int]로 저장
        self.exact_name: dict[str, list[int]] = defaultdict(list)

        # Trigram을 Key로, 해당 Trigram을 포함한 선박의 인덱스 목록을 Value로 저장
        # 3글자 조합으로 해당 조합을 포함하는 선박 인덱스 목록을 저장하고 이름이 비슷한 선박 후보를 검색
        self.trigram_postings: dict[str, list[int]] = defaultdict(list)

        for i, name in enumerate(self.names):
        # 정규화한 name이 있으면 
            if name:
                # exact_name 딕셔너리에 인덱스 번호 저장
                self.exact_name[name].append(i)
            # trigram_sets 딕셔너리에 gram이 있으면 trigram_postings에 인덱스 번호 저장
            for gram in self.trigram_sets[i]:
                self.trigram_postings[gram].append(i)

        # np.flatnonzero : 배열에서 0이 아닌 값의 인덱스(위치)를 찾아 반환하는 함수
        # gt가 유한한 수이고 0보다 크면 인덱스를 반환
        valid = np.flatnonzero(np.isfinite(self.gt) & (self.gt > 0))
        # np.argsort : 배열의 값을 정렬했을 때 각 값이 원래 어느 인덱스에 있었는지 반환하는 함수
        # gt의 키(valid)의 value로 오름차순 정렬하고 각 값이 어느 인덱스에 있었는지 반환
        # kind = "mergesort" : 정렬 알고리즘을 지정하는 옵션
            # 안정정렬 : 값이 같은 요소들의 기존 순서 유지
        ordering = np.argsort(self.gt[valid], kind="mergesort")

        # 선박의 총톤수를 오름차순으로 정렬했을 때 원본 데이터 인덱스를 저장
        self.sorted_gt_indices = valid[ordering]
        # 유효한 선박의 총톤수를 오름차순으로 정렬한 값을 저장
        self.sorted_gt_values = self.gt[self.sorted_gt_indices]

    # 입력한 선박의 총톤수와 가장 가까운 총톤수를 가진 선박을 최대 count개까지 찾아
    # 해당 선박들의 원본 인덱스를 리스트로 반환하기 위해 만들어진 함수
    def _nearest_gt(self, value: float | None, count: int) -> list[int]:
        # value가 None이거나 sorted_gt_values에 값이 없으면
        if value is None or len(self.sorted_gt_values) == 0:
            # 빈 리스트 반환
            return []
        # np.searchsorted : 정렬된 배열에 새로운 값을 넣었을 때 오름차순이 유지되는 삽입 위치를 찾는 함수 
        # sorted_gt_values에 value를 넣었을 때 오름차순이 유지되는 위치의 인덱스 값을 반환
        pos = int(np.searchsorted(self.sorted_gt_values, value))
        # 그 위치의 왼쪽과 오른쪽
        left, right = pos - 1, pos
        found: list[int] = []
        # found에 들어있는 값이 count보다 적고 left가 0보다 크거나 sorted_gt_values의 개수가 right보다 작을 때까지
        while len(found) < count and (left >= 0 or right < len(self.sorted_gt_values)):
            # right가 sorted_gt_values의 개수 이상일 때
            if right >= len(self.sorted_gt_values):
                # choose_left에 True 저장
                choose_left = True
            # left가 0보다 작으면
            elif left < 0:
                choose_left = False
            # sorted_gt_values의 left의 값에서 value를 뺀 값의 절댓값이 sorted_gt_values의 right값에서 value를 뺀 값의 절댓값보다 이하일 때
            # choose_left = True
            else:
                choose_left = abs(self.sorted_gt_values[left] - value) <= abs(
                    self.sorted_gt_values[right] - value
                )

            # choose_left가 true이면 left
            # sorted_gt_indices의 left의 값을 int로 변환하고
            # found에 저장
            found.append(int(self.sorted_gt_indices[left if choose_left else right]))
            # choose_left이 True-> 1
            # choose_left이 false -> 0
            left -= int(choose_left)
            # choose_left이 True-> not True -> false -> 0
            # choose_left이 false -> not false -> true -> 1
            right += int(not choose_left)
        return found

    def candidates(
        self,
        row: dict,
        name_limit: int = 16,
        tonnage_limit: int = 12,
        max_candidates: int = 40,
    ) -> list[int]:
        """Retrieve likely catalog rows from one query row only."""

        # row에서 vessel_name 값을 가져와서 정규화
        query_name = normalize_name(row.get("vessel_name"))
        # row에서 gross_tonnage/international_gross_tonnage 값 가져와서 형 변환하고 유한한 값이면 True
        gross = positive_float(row.get("gross_tonnage"))
        international = positive_float(row.get("international_gross_tonnage"))
        output: list[int] = []
        used: set[int] = set()

        # 최대 후보수보다 output에 저장된 후보 개수가 작고 used에 index가 없으면(중복제거)
        def append(indices: list[int]) -> None:
            for index in indices:
                if index not in used and len(output) < max_candidates:
                    # output에 index 저장
                    output.append(index)
                    # used에 index 저장
                    used.add(index)

        # 정규화된 vessel_name에 대해서
        if query_name:
            # exact_name에서 query_name을 가져오되, 없으면 빈 리스트 저장
            exact = self.exact_name.get(query_name, [])
            # 이름이 정확히 일치하는 선박 후보(exact)를 총톤수 차이가 작은 순서로 정렬
            # 차이가 같으면 인덱스가 작은 순서로 정렬
            # 최대 name_limit까지 output에 중복없이 저장
            append(sorted(exact, key=lambda i: (min(relative_gap(gross, positive_float(self.gt[i])),
                                                     relative_gap(international, positive_float(self.gt[i]))), i))[:name_limit])
            # 선백 이름을 trigram한 문자열이 저장
            counts: Counter[int] = Counter()

            # trigram으로 계산 후 
            for gram in grams(query_name, 3):
                # counts에 trigram_postings에 gram이 있으면 업데이트, 없으면 빈 집합 저장
                counts.update(self.trigram_postings.get(gram, ()))
            # nlargest : 데이터에서 가장 큰 값 N개를 찾아 리스트로 반환
            # Trigram이 많이 일치하는 후보를 최대 name_limit개만큼 찾기 위함
            near_names = nlargest(name_limit, counts, key=lambda i: (counts[i], -i))
            # 리스트 값 중복 제거
            append(near_names)

        # gross와 가장 가까운 선박을 최대 tonnage_limit개까지 찾아 
        # 해당 선박들의 원본 인덱스를 리스트로 반환하고 append로 중복 제거
        append(self._nearest_gt(gross, tonnage_limit))
        # international이 gross가 아니라면
        if international != gross:
            append(self._nearest_gt(international, tonnage_limit))
        # output이 아니라면
        if not output:
            output.append(0)
        return output


    # 입력된 선박과 후보 선박이 얼마나 비슷한지 비교하고 결과를 10개의 숫자로 만드는 것
    # RandomForest 모델이 두 선박이 같은 선박인지 판단하는 데 사용
    def pair_features(self, row: dict, candidate_index: int) -> list[float]:
        # 입력 선박 정보 : query_name, gross, international
        # 후보 선박 정보 : catalog_name, catalog_gt
        query_name = normalize_name(row.get("vessel_name"))
        catalog_name = self.names[candidate_index]
        gross = positive_float(row.get("gross_tonnage"))
        international = positive_float(row.get("international_gross_tonnage"))
        catalog_gt = positive_float(self.gt[candidate_index])

        # 일반 총톤수 차이
        gross_gap = relative_gap(gross, catalog_gt)
        # 국제 총톤수 차이
        international_gap = relative_gap(international, catalog_gt)
        # 0에 가까울수록 총톤수가 비슷하다


        # 문자열의 [0]부터 같으면 prefix를 1씩 증가
        # 처음부터 연속해서 일치하는 글자 수
        prefix = 0

        for left, right in zip(query_name, catalog_name):
            # 다르면 바로 break
            if left != right:
                break
            prefix += 1

        
        return [
            # 두 이름이 비어 있지 않으면서 정확히 같으면 1.0, 그렇지 않으면 0.0
            float(bool(query_name) and query_name == catalog_name),

            # prefix를 이름 길이로 나눔 (0.75가 나오면 두 이름이 75% 일치한다)
            jaccard(grams(query_name, 2), self.bigram_sets[candidate_index]),
            jaccard(grams(query_name, 3), self.trigram_sets[candidate_index]),
            prefix / max(len(query_name), len(catalog_name), 1),

            # 두 이름의 길이가 얼마나 비슷한지 계산
            # 이름 길이가 같으면 1.0 반환되지만 실제 이름도 같다는건 아님
            min(len(query_name), len(catalog_name)) / max(len(query_name), len(catalog_name), 1),
            gross_gap,
            international_gap,

            # 일반 총톤수(gross_gap)와 국제총톤수(international_gap) 중 후보 선박의 총톤수에 더 가까운 값을 선택
            min(gross_gap, international_gap),

            # 입력 선박의 일반 총톤수와 국제총톤수가 둘 다 없는지 확인(있으면 1.0, 없으면 0.0)
            float(gross is None and international is None),
            float(catalog_gt is None),
        ]
        # 이 반환값이 RandomForest 모델에 전달되며, 모델이 이를 바탕으로 두 선박이 같은 선박일 확률을 예측하는 데 활용

# 모델 학습 데이터 생성 -> 선박 ID 예측 -> 한글 이름 판별 -> 모델 성능 평가

# 학습 데이터 생성
# 입력 선박과 후보 선박을 비교해 RandomForest 학습에 사용할 특성(X)과 정답(y)을 만드는 함수
def make_training_pairs(
    train: pd.DataFrame, index: CatalogIndex, max_negatives: int = 20
) -> tuple[np.ndarray, np.ndarray, float]:
    """Positive pair plus retrieved negative pairs for every official Train row."""
    if "target_vessel_id" not in train.columns:
        raise ValueError("Training data needs target_vessel_id")
    
    features: list[list[float]] = [] # 모델 입력 특정(X)
    labels: list[int] = [] # 정답 레이블 (y)
    retrieved = 0 # 정답 선박이 후보에 포함된 횟수

    # true_id : 실제 정답 선박 ID
    # true_index : 정답 선박의 카탈로그 인덱스
    # candidates : 검색된 후보 선박 목록
    # negatives : 정답이 아닌 후보 선박 (최대 20개)
    for row in train.to_dict("records"): 
        true_id = row["target_vessel_id"]
        if true_id not in index.id_to_index:
            raise ValueError(f"Training target not in catalog: {true_id}")
        
        true_index = index.id_to_index[true_id]
        candidates = index.candidates(row)
        retrieved += int(true_index in candidates)
        negatives = [i for i in candidates if i != true_index][:max_negatives]

        # 정답 선박은 1, 오답 선박은 0
        for candidate_index in [true_index] + negatives:
            features.append(index.pair_features(row, candidate_index))
            labels.append(int(candidate_index == true_index))

    return (
        np.asarray(features, dtype=np.float32),
        np.asarray(labels, dtype=np.int8),
        retrieved / len(train),
    )

# 선박 ID 예측
# 학습된 모델을 이용해 각 입력 선박에 가장 적합한 후보 선박 ID 예측 함수
# batch_size = 200: 200개씩 입력 데이터 나눠 처리
# index.candidates(row) : 후보 선박 검색
# pair_features() : 각 후보의 특성 생성
# model.predict_proba() : 정답 선박일 확률 예측
# np.argmax() : 확률이 가장 높은 후보 선택

def predict_ids(
    model, queries: pd.DataFrame, index: CatalogIndex, batch_size: int = 200
) -> list[str]:
    """Predict one catalog ID per row without consulting other query rows."""
    missing = set(QUERY_COLUMNS) - set(queries.columns)
    if missing:
        raise ValueError(f"Query data is missing: {sorted(missing)}")
    predictions: list[str] = []
    for start in range(0, len(queries), batch_size):
        rows = queries.iloc[start : start + batch_size].to_dict("records")
        features: list[list[float]] = []
        candidate_lists: list[list[int]] = []
        for row in rows:
            candidates = index.candidates(row)
            candidate_lists.append(candidates)
            features.extend(index.pair_features(row, i) for i in candidates)
        matrix = np.asarray(features, dtype=np.float32)
        scores = model.predict_proba(matrix)[:, 1] # 각 후보가 정답 클래스(1)에 속할 확률을 가져옴
        offset = 0
        for candidates in candidate_lists:
            local = scores[offset : offset + len(candidates)]
            predictions.append(index.ids[candidates[int(np.argmax(local))]])
            offset += len(candidates)
    return predictions # 예측한 선박 ID 리스트

# 한글 선박 이름 판별
# 선박 이름에 한글 완성형 글자가 하나라도 있는지 확인
def is_hangul_name(value: object) -> bool:
    return isinstance(value, str) and any("가" <= char <= "힣" for char in value)

# 모델 성능 평가
# 한글 선박과 비한글 선박의 예측 정확도를 각각 계산하고 적용해 최종 정확도를 반환
def local_weighted_accuracy(truth: pd.DataFrame, predicted_ids: list[str]) -> dict[str, float]:
    if len(truth) != len(predicted_ids):
        raise ValueError("Prediction count differs from validation data")
    language = truth["vessel_name"].map(is_hangul_name).to_numpy(dtype=bool)
    correct = truth["target_vessel_id"].to_numpy() == np.asarray(predicted_ids)
    if not language.any() or language.all():
        raise ValueError("Validation must contain both language groups")
    hangul = float(correct[language].mean())
    non_hangul = float(correct[~language].mean())
    return {
        "hangul_accuracy": hangul,
        "non_hangul_accuracy": non_hangul,
        "weighted_accuracy": 0.7 * hangul + 0.3 * non_hangul,
    }
