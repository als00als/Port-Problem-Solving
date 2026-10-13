import pandas as pd

from difflib import SequenceMatcher
import math

import pandas as pd

from src.normalize import normalize_name


def positive_float(value):
    """양수인 숫자만 float으로 변환합니다."""
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None

    if not math.isfinite(number) or number <= 0:
        return None

    return number


def relative_gap(left, right) -> float:
    """두 톤수의 상대적인 차이를 계산합니다."""
    left = positive_float(left)
    right = positive_float(right)

    if left is None or right is None:
        return 2.0

    return min(abs(left - right) / max(left, right, 1.0), 2.0)


def name_similarity(left: str, right: str) -> float:
    """두 선박명의 문자열 유사도를 계산합니다."""
    if not left or not right:
        return 0.0

    return SequenceMatcher(None, left, right).ratio()


def prefix_similarity(left: str, right: str) -> float:
    """두 이름의 앞부분이 얼마나 같은지 계산합니다."""
    if not left or not right:
        return 0.0

    length = min(len(left), len(right))
    same_count = 0

    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        same_count += 1

    return same_count / length if length else 0.0


def length_similarity(left: str, right: str) -> float:
    """두 이름의 길이 유사도를 계산합니다."""
    if not left or not right:
        return 0.0

    return min(len(left), len(right)) / max(len(left), len(right))


def tonnage_similarity(row, catalog_row) -> float:
    """
    입력 행의 톤수와 카탈로그 행의 gross_tonnage 유사도를 계산합니다.

    입력 행에 gross_tonnage와 international_gross_tonnage가
    둘 다 있으면 더 가까운 값을 사용합니다.
    """
    catalog_tonnage = catalog_row.get("gross_tonnage")

    query_tonnages = [
        row.get("gross_tonnage"),
        row.get("international_gross_tonnage"),
    ]

    gaps = []

    for query_tonnage in query_tonnages:
        if positive_float(query_tonnage) is not None:
            gaps.append(relative_gap(query_tonnage, catalog_tonnage))

    if not gaps:
        return 0.0

    best_gap = min(gaps)

    # 상대 차이 0이면 1.0, 차이가 커질수록 0에 가까워집니다.
    return max(0.0, 1.0 - min(best_gap, 1.0))


def calculate_candidate_score(row, catalog_row) -> dict:
    """입력 행과 카탈로그 후보 하나의 특징과 최종 점수를 계산합니다."""
    query_name = normalize_name(row.get("vessel_name"))
    catalog_name = normalize_name(catalog_row.get("vessel_name"))

    exact_match = float(
        bool(query_name) and query_name == catalog_name
    )

    name_ratio = name_similarity(query_name, catalog_name)
    prefix_ratio = prefix_similarity(query_name, catalog_name)
    length_ratio = length_similarity(query_name, catalog_name)
    tonnage_ratio = tonnage_similarity(row, catalog_row)

    # 첫 번째 규칙 기반 재랭킹 가중치
    score = (
        exact_match * 0.40
        + name_ratio * 0.30
        + prefix_ratio * 0.10
        + length_ratio * 0.05
        + tonnage_ratio * 0.15
    )

    return {
        "score": score,
        "exact_match": exact_match,
        "name_ratio": name_ratio,
        "prefix_ratio": prefix_ratio,
        "length_ratio": length_ratio,
        "tonnage_ratio": tonnage_ratio,
    }


def rank_candidates(
    row,
    catalog: pd.DataFrame,
    candidate_indices: list[int],
) -> list[dict]:
    """
    후보 행 번호를 점수순으로 정렬합니다.

    반환값 예시:
    [
        {
            "index": 0,
            "vessel_id": "VESSEL_00001",
            "score": 0.91,
            ...
        }
    ]
    """
    ranked = []

    for candidate_index in candidate_indices:
        candidate_index = int(candidate_index)
        catalog_row = catalog.iloc[candidate_index]

        features = calculate_candidate_score(row, catalog_row)

        ranked.append(
            {
                "index": candidate_index,
                "vessel_id": str(catalog_row["vessel_id"]),
                **features,
            }
        )

    # 점수 내림차순
    # 점수가 같으면 vessel_id 오름차순으로 정렬하여 결과를 고정합니다.
    ranked.sort(
        key=lambda item: (
            -item["score"],
            item["vessel_id"],
        )
    )

    return ranked


def rerank_candidates(
    row,
    catalog: pd.DataFrame,
    candidate_indices: list[int],
) -> str | None:
    """후보를 재랭킹한 뒤 가장 높은 후보의 vessel_id를 반환합니다."""
    ranked = rank_candidates(
        row=row,
        catalog=catalog,
        candidate_indices=candidate_indices,
    )

    if not ranked:
        return None

    return ranked[0]["vessel_id"]