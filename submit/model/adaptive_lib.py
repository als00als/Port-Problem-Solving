"""Standalone EXP-013 adaptive candidate search and ranking."""

from __future__ import annotations

import math
import unicodedata
from difflib import SequenceMatcher

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from unidecode import unidecode


# EXP-014: fixed wide fuzzy search selected after three independent group splits.
FUZZY_LIMIT = 40
FUZZY_CUTOFF = 50
MAX_CANDIDATES = 60
EXPANDED_FUZZY_LIMIT = 20
EXPANDED_FUZZY_CUTOFF = 55
EXPANDED_MAX_CANDIDATES = 40
TOP1_THRESHOLD = 0.70
MARGIN_THRESHOLD = 0.02
FUZZY_SCORE_BONUS = 0.25


def normalize_name(value: object) -> str:
    if not isinstance(value, str):
        return ""
    value = unicodedata.normalize("NFKC", value).upper()
    return "".join(char for char in value if char.isalnum())


def is_korean(value: object) -> bool:
    return isinstance(value, str) and any("가" <= char <= "힣" for char in value)


def romanized_name(value: object) -> str:
    normalized = normalize_name(value)
    return normalize_name(unidecode(normalized)) if normalized else ""


def positive_float(value: object) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def relative_gap(left: object, right: object) -> float:
    left_value = positive_float(left)
    right_value = positive_float(right)
    if left_value is None or right_value is None:
        return 2.0
    return min(abs(left_value - right_value) / max(left_value, right_value, 1.0), 2.0)


def name_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return SequenceMatcher(None, left, right).ratio()


def prefix_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    length = min(len(left), len(right))
    same = 0
    for left_char, right_char in zip(left, right):
        if left_char != right_char:
            break
        same += 1
    return same / length if length else 0.0


def length_similarity(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    return min(len(left), len(right)) / max(len(left), len(right))


def tonnage_similarity(row: dict, catalog_row: pd.Series) -> float:
    catalog_tonnage = catalog_row.get("gross_tonnage")
    gaps = [
        relative_gap(row.get("gross_tonnage"), catalog_tonnage),
        relative_gap(row.get("international_gross_tonnage"), catalog_tonnage),
    ]
    valid_query_values = [
        positive_float(row.get("gross_tonnage")),
        positive_float(row.get("international_gross_tonnage")),
    ]
    if all(value is None for value in valid_query_values):
        return 0.0
    return max(0.0, 1.0 - min(gaps))


def base_features(row: dict, catalog_row: pd.Series) -> dict[str, float]:
    query_name = normalize_name(row.get("vessel_name"))
    candidate_name = normalize_name(catalog_row.get("vessel_name"))
    exact_match = float(bool(query_name) and query_name == candidate_name)
    name_ratio = name_similarity(query_name, candidate_name)
    prefix_ratio = prefix_similarity(query_name, candidate_name)
    length_ratio = length_similarity(query_name, candidate_name)
    tonnage_ratio = tonnage_similarity(row, catalog_row)
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


class AdaptiveCatalogIndex:
    def __init__(self, catalog: pd.DataFrame) -> None:
        required = {"vessel_id", "vessel_name", "gross_tonnage"}
        missing = required - set(catalog.columns)
        if missing:
            raise ValueError(f"Catalog is missing columns: {sorted(missing)}")
        self.catalog = catalog.reset_index(drop=True)
        if not self.catalog["vessel_id"].astype(str).is_unique:
            raise ValueError("vessel_id must be unique")
        self.ids = self.catalog["vessel_id"].astype(str).tolist()
        self.name_to_indices: dict[str, list[int]] = {}
        for index, name in enumerate(self.catalog["vessel_name"]):
            key = normalize_name(name)
            if key:
                self.name_to_indices.setdefault(key, []).append(index)
        self.romanized_names = [romanized_name(name) for name in self.catalog["vessel_name"]]

    def generate_base_candidates(self, row: dict, max_candidates: int) -> list[int]:
        query_name = normalize_name(row.get("vessel_name"))
        candidates: list[int] = []
        used: set[int] = set()

        def append(indices: list[int]) -> None:
            for index in indices:
                index = int(index)
                if index not in used:
                    candidates.append(index)
                    used.add(index)
                if len(candidates) >= max_candidates:
                    break

        append(self.name_to_indices.get(query_name, []))
        tonnage = positive_float(row.get("gross_tonnage"))
        if tonnage is not None:
            catalog_tonnage = pd.to_numeric(self.catalog["gross_tonnage"], errors="coerce")
            gaps = (catalog_tonnage - tonnage).abs()
            append(gaps.dropna().nsmallest(10).index.tolist())
        return candidates[:max_candidates]

    def fuzzy_confidence(self, row: dict) -> tuple[float, float]:
        if not is_korean(row.get("vessel_name")):
            return 1.0, 1.0
        query_key = romanized_name(row.get("vessel_name"))
        if not query_key:
            return 0.0, 0.0
        matches = process.extract(
            query_key,
            self.romanized_names,
            scorer=fuzz.ratio,
            limit=2,
            score_cutoff=0,
        )
        if not matches:
            return 0.0, 0.0
        top1 = float(matches[0][1]) / 100.0
        top2 = float(matches[1][1]) / 100.0 if len(matches) > 1 else 0.0
        return top1, top1 - top2

    def should_expand(self, row: dict) -> bool:
        top1, margin = self.fuzzy_confidence(row)
        return (
            is_korean(row.get("vessel_name"))
            and top1 < TOP1_THRESHOLD
            and margin < MARGIN_THRESHOLD
        )

    def candidates_with_scores(self, row: dict) -> tuple[list[int], dict[int, float]]:
        fuzzy_limit = FUZZY_LIMIT
        fuzzy_cutoff = FUZZY_CUTOFF
        max_candidates = MAX_CANDIDATES
        base_indices = self.generate_base_candidates(row, max_candidates)
        fuzzy_indices: list[int] = []
        fuzzy_scores: dict[int, float] = {}

        if is_korean(row.get("vessel_name")):
            query_key = romanized_name(row.get("vessel_name"))
            if query_key:
                matches = process.extract(
                    query_key,
                    self.romanized_names,
                    scorer=fuzz.ratio,
                    limit=fuzzy_limit,
                    score_cutoff=fuzzy_cutoff,
                )
                for _, score, index in matches:
                    index = int(index)
                    fuzzy_indices.append(index)
                    fuzzy_scores[index] = float(score) / 100.0

        merged: list[int] = []
        seen: set[int] = set()
        for index in fuzzy_indices + base_indices:
            index = int(index)
            if index not in seen:
                merged.append(index)
                seen.add(index)
            if len(merged) >= max_candidates:
                break
        return merged, fuzzy_scores

    def predict_row(self, row: dict) -> str:
        candidate_indices, fuzzy_scores = self.candidates_with_scores(row)
        ranked = []
        for index in candidate_indices:
            catalog_row = self.catalog.iloc[index]
            features = base_features(row, catalog_row)
            score = features["score"] + FUZZY_SCORE_BONUS * fuzzy_scores.get(index, 0.0)
            ranked.append((score, self.ids[index]))
        if not ranked:
            return self.ids[0]
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return ranked[0][1]

    def predict_ids(self, queries: pd.DataFrame) -> list[str]:
        return [self.predict_row(row) for row in queries.to_dict("records")]
