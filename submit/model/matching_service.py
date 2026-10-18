"""Service layer for single-record vessel matching and result formatting."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from .adaptive_lib import AdaptiveCatalogIndex, FUZZY_SCORE_BONUS, base_features
from .record_validation import VesselRecord, validate_record


@dataclass
class CandidateEvidence:
    vessel_id: str
    catalog_name: str
    rank: int
    features: dict[str, float]
    base_score: float
    fuzzy_score: float
    final_score: float


@dataclass
class MatchResult:
    baseline_prediction: str | None
    top_rank: list[CandidateEvidence]


@dataclass
class ExpandedSearchResult:
    profile: str
    suggested_prediction: str | None
    candidate_count: int
    top_rank: list[CandidateEvidence]


class MatchingService:
    def __init__(self, catalog: pd.DataFrame):
        self.index = AdaptiveCatalogIndex(catalog)
        self.id_to_index = {
            vessel_id: index for index, vessel_id in enumerate(self.index.ids)
        }

    def match_vessel(self, record: VesselRecord, *, top_k: int = 5) -> MatchResult:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k는 1 이상의 정수여야 합니다")

        row = asdict(record)
        candidate_indices, fuzzy_scores = self.index.candidates_with_scores(row)
        ranked = []

        for index in candidate_indices:
            catalog_row = self.index.catalog.iloc[index]
            features = base_features(row, catalog_row)
            base_score = features["score"]
            fuzzy_score = fuzzy_scores.get(index, 0.0)
            final_score = base_score + FUZZY_SCORE_BONUS * fuzzy_score

            ranked.append(
                CandidateEvidence(
                    vessel_id=self.index.ids[index],
                    catalog_name=str(catalog_row["vessel_name"]),
                    rank=0,
                    features={key: value for key, value in features.items() if key != "score"},
                    base_score=base_score,
                    fuzzy_score=fuzzy_score,
                    final_score=final_score,
                )
            )

        ranked.sort(key=lambda candidate: (-candidate.final_score, candidate.vessel_id))
        for rank, candidate in enumerate(ranked, start=1):
            candidate.rank = rank

        return MatchResult(
            baseline_prediction=self.index.predict_row(row),
            top_rank=ranked[:top_k],
        )

    def expand_search(
        self,
        record: VesselRecord,
        *,
        profile: str = "wide_v1",
        top_k: int = 5,
    ) -> ExpandedSearchResult:
        """Search a wider Korean-name candidate set without changing baseline."""
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k는 1 이상의 정수여야 합니다")

        row = asdict(record)
        candidate_indices, fuzzy_scores = self.index.expanded_candidates_with_scores(
            row, profile=profile
        )
        ranked = []
        for index in candidate_indices:
            catalog_row = self.index.catalog.iloc[index]
            features = base_features(row, catalog_row)
            base_score = features["score"]
            fuzzy_score = fuzzy_scores.get(index, 0.0)
            final_score = base_score + FUZZY_SCORE_BONUS * fuzzy_score
            ranked.append(
                CandidateEvidence(
                    vessel_id=self.index.ids[index],
                    catalog_name=str(catalog_row["vessel_name"]),
                    rank=0,
                    features={key: value for key, value in features.items() if key != "score"},
                    base_score=base_score,
                    fuzzy_score=fuzzy_score,
                    final_score=final_score,
                )
            )

        ranked.sort(key=lambda candidate: (-candidate.final_score, candidate.vessel_id))
        for rank, candidate in enumerate(ranked, start=1):
            candidate.rank = rank

        return ExpandedSearchResult(
            profile=profile,
            suggested_prediction=ranked[0].vessel_id if ranked else None,
            candidate_count=len(candidate_indices),
            top_rank=ranked[:top_k],
        )

    def compare_candidates(
        self, record: VesselRecord, candidate_ids: list[str]
    ) -> list[CandidateEvidence]:
        if not isinstance(candidate_ids, list):
            raise TypeError("candidate_ids는 리스트여야 합니다")
        if len(candidate_ids) > 5:
            raise ValueError("한 번에 비교할 후보는 최대 5개입니다.")

        unknown_ids = set(candidate_ids) - set(self.index.ids)
        if unknown_ids:
            raise ValueError(f"카탈로그에 없는 vessel_id: {unknown_ids}")

        row = asdict(record)
        _, fuzzy_scores = self.index.candidates_with_scores(row)
        candidate_results = []

        for candidate_id in candidate_ids:
            index = self.id_to_index[candidate_id]
            candidate_row = self.index.catalog.iloc[index]
            features = base_features(row, candidate_row)
            base_score = features["score"]
            fuzzy_score = fuzzy_scores.get(index, 0.0)
            final_score = base_score + FUZZY_SCORE_BONUS * fuzzy_score

            candidate_results.append(
                CandidateEvidence(
                    vessel_id=candidate_id,
                    catalog_name=str(candidate_row["vessel_name"]),
                    rank=0,
                    features={key: value for key, value in features.items() if key != "score"},
                    base_score=base_score,
                    fuzzy_score=fuzzy_score,
                    final_score=final_score,
                )
            )

        candidate_results.sort(
            key=lambda candidate: (-candidate.final_score, candidate.vessel_id)
        )
        for rank, candidate in enumerate(candidate_results, start=1):
            candidate.rank = rank

        return candidate_results


def load_matching_service(catalog_path: str | Path) -> MatchingService:
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    return MatchingService(catalog)


def predict_payload(
    payload: dict, service: MatchingService, *, top_k: int = 5
) -> MatchResult:
    record = validate_record(payload)
    return service.match_vessel(record, top_k=top_k)


def result_to_dict(result: MatchResult) -> dict[str, Any]:
    return asdict(result)


def result_to_json(result: MatchResult, *, ensure_ascii: bool = False) -> str:
    return json.dumps(result_to_dict(result), ensure_ascii=ensure_ascii)
