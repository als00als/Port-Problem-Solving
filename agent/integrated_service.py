"""Keep the original matcher available while v11 drives the primary recommendation."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .legacy_matching.matching_service import MatchingService, MatchResult
from .legacy_matching.record_validation import VesselRecord
from .team_v11.service import TeamV11Service


DEFAULT_ARTIFACT_DIR = Path(__file__).resolve().parents[1] / "team_v11_artifacts" / "v11" / "src"


class IntegratedMatchingService:
    def __init__(self, catalog: pd.DataFrame, artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR):
        self.team = TeamV11Service(catalog, artifact_dir)
        self.legacy = MatchingService(catalog)
        self.index = self.team.index

    def match_vessel(self, record: VesselRecord, *, top_k: int = 5) -> MatchResult:
        return self.team.match_vessel(record, top_k=top_k)

    def legacy_match_vessel(self, record: VesselRecord, *, top_k: int = 5) -> MatchResult:
        return self.legacy.match_vessel(record, top_k=top_k)

    def expand_search(self, record: VesselRecord, *, profile: str = "wide_v1", top_k: int = 5):
        return self.legacy.expand_search(record, profile=profile, top_k=top_k)

    def compare_candidates(self, record: VesselRecord, candidate_ids: list[str]):
        return self.legacy.compare_candidates(record, candidate_ids)


def load_integrated_service(
    catalog_path: str | Path,
    artifact_dir: str | Path = DEFAULT_ARTIFACT_DIR,
) -> IntegratedMatchingService:
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    return IntegratedMatchingService(catalog, artifact_dir)
