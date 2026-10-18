"""Single-record adapter for the teammate's v11 LightGBM ranker."""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import pickle

import lightgbm as lgb
import numpy as np
import pandas as pd

from ..legacy_matching.matching_service import CandidateEvidence, MatchResult
from ..legacy_matching.record_validation import VesselRecord
from .source import match_lib as ml


class TeamV11Service:
    """Use locally trained v11 artifacts; never fit on evaluation rows."""

    def __init__(self, catalog: pd.DataFrame, artifact_dir: str | Path):
        model_dir = Path(artifact_dir) / "model"
        model_path = model_dir / "lgb_ranker.txt"
        resources_path = model_dir / "resources.pkl"
        for path in (model_path, resources_path):
            if not path.is_file():
                raise FileNotFoundError(f"v11 학습 산출물이 없습니다: {path}")
        self.model = lgb.Booster(model_str=model_path.read_text(encoding="utf-8"))
        with resources_path.open("rb") as handle:
            self.resources = pickle.load(handle)
        self.index = ml.CatalogIndex(catalog)

    def match_vessel(self, record: VesselRecord, *, top_k: int = 5) -> MatchResult:
        if not isinstance(top_k, int) or isinstance(top_k, bool) or top_k < 1:
            raise ValueError("top_k는 1 이상의 정수여야 합니다")
        row = asdict(record)
        query = ml.prepare_query(row, self.resources)
        candidate_lists, feature_matrix = ml.candidates_and_features(
            [query], self.index, self.resources
        )
        candidates = candidate_lists[0]
        scores = np.asarray(self.model.predict(feature_matrix), dtype=float)
        if len(scores) != len(candidates) or not np.isfinite(scores).all():
            raise ValueError("v11 후보 점수가 유효하지 않습니다")

        winner = int(candidates[int(np.argmax(scores))])
        prediction = str(self.index.ids[winner])
        source = "ranking"
        if self.resources.get("use_name_lookup", False):
            lookup = ml.lookup_by_name(row, self.resources, self.index)
            if lookup is not None:
                prediction, source = str(lookup), "train_name_lookup"
        if self.resources.get("use_callsign", False):
            lookup = self.resources["callsign"].get(row.get("call_sign"))
            if lookup is not None:
                prediction, source = str(lookup), "train_callsign_lookup"

        # The teammate's predict() uses np.argmax, which keeps the first
        # candidate on a score tie. Keep displayed ranks consistent with it.
        order = sorted(range(len(candidates)), key=lambda pos: (-float(scores[pos]), pos))
        ranked = []
        for rank, pos in enumerate(order[:top_k], start=1):
            index = int(candidates[pos])
            score = float(scores[pos])
            ranked.append(
                CandidateEvidence(
                    vessel_id=str(self.index.ids[index]),
                    catalog_name=str(self.index.catalog.iloc[index]["vessel_name"]),
                    rank=rank,
                    features={
                        name: float(value) if np.isfinite(value) else None
                        for name, value in zip(ml.FEATURE_NAMES, feature_matrix[pos])
                    },
                    base_score=score,
                    fuzzy_score=0.0,
                    final_score=score,
                )
            )
        return MatchResult(
            baseline_prediction=prediction,
            top_rank=ranked,
            model_name="team_v11",
            prediction_source=source,
        )
