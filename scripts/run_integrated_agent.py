"""Inspect one official training row or a local JSON record with the integrated matcher.

This tool never creates a submission file or sends data to a service.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from agent.integrated_service import DEFAULT_ARTIFACT_DIR, load_integrated_service
from agent.legacy_matching.record_validation import validate_record


def main() -> None:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--sample-id", help="공식 train.csv의 sample_id")
    source.add_argument("--record", type=Path, help="단일 입력 행이 담긴 로컬 JSON 파일")
    parser.add_argument("--mode", choices=("match", "agent"), default="match")
    parser.add_argument("--details", action="store_true", help="후보별 특징을 포함한 전체 에이전트 상태 표시")
    parser.add_argument("--catalog", type=Path, default=ROOT / "data" / "open" / "vessel_catalog.csv")
    parser.add_argument("--train", type=Path, default=ROOT / "data" / "open" / "train.csv")
    parser.add_argument("--artifacts", type=Path, default=DEFAULT_ARTIFACT_DIR)
    args = parser.parse_args()

    if args.record is not None:
        payload = json.loads(args.record.read_text(encoding="utf-8"))
    else:
        train = pd.read_csv(args.train, encoding="utf-8-sig")
        selected = train.loc[train["sample_id"] == args.sample_id]
        if len(selected) != 1:
            parser.error("sample_id가 train.csv에서 정확히 한 행이어야 합니다")
        payload = selected.drop(columns=["target_vessel_id"]).iloc[0].to_dict()
    record = validate_record(payload)
    service = load_integrated_service(args.catalog, args.artifacts)

    if args.mode == "agent":
        try:
            from agent.graph import run_agent
        except ModuleNotFoundError as exc:
            raise SystemExit(f"에이전트 그래프 의존성이 없습니다: {exc.name}") from exc
        state = run_agent(asdict(record), service=service, session_id=record.sample_id)
        primary = state.get("primary_result") or {}
        legacy = state.get("legacy_result") or {}
        review = state.get("review_signals") or {}
        output = {
            "status": state.get("status"),
            "primary_model": primary.get("model_name"),
            "primary_prediction": primary.get("baseline_prediction"),
            "primary_prediction_source": primary.get("prediction_source"),
            "legacy_prediction": legacy.get("baseline_prediction"),
            "review_required": review.get("review_required"),
            "review_reasons": review.get("reasons"),
            "final_report": state.get("final_report"),
            "errors": state.get("errors"),
        }
        if args.details:
            output.update(
                primary_result=primary,
                legacy_result=legacy,
                review_signals=review,
            )
    else:
        primary = service.match_vessel(record)
        legacy = service.legacy_match_vessel(record)
        output = {
            "status": "match_only",
            "primary_model": primary.model_name,
            "primary_prediction": primary.baseline_prediction,
            "primary_prediction_source": primary.prediction_source,
            "legacy_prediction": legacy.baseline_prediction,
            "top_rank": [
                {
                    "vessel_id": candidate.vessel_id,
                    "catalog_name": candidate.catalog_name,
                    "rank": candidate.rank,
                    "raw_ranker_score": candidate.final_score,
                }
                for candidate in primary.top_rank
            ],
        }
    print(json.dumps(output, ensure_ascii=False, indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
