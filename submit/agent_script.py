"""Batch submission entry point using the implemented MatchingService."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from model.matching_service import load_matching_service, predict_payload


ROOT = Path(__file__).resolve().parent


def run(
    test_path: str | Path | None = None,
    catalog_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    test_path = Path(test_path) if test_path else ROOT / "data" / "test.csv"
    catalog_path = Path(catalog_path) if catalog_path else ROOT / "data" / "vessel_catalog.csv"
    output_path = Path(output_path) if output_path else ROOT / "output" / "submission.csv"

    test = pd.read_csv(test_path, encoding="utf-8-sig")
    if "sample_id" not in test.columns:
        raise ValueError("test.csv에 sample_id 컬럼이 없습니다")
    if test["sample_id"].isna().any() or not test["sample_id"].is_unique:
        raise ValueError("test.csv sample_id must be non-null and unique")

    service = load_matching_service(catalog_path)
    catalog_ids = set(service.index.ids)
    predictions: list[str] = []

    for payload in test.to_dict(orient="records"):
        result = predict_payload(payload, service, top_k=1)
        if result.top_rank:
            prediction = result.top_rank[0].vessel_id
        else:
            prediction = result.baseline_prediction

        if prediction is None or prediction not in catalog_ids:
            raise ValueError("MatchingService가 유효한 catalog vessel_id를 반환하지 않았습니다")
        predictions.append(prediction)

    if len(predictions) != len(test):
        raise ValueError("예측 수가 test.csv 행 수와 일치하지 않습니다")

    submission = pd.DataFrame(
        {
            "sample_id": test["sample_id"].tolist(),
            "target_vessel_id": predictions,
        }
    )
    if submission["sample_id"].tolist() != test["sample_id"].tolist():
        raise ValueError("제출 행 순서가 test.csv와 일치하지 않습니다")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(submission)} predictions to {output_path}")
    return output_path


if __name__ == "__main__":
    run()
