"""DACON submission entry point for the EXP-013 adaptive pipeline."""

from pathlib import Path
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "model"))

from adaptive_lib import AdaptiveCatalogIndex  # noqa: E402


def run(
    test_path: str | Path | None = None,
    catalog_path: str | Path | None = None,
    output_path: str | Path | None = None,
) -> Path:
    test_path = Path(test_path) if test_path else ROOT / "data" / "test.csv"
    catalog_path = Path(catalog_path) if catalog_path else ROOT / "data" / "vessel_catalog.csv"
    output_path = Path(output_path) if output_path else ROOT / "output" / "submission.csv"

    test = pd.read_csv(test_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    if test["sample_id"].isna().any() or not test["sample_id"].is_unique:
        raise ValueError("test.csv sample_id must be non-null and unique")

    index = AdaptiveCatalogIndex(catalog)
    predictions = index.predict_ids(test)
    if len(predictions) != len(test) or not set(predictions).issubset(set(index.ids)):
        raise ValueError("Invalid catalog predictions")

    result = pd.DataFrame({
        "sample_id": test["sample_id"].tolist(),
        "target_vessel_id": predictions,
    })
    if len(result) != len(test) or result["sample_id"].tolist() != test["sample_id"].tolist():
        raise ValueError("Submission row order does not match test input")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(result)} predictions to {output_path}")
    return output_path


if __name__ == "__main__":
    run()
