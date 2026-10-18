"""DACON 추론 진입점: 후보 검색 + LightGBM 랭커."""

from pathlib import Path
import pickle
import sys

import lightgbm as lgb
import pandas as pd

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE / "model"))
import match_lib as ml  # noqa: E402


def run(
    test_path: str | Path = "data/test.csv",
    catalog_path: str | Path = "data/vessel_catalog.csv",
    output_path: str | Path = "output/submission.csv",
) -> Path:
    test = pd.read_csv(test_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    ensemble_path = BASE / "model" / "ensemble.pkl"
    if ensemble_path.exists():
        with open(ensemble_path, "rb") as f:
            model = pickle.load(f)
    else:
        model = lgb.Booster(model_str=(BASE / "model" / "lgb_ranker.txt").read_text(encoding="utf-8"))
    with open(BASE / "model" / "resources.pkl", "rb") as f:
        res = pickle.load(f)

    index = ml.CatalogIndex(catalog)
    predictions = ml.predict(
        model, test, index, res,
        use_callsign=res.get("use_callsign", False),
        use_name_lookup=res.get("use_name_lookup", False),
    )
    if len(predictions) != len(test) or not set(predictions).issubset(index.id_to_index):
        raise ValueError("Invalid catalog predictions")
    result = pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": predictions})
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output_path, index=False, encoding="utf-8")
    print(f"Saved {len(result)} predictions to {output_path}")
    return output_path


if __name__ == "__main__":
    run()
