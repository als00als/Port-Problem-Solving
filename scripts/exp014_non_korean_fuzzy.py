"""Local comparison: add RapidFuzz candidates for non-Korean names only."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import pandas as pd

from scripts.exp014_validation import (
    PROJECT_ROOT,
    RESULTS_ROOT,
    SEEDS,
    evaluate_split,
    load_runtime,
    load_validation_split,
    read_data,
)


def main() -> int:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = RESULTS_ROOT / f"exp014_non_korean_fuzzy_{timestamp}"
    if result_dir.exists():
        raise FileExistsError(f"Refusing to overwrite results: {result_dir}")
    result_dir.mkdir(parents=True)

    runtime = load_runtime()
    index_class = runtime[0]
    train, catalog = read_data()
    results = []
    repeatability = {}

    for seed in SEEDS:
        valid = load_validation_split(train, seed)
        index = index_class(catalog)
        baseline_first = evaluate_split(
            valid, catalog, index, runtime, tie_policy="vessel_id"
        )
        baseline_second = evaluate_split(
            valid, catalog, index, runtime, tie_policy="vessel_id"
        )
        fuzzy_first = evaluate_split(
            valid,
            catalog,
            index,
            runtime,
            tie_policy="vessel_id",
            include_non_korean_fuzzy=True,
        )
        fuzzy_second = evaluate_split(
            valid,
            catalog,
            index,
            runtime,
            tie_policy="vessel_id",
            include_non_korean_fuzzy=True,
        )

        baseline_repeatable = baseline_first["details"]["prediction"].equals(
            baseline_second["details"]["prediction"]
        )
        fuzzy_repeatable = fuzzy_first["details"]["prediction"].equals(
            fuzzy_second["details"]["prediction"]
        )
        repeatability[str(seed)] = {
            "baseline": baseline_repeatable,
            "non_korean_fuzzy": fuzzy_repeatable,
        }

        baseline_first["details"].to_csv(
            result_dir / f"seed{seed}_exp014_details.csv",
            index=False,
            encoding="utf-8-sig",
        )
        fuzzy_first["details"].to_csv(
            result_dir / f"seed{seed}_non_korean_fuzzy_details.csv",
            index=False,
            encoding="utf-8-sig",
        )

        baseline_metrics = baseline_first["metrics"]
        fuzzy_metrics = fuzzy_first["metrics"]
        metric_names = (
            "weighted",
            "korean_accuracy",
            "non_korean_accuracy",
            "korean_coverage",
            "overall_coverage",
            "mean_candidate_count",
        )
        results.append(
            {
                "seed": seed,
                "baseline": baseline_metrics,
                "non_korean_fuzzy": fuzzy_metrics,
                "delta": {
                    name: fuzzy_metrics[name] - baseline_metrics[name]
                    for name in metric_names
                },
                "repeatability": repeatability[str(seed)],
            }
        )

    report = {
        "experiment": "EXP-014 plus non-Korean RapidFuzz candidate generation",
        "baseline": "EXP-014 fixed_40_50_60; original ranker and vessel_id tie-break",
        "change": {
            "applies_to": "non-Korean vessel_name rows only",
            "search": "RapidFuzz ratio over normalized catalog romanized names",
            "limit": 40,
            "cutoff": 50,
            "max_candidates": 60,
            "candidate_merge_order": "fuzzy matches, then existing name/tonnage candidates",
            "ranker": "unchanged; fuzzy bonus remains Korean-only",
        },
        "results": results,
        "repeatability": repeatability,
        "results_dir": str(result_dir.relative_to(PROJECT_ROOT)),
        "submission_created": False,
    }
    (result_dir / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    rows = []
    for item in results:
        for model in ("baseline", "non_korean_fuzzy"):
            rows.append({"seed": item["seed"], "model": model, **item[model]})
    pd.DataFrame(rows).to_csv(
        result_dir / "score_comparison.csv", index=False, encoding="utf-8-sig"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
