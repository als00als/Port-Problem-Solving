"""Reproduce EXP-014 and evaluate a local-only catalog-name tie break."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile
from datetime import datetime
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "open"
RESULTS_ROOT = PROJECT_ROOT / "experiments" / "results"
SPLITS_ROOT = PROJECT_ROOT / "experiments" / "splits"
SUBMIT_ROOT = PROJECT_ROOT / "submit"
CURRENT_ZIP = PROJECT_ROOT / "submissions" / "final_exp014.zip"
NEW_ZIP = PROJECT_ROOT / "submissions" / "final_exp015_catalog_name_tiebreak.zip"
SEEDS = (42, 123, 2026)
TIE_TOLERANCE = 1e-12
EXPECTED_ARCHIVE_MEMBERS = (
    "model/adaptive_lib.py",
    "script.py",
    "requirements.txt",
)
EXPECTED_LOG = {
    42: {
        "weighted": 0.7256,
        "korean_accuracy": 0.7202,
        "non_korean_accuracy": 0.7382,
        "korean_coverage": 0.9753,
        "mean_candidate_count": 13.7257,
    },
    123: {
        "weighted": 0.7277,
        "korean_accuracy": 0.7203,
        "non_korean_accuracy": 0.7449,
        "korean_coverage": 0.9157,
        "mean_candidate_count": 13.8496,
    },
    2026: {
        "weighted": 0.7192,
        "korean_accuracy": 0.7163,
        "non_korean_accuracy": 0.7261,
        "korean_coverage": 0.9628,
        "mean_candidate_count": 13.1428,
    },
}


def select_prediction(
    candidates: list[dict], tie_policy: str = "vessel_id"
) -> str | None:
    """Choose top score, resolving only near-equal top scores by the policy."""
    if tie_policy not in {"vessel_id", "catalog_name"}:
        raise ValueError(f"Unknown tie policy: {tie_policy}")
    if not candidates:
        return None

    top_score = max(float(item["score"]) for item in candidates)
    top_candidates = [
        item
        for item in candidates
        if abs(float(item["score"]) - top_score) <= TIE_TOLERANCE
    ]
    if tie_policy == "catalog_name":
        winner = min(
            top_candidates,
            key=lambda item: (
                normalize_catalog_name(item.get("vessel_name")),
                str(item["vessel_id"]),
            ),
        )
    else:
        winner = min(top_candidates, key=lambda item: str(item["vessel_id"]))
    return str(winner["vessel_id"])


def normalize_catalog_name(value: object) -> str:
    if not isinstance(value, str):
        return ""
    import unicodedata

    value = unicodedata.normalize("NFKC", value).upper()
    return "".join(char for char in value if char.isalnum())


def count_prediction_changes(left: list[object], right: list[object]) -> int:
    if len(left) != len(right):
        raise ValueError("Prediction lists must have the same length")

    def normalized(value: object) -> str | None:
        return None if pd.isna(value) else str(value)

    return sum(normalized(a) != normalized(b) for a, b in zip(left, right))


def load_runtime():
    sys.path.insert(0, str(SUBMIT_ROOT / "model"))
    from adaptive_lib import (  # pylint: disable=import-outside-toplevel
        AdaptiveCatalogIndex,
        FUZZY_CUTOFF,
        FUZZY_LIMIT,
        FUZZY_SCORE_BONUS,
        MAX_CANDIDATES,
        base_features,
        is_korean,
        normalize_name,
        romanized_name,
    )
    from rapidfuzz import fuzz, process  # pylint: disable=import-outside-toplevel

    config = {
        "FUZZY_LIMIT": FUZZY_LIMIT,
        "FUZZY_CUTOFF": FUZZY_CUTOFF,
        "MAX_CANDIDATES": MAX_CANDIDATES,
    }
    if config != {"FUZZY_LIMIT": 40, "FUZZY_CUTOFF": 50, "MAX_CANDIDATES": 60}:
        raise AssertionError(f"Current submission is not EXP-014: {config}")
    return (
        AdaptiveCatalogIndex,
        FUZZY_SCORE_BONUS,
        base_features,
        fuzz,
        is_korean,
        normalize_name,
        process,
        romanized_name,
    )


def read_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(DATA_ROOT / "train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(DATA_ROOT / "vessel_catalog.csv", encoding="utf-8-sig")
    return train, catalog


def load_validation_split(train: pd.DataFrame, seed: int) -> pd.DataFrame:
    split_path = SPLITS_ROOT / f"group_split_seed{seed}.json"
    split = json.loads(split_path.read_text(encoding="utf-8"))
    valid_ids = set(split["valid_sample_ids"])
    valid = train[train["sample_id"].isin(valid_ids)].copy()
    if len(valid) != len(valid_ids):
        raise AssertionError(f"seed {seed}: split IDs do not all exist in train.csv")
    return valid


def generate_fixed_candidates(
    row,
    index,
    fuzz,
    process,
    romanized_name,
    include_non_korean_fuzzy: bool = False,
):
    base_indices = index.generate_base_candidates(row, max_candidates=60)
    fuzzy_indices: list[int] = []
    fuzzy_scores: dict[int, float] = {}

    from adaptive_lib import (  # pylint: disable=import-outside-toplevel
        is_korean,
        normalize_name,
    )

    korean_name = is_korean(row.get("vessel_name"))
    if korean_name or include_non_korean_fuzzy:
        query_key = (
            romanized_name(row.get("vessel_name"))
            if korean_name
            else normalize_name(row.get("vessel_name"))
        )
        if query_key:
            matches = process.extract(
                query_key,
                index.romanized_names,
                scorer=fuzz.ratio,
                limit=40,
                score_cutoff=50,
            )
            for _, score, candidate_index in matches:
                candidate_index = int(candidate_index)
                fuzzy_indices.append(candidate_index)
                fuzzy_scores[candidate_index] = float(score) / 100.0

    merged: list[int] = []
    seen: set[int] = set()
    for candidate_index in fuzzy_indices + base_indices:
        if candidate_index not in seen:
            merged.append(candidate_index)
            seen.add(candidate_index)
        if len(merged) >= 60:
            break
    return merged, fuzzy_scores


def evaluate_split(
    valid,
    catalog,
    index,
    runtime,
    tie_policy: str,
    include_non_korean_fuzzy: bool = False,
) -> dict:
    (
        _,
        fuzzy_score_bonus,
        base_features,
        fuzz,
        is_korean,
        normalize_name,
        process,
        romanized_name,
    ) = runtime
    catalog = catalog.reset_index(drop=True)
    records: list[dict] = []
    started = time.perf_counter()

    for row in valid.to_dict("records"):
        candidate_indices, fuzzy_scores = generate_fixed_candidates(
            row,
            index,
            fuzz,
            process,
            romanized_name,
            include_non_korean_fuzzy=include_non_korean_fuzzy,
        )
        scored = []
        for candidate_index in candidate_indices:
            catalog_row = catalog.iloc[candidate_index]
            features = base_features(row, catalog_row)
            row_fuzzy_score = (
                fuzzy_scores.get(candidate_index, 0.0)
                if is_korean(row.get("vessel_name"))
                else 0.0
            )
            scored.append(
                {
                    "score": features["score"]
                    + fuzzy_score_bonus * row_fuzzy_score,
                    "vessel_id": str(catalog_row["vessel_id"]),
                    "vessel_name": catalog_row.get("vessel_name"),
                }
            )
        prediction = select_prediction(scored, tie_policy=tie_policy)
        target = str(row["target_vessel_id"])
        top_score = max((item["score"] for item in scored), default=None)
        top_tied = (
            [item for item in scored if abs(item["score"] - top_score) <= TIE_TOLERANCE]
            if top_score is not None
            else []
        )
        target_is_candidate = target in {item["vessel_id"] for item in scored}
        target_top_tied = any(item["vessel_id"] == target for item in top_tied)
        baseline_prediction = select_prediction(scored, tie_policy="vessel_id")
        records.append(
            {
                "sample_id": row["sample_id"],
                "vessel_name": row.get("vessel_name"),
                "target_vessel_id": target,
                "prediction": prediction,
                "baseline_prediction": baseline_prediction,
                "is_korean": is_korean(row.get("vessel_name")),
                "correct": prediction == target,
                "baseline_correct": baseline_prediction == target,
                "candidate_contains_target": target_is_candidate,
                "target_tied_for_top_score": target_top_tied,
                "baseline_tie_break_wrong": (
                    target_top_tied and baseline_prediction != target
                ),
                "top_tie_count": len(top_tied),
                "candidate_count": len(candidate_indices),
                "top_score": top_score,
            }
        )

    details = pd.DataFrame.from_records(records)
    korean = details[details["is_korean"]]
    non_korean = details[~details["is_korean"]]
    korean_accuracy = float(korean["correct"].mean())
    non_korean_accuracy = float(non_korean["correct"].mean())
    korean_coverage = float(korean["candidate_contains_target"].mean())
    overall_coverage = float(details["candidate_contains_target"].mean())
    metrics = {
        "weighted": 0.7 * korean_accuracy + 0.3 * non_korean_accuracy,
        "korean_accuracy": korean_accuracy,
        "non_korean_accuracy": non_korean_accuracy,
        "korean_coverage": korean_coverage,
        "overall_coverage": overall_coverage,
        "mean_candidate_count": float(details["candidate_count"].mean()),
        "candidate_missing_rows": int((~details["candidate_contains_target"]).sum()),
        "reranking_failed_rows": int(
            (details["candidate_contains_target"] & ~details["correct"]).sum()
        ),
        "korean_candidate_missing_rows": int(
            (~korean["candidate_contains_target"]).sum()
        ),
        "korean_reranking_failed_rows": int(
            (korean["candidate_contains_target"] & ~korean["correct"]).sum()
        ),
        "non_korean_candidate_missing_rows": int(
            (~non_korean["candidate_contains_target"]).sum()
        ),
        "non_korean_reranking_failed_rows": int(
            (non_korean["candidate_contains_target"] & ~non_korean["correct"]).sum()
        ),
        "top_tied_rows": int((details["top_tie_count"] > 1).sum()),
        "vessel_id_tie_break_wrong_rows": int(details["baseline_tie_break_wrong"].sum()),
        "tie_break_rows_corrected": int(
            (details["baseline_tie_break_wrong"] & details["correct"]).sum()
        ),
        "new_errors_from_tie_policy": int(
            (details["baseline_correct"] & ~details["correct"]).sum()
        ),
        "changed_predictions": count_prediction_changes(
            details["baseline_prediction"].tolist(), details["prediction"].tolist()
        ),
        "valid_rows": int(len(details)),
        "korean_rows": int(len(korean)),
        "elapsed_seconds": time.perf_counter() - started,
    }
    return {"metrics": metrics, "details": details}


def validate_submission_output(
    archive_path: Path,
    valid: pd.DataFrame,
    catalog: pd.DataFrame,
    result_dir: Path,
    expected_model_hash: str | None = None,
) -> dict:
    with zipfile.ZipFile(archive_path) as archive:
        members = tuple(sorted(archive.namelist()))
        if members != tuple(sorted(EXPECTED_ARCHIVE_MEMBERS)):
            raise AssertionError(f"Unexpected zip members: {members}")
        bad_member = archive.testzip()
        if bad_member:
            raise AssertionError(f"Corrupt zip member: {bad_member}")
        payload_hash = hashlib.sha256(archive.read("model/adaptive_lib.py")).hexdigest()
        if expected_model_hash is None:
            expected_model_hash = hashlib.sha256(
                (SUBMIT_ROOT / "model" / "adaptive_lib.py").read_bytes()
            ).hexdigest()
        if payload_hash != expected_model_hash:
            raise AssertionError("Zip model/adaptive_lib.py differs from expected source")

    test_frame = valid.drop(columns=["target_vessel_id"]).copy()
    output_path = result_dir / f"{archive_path.stem}_validation_seed42.csv"
    with tempfile.TemporaryDirectory(prefix="exp014_package_check_") as tmp:
        root = Path(tmp)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(root)
        data_dir = root / "data"
        data_dir.mkdir()
        test_frame.to_csv(data_dir / "test.csv", index=False, encoding="utf-8-sig")
        catalog.to_csv(data_dir / "vessel_catalog.csv", index=False, encoding="utf-8-sig")
        started = time.perf_counter()
        completed = subprocess.run(
            [sys.executable, str(root / "script.py")],
            cwd=root,
            text=True,
            capture_output=True,
            check=False,
            timeout=120,
        )
        runtime_seconds = time.perf_counter() - started
        if completed.returncode:
            raise RuntimeError(
                f"Submission script failed ({completed.returncode}):\n{completed.stdout}\n{completed.stderr}"
            )
        produced = root / "output" / "submission.csv"
        output = pd.read_csv(produced, encoding="utf-8-sig")
        expected_ids = test_frame["sample_id"].tolist()
        if output.columns.tolist() != ["sample_id", "target_vessel_id"]:
            raise AssertionError(f"Unexpected output columns: {output.columns.tolist()}")
        if len(output) != len(test_frame):
            raise AssertionError(f"Output has {len(output)} rows; expected {len(test_frame)}")
        if output["sample_id"].tolist() != expected_ids:
            raise AssertionError("Output sample_id order differs from test input")
        if output["target_vessel_id"].isna().any():
            raise AssertionError("Output has missing predictions")
        allowed_ids = set(catalog["vessel_id"].astype(str))
        if not set(output["target_vessel_id"].astype(str)).issubset(allowed_ids):
            raise AssertionError("Output contains vessel_id values outside the catalog")
        shutil.copy2(produced, output_path)

    return {
        "archive": str(archive_path.relative_to(PROJECT_ROOT)),
        "members": list(EXPECTED_ARCHIVE_MEMBERS),
        "model_sha256": payload_hash,
        "output_rows": len(test_frame),
        "output_columns": ["sample_id", "target_vessel_id"],
        "sample_id_order_matches": True,
        "missing_predictions": 0,
        "catalog_ids_valid": True,
        "validation_csv": str(output_path.relative_to(PROJECT_ROOT)),
        "runtime_seconds": runtime_seconds,
    }


def build_variant_library() -> bytes:
    source_path = SUBMIT_ROOT / "model" / "adaptive_lib.py"
    source = source_path.read_text(encoding="utf-8")
    old_append = '            ranked.append((score, self.ids[index]))'
    old_sort = '        ranked.sort(key=lambda item: (-item[0], item[1]))\n        return ranked[0][1]'
    new_append = (
        '            ranked.append((score, self.ids[index], '
        'normalize_name(catalog_row.get("vessel_name"))))'
    )
    new_sort = (
        "        top_score = max(item[0] for item in ranked)\n"
        "        top_candidates = [\n"
        "            item for item in ranked\n"
        "            if abs(item[0] - top_score) <= 1e-12\n"
        "        ]\n"
        "        return min(top_candidates, key=lambda item: (item[2], item[1]))[1]"
    )
    if source.count(old_append) != 1 or source.count(old_sort) != 1:
        raise AssertionError("Could not apply the planned tie-only change to the package source")
    variant = source.replace(old_append, new_append).replace(old_sort, new_sort)
    compile(variant, str(source_path), "exec")
    return variant.encode("utf-8")


def build_and_validate_variant_zip(
    valid: pd.DataFrame, catalog: pd.DataFrame, result_dir: Path
) -> dict:
    variant_lib = build_variant_library()
    temporary_zip = result_dir / "final_exp015_catalog_name_tiebreak.candidate.zip"
    with zipfile.ZipFile(temporary_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model/adaptive_lib.py", variant_lib)
        archive.write(SUBMIT_ROOT / "script.py", "script.py")
        archive.write(SUBMIT_ROOT / "requirements.txt", "requirements.txt")
    package_check = validate_submission_output(
        temporary_zip,
        valid,
        catalog,
        result_dir,
        expected_model_hash=hashlib.sha256(variant_lib).hexdigest(),
    )
    if NEW_ZIP.exists():
        with zipfile.ZipFile(NEW_ZIP) as existing, zipfile.ZipFile(temporary_zip) as candidate:
            if tuple(sorted(existing.namelist())) != tuple(sorted(candidate.namelist())):
                raise FileExistsError(f"Refusing to overwrite a different candidate zip: {NEW_ZIP}")
            if any(existing.read(name) != candidate.read(name) for name in candidate.namelist()):
                raise FileExistsError(f"Refusing to overwrite a different candidate zip: {NEW_ZIP}")
    else:
        shutil.copy2(temporary_zip, NEW_ZIP)
    return {
        "archive": str(NEW_ZIP.relative_to(PROJECT_ROOT)),
        "sha256": hashlib.sha256(NEW_ZIP.read_bytes()).hexdigest(),
        "size_bytes": NEW_ZIP.stat().st_size,
        "package_validation": package_check,
    }


def write_case_diagnostics(result_dir: Path, train: pd.DataFrame, catalog: pd.DataFrame, runtime) -> None:
    _, fuzzy_score_bonus, base_features, fuzz, _, _, process, romanized_name = runtime
    index_class = runtime[0]
    index = index_class(catalog)
    train_names = train[["sample_id", "vessel_name"]]
    tie_records = []

    for seed in SEEDS:
        details_path = result_dir / f"seed{seed}_fixed_40_50_60_details.csv"
        details = pd.read_csv(details_path, encoding="utf-8-sig")
        variant_details = pd.read_csv(
            result_dir / f"seed{seed}_catalog_name_tiebreak_details.csv",
            encoding="utf-8-sig",
        )
        variant_predictions = variant_details.set_index("sample_id")["prediction"].to_dict()
        diagnostics = details.merge(train_names, on="sample_id", how="left")
        diagnostics["error_type"] = "correct"
        diagnostics.loc[~diagnostics["candidate_contains_target"], "error_type"] = "candidate_missing"
        diagnostics.loc[
            diagnostics["candidate_contains_target"] & ~diagnostics["baseline_correct"],
            "error_type",
        ] = "reranking_failed"
        diagnostics.to_csv(
            result_dir / f"seed{seed}_row_diagnostics.csv",
            index=False,
            encoding="utf-8-sig",
        )

        tied_errors = diagnostics[
            diagnostics["target_tied_for_top_score"] & ~diagnostics["baseline_correct"]
        ]
        valid = load_validation_split(train, seed)
        rows_by_id = valid.set_index("sample_id").to_dict("index")
        for case in tied_errors.to_dict("records"):
            row = rows_by_id[case["sample_id"]]
            candidate_indices, fuzzy_scores = generate_fixed_candidates(
                row, index, fuzz, process, romanized_name
            )
            scored = []
            for candidate_index in candidate_indices:
                catalog_row = catalog.iloc[candidate_index]
                base_score = base_features(row, catalog_row)["score"]
                fuzzy_score = fuzzy_scores.get(candidate_index, 0.0)
                scored.append(
                    {
                        "vessel_id": str(catalog_row["vessel_id"]),
                        "catalog_name": catalog_row.get("vessel_name"),
                        "base_score": base_score,
                        "fuzzy_score": fuzzy_score,
                        "final_score": base_score + fuzzy_score_bonus * fuzzy_score,
                    }
                )
            top_score = max(item["final_score"] for item in scored)
            top_candidates = [
                item
                for item in scored
                if abs(item["final_score"] - top_score) <= TIE_TOLERANCE
            ]
            target_fuzzy_score = next(
                item["fuzzy_score"]
                for item in scored
                if item["vessel_id"] == str(case["target_vessel_id"])
            )
            for item in top_candidates:
                tie_records.append(
                    {
                        "seed": seed,
                        "sample_id": case["sample_id"],
                        "input_name": case["vessel_name"],
                        "target_vessel_id": str(case["target_vessel_id"]),
                        "baseline_prediction": case["baseline_prediction"],
                        "catalog_name_tiebreak_prediction": variant_predictions[
                            case["sample_id"]
                        ],
                        "candidate_contains_target": case["candidate_contains_target"],
                        "candidate_vessel_id": item["vessel_id"],
                        "candidate_name": item["catalog_name"],
                        "candidate_is_target": item["vessel_id"] == str(case["target_vessel_id"]),
                        "base_score": item["base_score"],
                        "fuzzy_score": item["fuzzy_score"],
                        "target_fuzzy_score": target_fuzzy_score,
                        "fuzzy_score_tied_with_target": abs(
                            item["fuzzy_score"] - target_fuzzy_score
                        ) <= TIE_TOLERANCE,
                        "final_score": item["final_score"],
                        "selected_by_vessel_id_tiebreak": (
                            item["vessel_id"] == str(case["baseline_prediction"])
                        ),
                    }
                )

    pd.DataFrame.from_records(tie_records).to_csv(
        result_dir / "tie_break_case_details.csv", index=False, encoding="utf-8-sig"
    )


def run_diagnostics_only(result_dir: Path) -> None:
    runtime = load_runtime()
    train, catalog = read_data()
    write_case_diagnostics(result_dir, train, catalog, runtime)
    print(f"Wrote tie and row diagnostics in {result_dir.relative_to(PROJECT_ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--diagnostics-only",
        type=Path,
        help="Add row and fuzzy-score tie diagnostics to an existing result directory.",
    )
    args = parser.parse_args()
    if args.diagnostics_only:
        run_diagnostics_only(args.diagnostics_only.resolve())
        return 0

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_dir = RESULTS_ROOT / f"exp014_recheck_{timestamp}"
    if result_dir.exists():
        raise FileExistsError(f"Refusing to overwrite result directory: {result_dir}")
    result_dir.mkdir(parents=True)

    runtime = load_runtime()
    AdaptiveCatalogIndex = runtime[0]
    train, catalog = read_data()
    baseline_results: dict[int, dict] = {}
    variant_results: dict[int, dict] = {}
    baseline_repeatability: dict[int, bool] = {}
    variant_repeatability: dict[int, bool] = {}
    valid_splits: dict[int, pd.DataFrame] = {}

    for seed in SEEDS:
        valid = load_validation_split(train, seed)
        valid_splits[seed] = valid
        index = AdaptiveCatalogIndex(catalog)
        baseline_first = evaluate_split(valid, catalog, index, runtime, "vessel_id")
        baseline_second = evaluate_split(valid, catalog, index, runtime, "vessel_id")
        baseline_repeatability[seed] = baseline_first["details"]["prediction"].equals(
            baseline_second["details"]["prediction"]
        )
        baseline_results[seed] = baseline_first
        baseline_first["details"].to_csv(
            result_dir / f"seed{seed}_fixed_40_50_60_details.csv",
            index=False,
            encoding="utf-8-sig",
        )

        variant_first = evaluate_split(valid, catalog, index, runtime, "catalog_name")
        variant_second = evaluate_split(valid, catalog, index, runtime, "catalog_name")
        variant_repeatability[seed] = variant_first["details"]["prediction"].equals(
            variant_second["details"]["prediction"]
        )
        variant_results[seed] = variant_first
        variant_first["details"].to_csv(
            result_dir / f"seed{seed}_catalog_name_tiebreak_details.csv",
            index=False,
            encoding="utf-8-sig",
        )

    write_case_diagnostics(result_dir, train, catalog, runtime)

    summary_rows = []
    all_gates_pass = all(baseline_repeatability.values()) and all(
        variant_repeatability.values()
    )
    for seed in SEEDS:
        base = baseline_results[seed]["metrics"]
        variant = variant_results[seed]["metrics"]
        delta = {
            metric: variant[metric] - base[metric]
            for metric in (
                "weighted",
                "korean_accuracy",
                "non_korean_accuracy",
                "korean_coverage",
                "overall_coverage",
                "mean_candidate_count",
            )
        }
        log_deltas = {
            metric: base[metric] - expected
            for metric, expected in EXPECTED_LOG[seed].items()
        }
        gates = {
            "weighted_not_lower": delta["weighted"] >= -1e-15,
            "korean_drop_below_0_005": delta["korean_accuracy"] > -0.005,
            "non_korean_not_lower": delta["non_korean_accuracy"] >= -1e-15,
            "korean_coverage_not_lower": delta["korean_coverage"] >= -1e-15,
            "overall_coverage_not_lower": delta["overall_coverage"] >= -1e-15,
            "baseline_repeatable": baseline_repeatability[seed],
            "variant_repeatable": variant_repeatability[seed],
        }
        all_gates_pass = all_gates_pass and all(gates.values())
        summary_rows.append(
            {
                "seed": seed,
                "baseline": base,
                "catalog_name_tiebreak": variant,
                "delta": delta,
                "difference_from_log": log_deltas,
                "gates": gates,
            }
        )

    baseline_package_validation = validate_submission_output(
        CURRENT_ZIP, valid_splits[42], catalog, result_dir
    )
    final_package = None
    if all_gates_pass:
        final_package = build_and_validate_variant_zip(
            valid_splits[42], catalog, result_dir
        )

    report = {
        "baseline_reference": {
            "experiment": "EXP-014 fixed_40_50_60",
            "leaderboard_score": 0.6763333333,
            "archive_name_in_log": "final_exp014_fixed_40_50_60.zip",
            "archive_present": str(CURRENT_ZIP.relative_to(PROJECT_ROOT)),
        },
        "seed_results": summary_rows,
        "candidate_missing_and_reranking_failures": {
            str(seed): {
                key: baseline_results[seed]["metrics"][key]
                for key in (
                    "candidate_missing_rows",
                    "reranking_failed_rows",
                    "korean_candidate_missing_rows",
                    "korean_reranking_failed_rows",
                    "non_korean_candidate_missing_rows",
                    "non_korean_reranking_failed_rows",
                    "top_tied_rows",
                    "vessel_id_tie_break_wrong_rows",
                    "tie_break_rows_corrected",
                    "new_errors_from_tie_policy",
                    "changed_predictions",
                    "valid_rows",
                    "korean_rows",
                )
            }
            for seed in SEEDS
        },
        "repeatability": {
            "baseline": {str(k): v for k, v in baseline_repeatability.items()},
            "catalog_name_tiebreak": {
                str(k): v for k, v in variant_repeatability.items()
            },
        },
        "all_acceptance_gates_pass": all_gates_pass,
        "existing_submission_package_validation": baseline_package_validation,
        "new_submission_package": final_package,
        "decision": (
            "new candidate package created; user approval required before leaderboard submission"
            if final_package
            else "retain EXP-014; do not create a new package or submit"
        ),
    }
    report_path = result_dir / "summary.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(
        [
            {
                "seed": row["seed"],
                "model": model,
                **row[model],
            }
            for row in summary_rows
            for model in ("baseline", "catalog_name_tiebreak")
        ]
    ).to_csv(result_dir / "score_comparison.csv", index=False, encoding="utf-8-sig")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(f"\nSaved results: {result_dir.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
