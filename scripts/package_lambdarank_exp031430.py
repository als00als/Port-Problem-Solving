"""Build and verify a separate exploratory LambdaRank submission candidate."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import lightgbm as lgb
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "open" / "train.csv"
CATALOG = ROOT / "data" / "open" / "vessel_catalog.csv"
BASELINE_ZIP = ROOT / "submissions" / "final_exp022.zip"
BASELINE_SHA256 = "fd7833eba4f0c417c226ca8054a828b3f1963195cd64331a63267f761d874255"
SPLIT_PATH = ROOT / "experiments" / "splits" / "agent_syllable_transliteration_cv_seed31426.json"
PREREG_PATH = ROOT / "experiments" / "splits" / "agent_learned_reranker_nested_oof_seed31430.json"
EXPERIMENT_RESULT = (
    ROOT
    / "experiments"
    / "results"
    / "agent_learned_reranker_nested_oof_20261018"
    / "summary_lgbm_4_6_0.json"
)
RESULT_DIR = ROOT / "experiments" / "results" / "agent_lambdarank_package_20261018"
OUTPUT_ZIP = ROOT / "submissions" / "candidate_exp031430_lambdarank.zip"
PACKAGE_REPORT = RESULT_DIR / "package_verification.json"
MAX_VALIDATION_SECONDS = 600.0

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submit" / "model"))

from adaptive_lib import AdaptiveCatalogIndex, normalize_name, romanized_name  # noqa: E402
from scripts.agent_learned_reranker_nested_oof import (  # noqa: E402
    FEATURE_NAMES,
    collect_training_queries,
    make_query_examples,
)
from scripts.agent_syllable_transliteration_probe import (  # noqa: E402
    file_sha256,
    learn_syllable_mappings,
)
from submit.model.ranker_runtime import make_ranker_query  # noqa: E402


EXPECTED_MEMBERS = {
    "model/adaptive_lib.py",
    "model/ranker_runtime.py",
    "model/syllable_runtime.py",
    "model/syllable_mappings.json",
    "model/lambdarank_model.txt",
    "model/training_metadata.json",
    "script.py",
    "requirements.txt",
}


def require_frozen_inputs() -> tuple[dict, dict, pd.DataFrame, pd.DataFrame]:
    if lgb.__version__ != "4.6.0":
        raise AssertionError(f"Expected LightGBM 4.6.0, found {lgb.__version__}")
    if file_sha256(BASELINE_ZIP) != BASELINE_SHA256:
        raise AssertionError("Frozen EXP-022 archive hash mismatch")
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    for filename, path in (("train.csv", TRAIN), ("vessel_catalog.csv", CATALOG)):
        if file_sha256(path) != prereg["data_hashes"][filename]:
            raise AssertionError(f"Frozen input hash mismatch: {filename}")
    experiment = json.loads(EXPERIMENT_RESULT.read_text(encoding="utf-8"))
    if experiment["lightgbm_version"] != "4.6.0":
        raise AssertionError("The referenced experiment result is not from LightGBM 4.6.0")
    if experiment["submission_promotion_eligible"]:
        raise AssertionError("Exploratory CV must not be treated as an independent promotion gate")

    train = pd.read_csv(TRAIN, encoding="utf-8-sig")
    catalog = pd.read_csv(CATALOG, encoding="utf-8-sig")
    group_to_fold = {
        str(group_id): int(fold["fold"])
        for fold in split["folds"]
        for group_id in fold["target_group_ids"]
    }
    if set(group_to_fold) != set(train["target_vessel_id"].astype(str)):
        raise AssertionError("Frozen grouped folds do not cover the official training targets")
    return prereg, split, train, catalog


def train_candidate_model(model_dir: Path) -> dict:
    prereg, split, train, catalog = require_frozen_inputs()
    index = AdaptiveCatalogIndex(catalog.reset_index(drop=True))
    group_to_fold = {
        str(group_id): int(fold["fold"])
        for fold in split["folds"]
        for group_id in fold["target_group_ids"]
    }

    all_features: list[list[float]] = []
    all_labels: list[int] = []
    all_group_sizes: list[int] = []
    fold_statistics: list[dict] = []
    skipped_queries = 0

    for fold in split["folds"]:
        fold_id = int(fold["fold"])
        heldout = train[
            train["target_vessel_id"].astype(str).map(group_to_fold).eq(fold_id)
        ].copy()
        mapping_train = train[
            train["target_vessel_id"].astype(str).map(group_to_fold).ne(fold_id)
        ].copy()
        if set(mapping_train["target_vessel_id"].astype(str)) & set(
            heldout["target_vessel_id"].astype(str)
        ):
            raise AssertionError(f"Target-group leakage while building fold {fold_id}")

        mappings, pair_count, aligned_count = learn_syllable_mappings(
            mapping_train, catalog, normalize_name, romanized_name
        )
        features, labels, groups, kept, skipped = collect_training_queries(
            heldout, index, mappings
        )
        if sum(labels) != kept:
            raise AssertionError(f"Unexpected positive labels in fold {fold_id}")
        all_features.extend(features)
        all_labels.extend(labels)
        all_group_sizes.extend(groups)
        skipped_queries += skipped
        fold_statistics.append(
            {
                "fold": fold_id,
                "mapping_fit_target_groups": int(mapping_train["target_vessel_id"].nunique()),
                "mapping_training_pairs": pair_count,
                "aligned_mapping_pairs": aligned_count,
                "ranker_training_queries": kept,
                "queries_without_target_candidate": skipped,
            }
        )

    if sum(all_group_sizes) != len(all_labels) or sum(all_labels) == 0:
        raise AssertionError("Invalid cross-fitted ranking training data")
    if sum(all_group_sizes) != len(all_features):
        raise AssertionError("Ranking feature and group counts differ")

    settings = prereg["ranker"]
    ranker = lgb.LGBMRanker(
        objective=settings["objective"],
        metric="ndcg",
        n_estimators=int(settings["n_estimators"]),
        learning_rate=float(settings["learning_rate"]),
        num_leaves=int(settings["num_leaves"]),
        max_depth=int(settings["max_depth"]),
        min_child_samples=int(settings["min_child_samples"]),
        reg_lambda=float(settings["reg_lambda"]),
        random_state=int(settings["random_state"]),
        n_jobs=int(settings["n_jobs"]),
        verbosity=-1,
        label_gain=settings["label_gain"],
        deterministic=True,
        force_col_wise=True,
    )
    ranker.fit(
        pd.DataFrame(all_features, columns=FEATURE_NAMES),
        all_labels,
        group=all_group_sizes,
    )
    ranker.booster_.save_model(str(model_dir / "lambdarank_model.txt"))

    full_mappings, full_pair_count, full_aligned_count = learn_syllable_mappings(
        train, catalog, normalize_name, romanized_name
    )
    (model_dir / "syllable_mappings.json").write_text(
        json.dumps(full_mappings, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    metadata = {
        "candidate": "EXP-031430 exploratory Korean LambdaRank",
        "python_major_minor": "3.11",
        "lightgbm_version": lgb.__version__,
        "ranker_features": list(FEATURE_NAMES),
        "ranker_settings": settings,
        "training_feature_method": "five-fold target-group OOF; mapping fit excludes each feature fold",
        "training_target_groups": int(train["target_vessel_id"].nunique()),
        "ranker_training_queries": int(len(all_group_sizes)),
        "ranker_training_queries_without_target_candidate": int(skipped_queries),
        "ranker_training_candidate_rows": int(len(all_labels)),
        "ranker_training_positive_rows": int(sum(all_labels)),
        "syllable_mapping_training_pairs": int(full_pair_count),
        "syllable_mapping_aligned_pairs": int(full_aligned_count),
        "folds": fold_statistics,
        "data_sha256": prereg["data_hashes"],
        "historical_validation_and_error_analysis_exposure": True,
        "independent_performance_estimate": False,
        "leaderboard_submitted": False,
    }
    (model_dir / "training_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return metadata


def create() -> None:
    if OUTPUT_ZIP.exists():
        raise FileExistsError(f"Refusing to overwrite existing candidate: {OUTPUT_ZIP}")
    if PACKAGE_REPORT.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {PACKAGE_REPORT}")

    with zipfile.ZipFile(BASELINE_ZIP) as base_archive:
        if set(base_archive.namelist()) != {
            "model/adaptive_lib.py",
            "script.py",
            "requirements.txt",
        }:
            raise AssertionError("Unexpected frozen EXP-022 package structure")
        adaptive_source = base_archive.read("model/adaptive_lib.py")
        requirements = base_archive.read("requirements.txt")
    script_source = (ROOT / "submit" / "lambdarank_script.py").read_bytes()
    runtime_source = (ROOT / "submit" / "model" / "ranker_runtime.py").read_bytes()
    syllable_source = (ROOT / "submit" / "model" / "syllable_runtime.py").read_bytes()

    with tempfile.TemporaryDirectory(prefix="exp031430_model_") as directory:
        model_dir = Path(directory)
        metadata = train_candidate_model(model_dir)
        with zipfile.ZipFile(OUTPUT_ZIP, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            archive.writestr("model/adaptive_lib.py", adaptive_source)
            archive.writestr("model/ranker_runtime.py", runtime_source)
            archive.writestr("model/syllable_runtime.py", syllable_source)
            archive.write(model_dir / "syllable_mappings.json", "model/syllable_mappings.json")
            archive.write(model_dir / "lambdarank_model.txt", "model/lambdarank_model.txt")
            archive.write(model_dir / "training_metadata.json", "model/training_metadata.json")
            archive.writestr("script.py", script_source)
            archive.writestr("requirements.txt", requirements)

    print(
        json.dumps(
            {
                "archive": str(OUTPUT_ZIP),
                "sha256": file_sha256(OUTPUT_ZIP),
                "members": sorted(EXPECTED_MEMBERS),
                "ranker_training_queries": metadata["ranker_training_queries"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def run_archive(root: Path, rows: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    rows.to_csv(root / "data" / "test.csv", index=False, encoding="utf-8-sig")
    started = time.perf_counter()
    environment = os.environ.copy()
    environment["HF_HUB_OFFLINE"] = "1"
    environment["TRANSFORMERS_OFFLINE"] = "1"
    completed = subprocess.run(
        [sys.executable, str(root / "script.py")],
        cwd=root,
        env=environment,
        text=True,
        capture_output=True,
        timeout=MAX_VALIDATION_SECONDS,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(f"Packaged script failed: {completed.stderr[-2500:]}")
    output = pd.read_csv(root / "output" / "submission.csv", encoding="utf-8-sig")
    if output.columns.tolist() != ["sample_id", "target_vessel_id"]:
        raise AssertionError("Submission columns changed")
    if len(output) != len(rows):
        raise AssertionError("Submission row count differs from input")
    if output["sample_id"].astype(str).tolist() != rows["sample_id"].astype(str).tolist():
        raise AssertionError("Submission sample_id order differs from input")
    if output["target_vessel_id"].isna().any():
        raise AssertionError("Submission contains missing predictions")
    return output, elapsed


def verify() -> None:
    if not OUTPUT_ZIP.exists():
        raise FileNotFoundError(OUTPUT_ZIP)
    if PACKAGE_REPORT.exists():
        raise FileExistsError(f"Refusing to overwrite existing report: {PACKAGE_REPORT}")
    prereg, split, train, catalog = require_frozen_inputs()
    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        if set(archive.namelist()) != EXPECTED_MEMBERS or archive.testzip() is not None:
            raise AssertionError("Candidate archive structure or CRC validation failed")
        if any(name.startswith(("data/", "experiments/", "scripts/")) for name in archive.namelist()):
            raise AssertionError("Training data or research code was included in the candidate")
        if archive.read("requirements.txt").strip() != b"# Runtime packages are preinstalled by the evaluation environment.":
            raise AssertionError("Submission dependency declaration differs from official policy")
        if archive.read("script.py") != (ROOT / "submit" / "lambdarank_script.py").read_bytes():
            raise AssertionError("Packaged entry point differs from candidate source")
        if archive.read("model/ranker_runtime.py") != (
            ROOT / "submit" / "model" / "ranker_runtime.py"
        ).read_bytes():
            raise AssertionError("Packaged inference runtime differs from candidate source")

        package_metadata = json.loads(archive.read("model/training_metadata.json"))
        if package_metadata["data_sha256"] != prereg["data_hashes"]:
            raise AssertionError("Packaged model metadata has incorrect data hashes")
        if package_metadata["lightgbm_version"] != "4.6.0":
            raise AssertionError("Packaged ranker was not created with LightGBM 4.6.0")

        source = archive.read("model/ranker_runtime.py").decode("utf-8").lower()
        if any(token in source for token in ("requests.", "urllib.request", "http://", "https://")):
            raise AssertionError("Inference runtime contains an internet access path")

    group_to_fold = {
        str(group_id): int(fold["fold"])
        for fold in split["folds"]
        for group_id in fold["target_group_ids"]
    }
    validation_fold = min(int(fold["fold"]) for fold in split["folds"])
    valid = train[
        train["target_vessel_id"].astype(str).map(group_to_fold).eq(validation_fold)
    ]
    supplemental = train[~train["sample_id"].astype(str).isin(valid["sample_id"].astype(str))]
    input_frame = pd.concat([valid, supplemental], ignore_index=True).head(3000)
    input_rows = input_frame.drop(columns=["target_vessel_id"]).copy()
    if len(input_rows) != 3000 or input_rows["sample_id"].astype(str).duplicated().any():
        raise AssertionError("Could not create 3,000 unique official-format validation rows")

    # Confirm package feature generation is byte-for-byte aligned with the frozen probe.
    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        package_mappings = json.loads(archive.read("model/syllable_mappings.json"))
    index = AdaptiveCatalogIndex(catalog.reset_index(drop=True))
    from submit.model.ranker_runtime import make_ranker_query as runtime_make_query

    for row in valid[valid["vessel_name"].map(lambda value: isinstance(value, str) and any("가" <= c <= "힣" for c in value))].head(20).to_dict("records"):
        expected = make_query_examples(index, row, package_mappings)
        actual = runtime_make_query(index, row, package_mappings)
        if expected["candidate_ids"] != actual["candidate_ids"] or expected["features"] != actual["features"]:
            raise AssertionError("Packaged feature generation differs from the frozen experiment")

    with tempfile.TemporaryDirectory(prefix="exp031430_package_verify_") as directory:
        root = Path(directory)
        with zipfile.ZipFile(OUTPUT_ZIP) as archive:
            archive.extractall(root)
        (root / "data").mkdir()
        catalog.to_csv(root / "data" / "vessel_catalog.csv", index=False, encoding="utf-8-sig")
        first, first_seconds = run_archive(root, input_rows)
        second, second_seconds = run_archive(root, input_rows)
        if not first.equals(second):
            raise AssertionError("Repeated inference changed predictions")

        expected = dict(zip(first["sample_id"].astype(str), first["target_vessel_id"].astype(str)))
        shuffled_rows = input_rows.iloc[:10].sample(frac=1.0, random_state=31430).reset_index(drop=True)
        shuffled, _ = run_archive(root, shuffled_rows)
        if any(
            expected[str(sample_id)] != str(prediction)
            for sample_id, prediction in shuffled.itertuples(index=False, name=None)
        ):
            raise AssertionError("Shuffled subset changed a row's prediction")
        single, _ = run_archive(root, input_rows.iloc[[0]])
        if str(single.iloc[0]["target_vessel_id"]) != expected[str(input_rows.iloc[0]["sample_id"])]:
            raise AssertionError("Single-row run changed the row's prediction")

    if not set(first["target_vessel_id"].astype(str)).issubset(set(catalog["vessel_id"].astype(str))):
        raise AssertionError("Candidate produced an ID absent from the official catalog")
    elapsed = max(first_seconds, second_seconds)
    if elapsed > MAX_VALIDATION_SECONDS:
        raise AssertionError(f"3,000-row run exceeded the {MAX_VALIDATION_SECONDS}s budget")

    report = {
        "archive": str(OUTPUT_ZIP.relative_to(ROOT)),
        "archive_sha256": file_sha256(OUTPUT_ZIP),
        "archive_members": sorted(EXPECTED_MEMBERS),
        "lightgbm_version": lgb.__version__,
        "validation_rows": len(input_rows),
        "output_rows": len(first),
        "output_columns": first.columns.tolist(),
        "all_predictions_in_official_catalog": True,
        "same_input_repeatable": True,
        "shuffled_subset_and_single_row_independent": True,
        "feature_generation_matches_experiment": True,
        "internet_access_in_inference_runtime": False,
        "full_3000_row_run_seconds": [first_seconds, second_seconds],
        "runtime_budget_seconds": MAX_VALIDATION_SECONDS,
        "historical_group_exposure": True,
        "independent_performance_estimate": False,
        "leaderboard_submitted": False,
    }
    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    PACKAGE_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in {"create", "verify"}:
        raise SystemExit("Usage: python scripts/package_lambdarank_exp031430.py create|verify")
    create() if sys.argv[1] == "create" else verify()
