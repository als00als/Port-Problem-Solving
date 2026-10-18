"""Freeze seed31438 expanded-negative CatBoost as a separate DACON archive."""

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

import catboost
import joblib
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submit" / "model"))

from adaptive_lib import AdaptiveCatalogIndex, is_korean, normalize_name, romanized_name  # noqa: E402
from scripts import agent_catboost_char_ngram_candidate as source  # noqa: E402
from scripts.agent_catboost_ngram_negative_ablation import append_training_queries, fit_ranker  # noqa: E402
from scripts.agent_learned_reranker_nested_oof import FEATURE_NAMES  # noqa: E402
from scripts.agent_syllable_transliteration_probe import file_sha256, learn_syllable_mappings  # noqa: E402
from submit.model.catboost_ngram_runtime import make_catboost_query  # noqa: E402

DATA = ROOT / "data" / "open"
SPLIT = ROOT / "experiments/splits/agent_syllable_transliteration_cv_seed31426.json"
PREREG = ROOT / "experiments/splits/agent_catboost_ngram_negative_ablation_seed31438.json"
RANKER_PREREG = ROOT / "experiments/splits/agent_catboost_ranker_nested_oof_seed31432.json"
EXPERIMENT = ROOT / "experiments/results/agent_catboost_ngram_negative_ablation_20261018/summary.json"
ZIP = ROOT / "submissions/candidate_catboost_ngram_negatives_seed31438.zip"
REPORT = ROOT / "experiments/results/agent_catboost_ngram_package_20261018/package_verification.json"
PYTHON = ROOT / ".venv_competition_py311/Scripts/python.exe"
OLD_ZIPS = {
    "final_exp022.zip": "fd7833eba4f0c417c226ca8054a828b3f1963195cd64331a63267f761d874255",
    "candidate_lambdarank.zip": "39cd14a82f06a3441981b71678baf677a1f384bac8c8cca4a533fd90446d0362",
}
MEMBERS = {
    "model/adaptive_lib.py": ROOT / "submit/model/adaptive_lib.py",
    "model/syllable_runtime.py": ROOT / "submit/model/syllable_runtime.py",
    "model/catboost_ngram_runtime.py": ROOT / "submit/model/catboost_ngram_runtime.py",
    "script.py": ROOT / "submit/catboost_ngram_script.py",
    "requirements.txt": ROOT / "submit/requirements.txt",
}
GENERATED = {
    "model/syllable_mappings.json",
    "model/ngram_vectorizer.joblib",
    "model/catboost_ranker.cbm",
    "model/training_metadata.json",
}


def frozen_inputs():
    if sys.version_info[:2] != (3, 11) or catboost.__version__ != "1.2.8":
        raise AssertionError("Packaging requires Python 3.11 and CatBoost 1.2.8")
    for name, digest in OLD_ZIPS.items():
        if file_sha256(ROOT / "submissions" / name) != digest:
            raise AssertionError(f"Existing submission changed: {name}")
    prereg = json.loads(PREREG.read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    experiment = json.loads(EXPERIMENT.read_text(encoding="utf-8"))
    settings = json.loads(RANKER_PREREG.read_text(encoding="utf-8"))["method"]["ranker"]
    if experiment["catboost_version"] != catboost.__version__ or experiment["independent_performance_estimate"]:
        raise AssertionError("Experiment provenance mismatch")
    for filename, digest in prereg["data_hashes"].items():
        if file_sha256(DATA / filename) != digest:
            raise AssertionError(f"Official input changed: {filename}")
    train = pd.read_csv(DATA / "train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(DATA / "vessel_catalog.csv", encoding="utf-8-sig")
    group_to_fold = {str(group_id): int(fold["fold"]) for fold in split["folds"] for group_id in fold["target_group_ids"]}
    if set(group_to_fold) != set(train["target_vessel_id"].astype(str)):
        raise AssertionError("Frozen group split differs from training targets")
    return prereg, split, settings, train, catalog, group_to_fold


def train_model(model_dir: Path):
    prereg, split, settings, train, catalog, group_to_fold = frozen_inputs()
    index = AdaptiveCatalogIndex(catalog.reset_index(drop=True))
    source._NGRAM_CACHE.clear()
    source._VECTORIZER = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=True, norm="l2", dtype=np.float32)
    source._CATALOG_TFIDF = source._VECTORIZER.fit_transform(index.romanized_names)
    data = {arm: {"features": [], "labels": [], "groups": [], "queries": 0} for arm in ("original_negatives", "expanded_negatives")}
    fold_reports = []
    for fold in split["folds"]:
        fold_id = int(fold["fold"])
        fold_ids = train["target_vessel_id"].astype(str).map(group_to_fold)
        heldout = train[fold_ids.eq(fold_id)].copy()
        fit_rows = train[fold_ids.ne(fold_id)].copy()
        if set(heldout["target_vessel_id"].astype(str)) & set(fit_rows["target_vessel_id"].astype(str)):
            raise AssertionError("Syllable mapping target-group leakage")
        mappings, _, _ = learn_syllable_mappings(fit_rows, catalog, normalize_name, romanized_name)
        counts = append_training_queries(heldout, index, mappings, data)
        fold_reports.append({"fold": fold_id, **counts})
        print(f"Generated training fold {fold_id}: {counts}", flush=True)
    if data["original_negatives"]["queries"] != data["expanded_negatives"]["queries"]:
        raise AssertionError("Training arms have different query counts")
    ranker, fit_seconds = fit_ranker(data["expanded_negatives"], settings, "CPU")
    ranker.save_model(str(model_dir / "catboost_ranker.cbm"))
    mappings, pair_count, aligned_count = learn_syllable_mappings(train, catalog, normalize_name, romanized_name)
    (model_dir / "syllable_mappings.json").write_text(json.dumps(mappings, ensure_ascii=False, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    joblib.dump(source._VECTORIZER, model_dir / "ngram_vectorizer.joblib")
    metadata = {
        "candidate": "CatBoost seed31438 expanded n-gram negatives",
        "python_major_minor": "3.11",
        "catboost_version": catboost.__version__,
        "sklearn_vectorizer": "char TF-IDF 2-4 grams, float32, official catalog names only",
        "ranker_features": list(FEATURE_NAMES),
        "ranker_settings": settings,
        "selected_training_device": "CPU",
        "ranker_fit_seconds": fit_seconds,
        "ranker_training_queries": data["expanded_negatives"]["queries"],
        "ranker_training_candidate_rows": len(data["expanded_negatives"]["labels"]),
        "ranker_training_positive_rows": sum(data["expanded_negatives"]["labels"]),
        "added_ngram_negative_training_rows": len(data["expanded_negatives"]["labels"]) - len(data["original_negatives"]["labels"]),
        "training_feature_method": "Five-fold target-group OOF syllable mapping; n-gram vectorizer fit on official catalog only",
        "syllable_mapping_training_pairs": pair_count,
        "syllable_mapping_aligned_pairs": aligned_count,
        "folds": fold_reports,
        "data_sha256": prereg["data_hashes"],
        "experiment_summary_sha256": file_sha256(EXPERIMENT),
        "historical_validation_and_error_analysis_exposure": True,
        "independent_performance_estimate": False,
        "leaderboard_submitted": False,
    }
    (model_dir / "training_metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metadata


def create():
    if ZIP.exists() or REPORT.exists():
        raise FileExistsError("Frozen package or report already exists")
    with tempfile.TemporaryDirectory(prefix="catboost_ngram_model_") as directory:
        model_dir = Path(directory)
        metadata = train_model(model_dir)
        with zipfile.ZipFile(ZIP, "x", compression=zipfile.ZIP_DEFLATED) as archive:
            for name, path in MEMBERS.items():
                archive.write(path, name)
            for name in sorted(GENERATED):
                archive.write(model_dir / Path(name).name, name)
    print(json.dumps({"zip": str(ZIP), "sha256": file_sha256(ZIP), "training_queries": metadata["ranker_training_queries"]}, ensure_ascii=False, indent=2))


def run_archive(root: Path, rows: pd.DataFrame):
    rows.to_csv(root / "data/test.csv", index=False, encoding="utf-8-sig")
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    started = time.perf_counter()
    process = subprocess.run([str(PYTHON), str(root / "script.py")], cwd=root, env=env, capture_output=True, text=True, timeout=600)
    elapsed = time.perf_counter() - started
    if process.returncode:
        raise RuntimeError(process.stderr[-3000:])
    output = pd.read_csv(root / "output/submission.csv", encoding="utf-8-sig")
    if output.columns.tolist() != ["sample_id", "target_vessel_id"] or len(output) != len(rows):
        raise AssertionError("Invalid submission shape")
    if output["sample_id"].astype(str).tolist() != rows["sample_id"].astype(str).tolist():
        raise AssertionError("Output row order differs from input")
    if output["target_vessel_id"].isna().any():
        raise AssertionError("Missing prediction")
    return output, elapsed


def verify():
    if REPORT.exists():
        raise FileExistsError("Frozen package report already exists")
    prereg, split, _, train, catalog, group_to_fold = frozen_inputs()
    with zipfile.ZipFile(ZIP) as archive:
        names = archive.namelist()
        if set(names) != set(MEMBERS) | GENERATED or len(names) != len(set(names)) or archive.testzip() is not None:
            raise AssertionError("Archive member or CRC mismatch")
        for name, path in MEMBERS.items():
            if archive.read(name) != path.read_bytes():
                raise AssertionError(f"Package source mismatch: {name}")
        metadata = json.loads(archive.read("model/training_metadata.json"))
        if metadata["data_sha256"] != prereg["data_hashes"] or metadata["catboost_version"] != "1.2.8":
            raise AssertionError("Package model metadata mismatch")
        if archive.read("requirements.txt").strip() != b"# Runtime packages are preinstalled by the evaluation environment.":
            raise AssertionError("Unexpected runtime requirements")
        for name in names:
            if name.endswith(".py") and any(token in archive.read(name).decode("utf-8").lower() for token in ("requests.", "urllib.request", "http://", "https://")):
                raise AssertionError(f"Network access in {name}")
        member_hashes = {name: hashlib.sha256(archive.read(name)).hexdigest() for name in names}

    index = AdaptiveCatalogIndex(catalog.reset_index(drop=True))
    with tempfile.TemporaryDirectory(prefix="catboost_ngram_verify_") as directory:
        root = Path(directory)
        with zipfile.ZipFile(ZIP) as archive:
            archive.extractall(root)
        mappings = json.loads((root / "model/syllable_mappings.json").read_text(encoding="utf-8"))
        vectorizer = joblib.load(root / "model/ngram_vectorizer.joblib")
        source._VECTORIZER = vectorizer
        source._CATALOG_TFIDF = vectorizer.transform(index.romanized_names)
        source._NGRAM_CACHE.clear()
        parity_rows = train[train["vessel_name"].map(is_korean)].head(20)
        for row in parity_rows.to_dict("records"):
            expected = source.make_treatment_query(index, row, mappings)
            actual = make_catboost_query(index, row, mappings, vectorizer, source._CATALOG_TFIDF)
            if expected["candidate_ids"] != actual["candidate_ids"] or not np.allclose(expected["features"], actual["features"], atol=1e-6, rtol=0):
                raise AssertionError("Packaged candidate generation or features differ from experiment")
        (root / "data").mkdir()
        catalog.to_csv(root / "data/vessel_catalog.csv", index=False, encoding="utf-8-sig")
        fold_ids = train["target_vessel_id"].astype(str).map(group_to_fold)
        validation = train[fold_ids.eq(0)]
        supplemental = train[fold_ids.ne(0)]
        input_rows = pd.concat([validation, supplemental], ignore_index=True).head(3000).drop(columns=["target_vessel_id"])
        if len(input_rows) != 3000 or input_rows["sample_id"].astype(str).duplicated().any():
            raise AssertionError("Could not construct 3000 unique official rows")
        first, first_seconds = run_archive(root, input_rows)
        second, second_seconds = run_archive(root, input_rows)
        if not first.equals(second):
            raise AssertionError("Repeated predictions differ")
        by_id = dict(zip(first["sample_id"].astype(str), first["target_vessel_id"].astype(str)))
        sample = pd.concat([input_rows[input_rows["vessel_name"].map(is_korean)].head(5), input_rows[~input_rows["vessel_name"].map(is_korean)].head(5)])
        shuffled, shuffled_seconds = run_archive(root, sample.sample(frac=1, random_state=31438))
        if any(by_id[str(row.sample_id)] != str(row.target_vessel_id) for row in shuffled.itertuples(index=False)):
            raise AssertionError("Shuffled subset changed predictions")
        singles = []
        for korean in (True, False):
            row = sample[sample["vessel_name"].map(is_korean).eq(korean)].head(1)
            one, seconds = run_archive(root, row)
            if by_id[str(row.iloc[0]["sample_id"])] != str(one.iloc[0]["target_vessel_id"]):
                raise AssertionError("Single-row prediction changed")
            singles.append(seconds)
    if not set(first["target_vessel_id"].astype(str)).issubset(set(catalog["vessel_id"].astype(str))):
        raise AssertionError("Prediction outside official catalog")
    if max(first_seconds, second_seconds) > 600:
        raise AssertionError("3000-row runtime exceeds 600 seconds")
    report = {
        "archive": str(ZIP.relative_to(ROOT)), "archive_sha256": file_sha256(ZIP),
        "members_sha256": member_hashes, "catboost_version": catboost.__version__,
        "training_queries": metadata["ranker_training_queries"],
        "training_candidate_rows": metadata["ranker_training_candidate_rows"],
        "training_positive_rows": metadata["ranker_training_positive_rows"],
        "added_ngram_negative_training_rows": metadata["added_ngram_negative_training_rows"],
        "validation_rows": len(input_rows), "output_columns": first.columns.tolist(),
        "all_predictions_in_official_catalog": True, "repeatable": True,
        "shuffled_and_single_row_independent": True, "feature_generation_matches_experiment": True,
        "full_3000_row_seconds": [first_seconds, second_seconds],
        "subset_seconds": shuffled_seconds, "single_row_seconds": singles,
        "historical_validation_and_error_analysis_exposure": True,
        "independent_performance_estimate": False, "leaderboard_submitted": False,
        "preserved_zip_sha256": {name: file_sha256(ROOT / "submissions" / name) for name in OLD_ZIPS},
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("create", "verify"):
        raise SystemExit("Usage: python scripts/package_catboost_ngram_seed31438.py create|verify")
    create() if sys.argv[1] == "create" else verify()
