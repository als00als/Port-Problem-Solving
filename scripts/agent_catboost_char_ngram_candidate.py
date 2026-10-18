"""Evaluate an added WRatio retrieval channel with fixed CatBoost nested OOF."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import catboost
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submit" / "model"))

from adaptive_lib import base_features, is_korean, normalize_name, positive_float, romanized_name  # noqa: E402
from scripts.agent_catboost_ranker_nested_oof import create_ranker  # noqa: E402
from scripts.agent_learned_reranker_nested_oof import FEATURE_NAMES, make_query_examples  # noqa: E402
from scripts.agent_syllable_transliteration_probe import (  # noqa: E402
    MAX_UNION_CANDIDATES,
    SPLIT_PATH,
    file_sha256,
    generate_variants,
    learn_syllable_mappings,
)

PREREG_PATH = ROOT / "experiments/splits/agent_catboost_char_ngram_candidate_seed31437.json"
RESULT_PATH = ROOT / "experiments/results/agent_catboost_char_ngram_candidate_20261018/summary.json"
SPLIT_PATH = ROOT / "experiments/splits/agent_syllable_transliteration_cv_seed31426.json"
CAT_PREREG_PATH = ROOT / "experiments/splits/agent_catboost_ranker_nested_oof_seed31432.json"
CAT_RESULT_PATH = ROOT / "experiments/results/agent_catboost_ranker_nested_oof_20261018/summary_catboost_1_2_8.json"
DATA_ROOT = ROOT / "data/open"
NGRAM_LIMIT = 40
NGRAM_CUTOFF = 0.10
_NGRAM_CACHE: dict[str, list[tuple[int, float]]] = {}
_VECTORIZER: TfidfVectorizer | None = None
_CATALOG_TFIDF = None


def make_treatment_query(index, row: dict, mappings: dict) -> dict:
    control = make_query_examples(index, row, mappings)
    candidate_ids = list(control["candidate_ids"])
    feature_rows = list(control["features"])
    seen = set(candidate_ids)
    variants = generate_variants(row["vessel_name"], mappings, romanized_name)
    queries = [romanized_name(row["vessel_name"]), *variants]
    retrieval_scores: dict[int, float] = {}
    for query in queries:
        if not query:
            continue
        cached = _NGRAM_CACHE.get(query)
        if cached is None:
            query_tfidf = _VECTORIZER.transform([query])
            scores = (_CATALOG_TFIDF @ query_tfidf.T).toarray().ravel()
            eligible = np.flatnonzero(scores >= NGRAM_CUTOFF)
            top_indices = sorted(
                eligible,
                key=lambda candidate_index: (-float(scores[candidate_index]), index.ids[candidate_index]),
            )[:NGRAM_LIMIT]
            cached = [
                (int(candidate_index), float(scores[candidate_index]))
                for candidate_index in top_indices
            ]
            _NGRAM_CACHE[query] = cached
        for candidate_index, score in cached:
            retrieval_scores[candidate_index] = max(retrieval_scores.get(candidate_index, 0.0), score)

    remaining = max(0, MAX_UNION_CANDIDATES - len(candidate_ids))
    added_indices = sorted(
        (candidate_index for candidate_index in retrieval_scores if index.ids[candidate_index] not in seen),
        key=lambda candidate_index: (-retrieval_scores[candidate_index], index.ids[candidate_index]),
    )[:remaining]
    for candidate_index in added_indices:
        vessel_id = index.ids[candidate_index]
        catalog_row = index.catalog.iloc[candidate_index]
        features = base_features(row, catalog_row)
        feature_rows.append(
            [
                float(features["exact_match"]),
                float(features["name_ratio"]),
                float(features["prefix_ratio"]),
                float(features["length_ratio"]),
                float(features["tonnage_ratio"]),
                float(features["score"]),
                0.0,
                float(retrieval_scores[candidate_index]),
                0.0,
                1.0,
                float(positive_float(row.get("gross_tonnage")) is not None),
                float(positive_float(row.get("international_gross_tonnage")) is not None),
                float(positive_float(catalog_row.get("gross_tonnage")) is not None),
            ]
        )
        candidate_ids.append(vessel_id)
        seen.add(vessel_id)

    treatment = dict(control)
    treatment.update(
        {
            "features": feature_rows,
            "candidate_ids": candidate_ids,
            "candidate_count": len(candidate_ids),
            "ngram_added_candidates": len(added_indices),
        }
    )
    return treatment


def collect_queries(frame: pd.DataFrame, index, mappings: dict, treatment: bool):
    features: list[list[float]] = []
    labels: list[int] = []
    group_sizes: list[int] = []
    kept = 0
    skipped = 0
    for row in frame.to_dict("records"):
        if not is_korean(row.get("vessel_name")):
            continue
        query = make_treatment_query(index, row, mappings) if treatment else make_query_examples(index, row, mappings)
        query_labels = [int(candidate_id == query["target"]) for candidate_id in query["candidate_ids"]]
        if not any(query_labels):
            skipped += 1
            continue
        features.extend(query["features"])
        labels.extend(query_labels)
        group_sizes.append(len(query_labels))
        kept += 1
    return features, labels, group_sizes, kept, skipped


def rank_prediction(model, query: dict) -> str:
    if not query["candidate_ids"]:
        return ""
    frame = pd.DataFrame(query["features"], columns=FEATURE_NAMES)
    scores = model.predict(frame)
    best_index = min(
        range(len(query["candidate_ids"])),
        key=lambda i: (-float(scores[i]), query["candidate_ids"][i]),
    )
    return query["candidate_ids"][best_index]


def summarize(rows: list[dict], prediction_key: str) -> dict:
    frame = pd.DataFrame(rows)
    target = frame["target"]
    correct = frame[prediction_key] == target
    arm = prediction_key.removesuffix("_prediction")
    return {
        "korean_rows": int(len(frame)),
        "top1": float(correct.mean()),
        "baseline_candidate_recall": float(frame["baseline_has_target"].mean()),
        "candidate_recall": float(frame[f"{arm}_has_target"].mean()),
        "mean_candidate_count": float(frame[f"{arm}_candidate_count"].mean()),
        "max_candidate_count": int(frame[f"{arm}_candidate_count"].max()),
        "ngram_added_candidates_mean": float(frame["ngram_added_candidates"].mean()),
    }


def probe_catboost_gpu(settings: dict) -> bool:
    if len(sys.argv) > 1 and sys.argv[1] == "--probe-catboost-gpu":
        probe = CatBoostRanker(
            loss_function=settings["loss_function"], iterations=2, learning_rate=0.05,
            depth=3, random_seed=int(settings["random_seed"]), thread_count=1,
            task_type="GPU", verbose=False, allow_writing_files=False,
        )
        pool = Pool([[0.0], [1.0], [0.2], [0.8]], label=[1, 0, 1, 0], group_id=[0, 0, 1, 1])
        probe.fit(pool)
        print(json.dumps({"backend": "CUDA"}))
        return True
    done = subprocess.run(
        [sys.executable, str(Path(__file__).resolve()), "--probe-catboost-gpu"],
        capture_output=True, text=True, timeout=90, check=False,
    )
    return done.returncode == 0


def evaluate() -> dict:
    if RESULT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite result: {RESULT_PATH}")
    if catboost.__version__ != "1.2.8":
        raise AssertionError(f"Expected CatBoost 1.2.8, found {catboost.__version__}")
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    cat_prereg = json.loads(CAT_PREREG_PATH.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    for filename, expected_hash in prereg["data_hashes"].items():
        if file_sha256(DATA_ROOT / filename) != expected_hash:
            raise AssertionError(f"Frozen input hash mismatch: {filename}")
    train = pd.read_csv(DATA_ROOT / "train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(DATA_ROOT / "vessel_catalog.csv", encoding="utf-8-sig")
    group_to_fold = {
        str(group_id): int(fold["fold"])
        for fold in split["folds"] for group_id in fold["target_group_ids"]
    }
    if set(group_to_fold) != set(train["target_vessel_id"].astype(str)):
        raise AssertionError("Frozen split does not cover target groups exactly")
    global _VECTORIZER, _CATALOG_TFIDF
    index = __import__("adaptive_lib").AdaptiveCatalogIndex(catalog.reset_index(drop=True))
    _VECTORIZER = TfidfVectorizer(
        analyzer="char", ngram_range=(2, 4), lowercase=True, norm="l2", dtype=np.float32
    )
    _CATALOG_TFIDF = _VECTORIZER.fit_transform(index.romanized_names)
    settings = cat_prereg["method"]["ranker"]
    use_gpu = probe_catboost_gpu(settings)
    device = "GPU" if use_gpu else "CPU"
    fold_reports = []
    observations = []
    fit_seconds = {"control": 0.0, "treatment": 0.0}
    generation_seconds = 0.0

    for outer in split["folds"]:
        outer_id = int(outer["fold"])
        outer_group_ids = train["target_vessel_id"].astype(str).map(group_to_fold)
        outer_train = train[outer_group_ids.ne(outer_id)].copy()
        outer_valid = train[outer_group_ids.eq(outer_id)].copy()
        outer_valid_groups = set(outer_valid["target_vessel_id"].astype(str))
        if outer_valid_groups & set(outer_train["target_vessel_id"].astype(str)):
            raise AssertionError(f"Outer target leakage in fold {outer_id}")
        train_data = {arm: {"x": [], "y": [], "sizes": [], "kept": 0, "skipped": 0} for arm in ("control", "treatment")}
        inner_reports = []
        for inner in split["folds"]:
            inner_id = int(inner["fold"])
            if inner_id == outer_id:
                continue
            inner_ids = outer_train["target_vessel_id"].astype(str).map(group_to_fold)
            inner_valid = outer_train[inner_ids.eq(inner_id)].copy()
            inner_train = outer_train[inner_ids.ne(inner_id)].copy()
            inner_valid_groups = set(inner_valid["target_vessel_id"].astype(str))
            if inner_valid_groups & set(inner_train["target_vessel_id"].astype(str)) or inner_valid_groups & outer_valid_groups:
                raise AssertionError(f"Nested target leakage in outer {outer_id}, inner {inner_id}")
            mappings, _, _ = learn_syllable_mappings(inner_train, catalog, normalize_name, romanized_name)
            for arm in ("control", "treatment"):
                started = time.perf_counter()
                x, y, sizes, kept, skipped = collect_queries(inner_valid, index, mappings, arm == "treatment")
                generation_seconds += time.perf_counter() - started
                target = train_data[arm]
                target["x"].extend(x)
                target["y"].extend(y)
                target["sizes"].extend(sizes)
                target["kept"] += kept
                target["skipped"] += skipped
            inner_reports.append({"inner_fold": inner_id, "training_groups": len(inner_train), "heldout_groups": len(inner_valid_groups)})

        models = {}
        for arm in ("control", "treatment"):
            data = train_data[arm]
            if not data["sizes"] or not any(data["y"]):
                raise AssertionError(f"No positive CatBoost examples for {arm}, outer fold {outer_id}")
            frame = pd.DataFrame(data["x"], columns=FEATURE_NAMES)
            group_ids = [qid for qid, size in enumerate(data["sizes"]) for _ in range(size)]
            pool = Pool(frame, label=data["y"], group_id=group_ids, feature_names=list(FEATURE_NAMES))
            ranker = create_ranker(settings)
            if use_gpu:
                ranker.set_params(task_type="GPU")
            started = time.perf_counter()
            ranker.fit(pool)
            fit_seconds[arm] += time.perf_counter() - started
            models[arm] = ranker

        mappings, _, _ = learn_syllable_mappings(outer_train, catalog, normalize_name, romanized_name)
        fold_rows = []
        for row in outer_valid.to_dict("records"):
            if not is_korean(row.get("vessel_name")):
                continue
            started = time.perf_counter()
            control = make_query_examples(index, row, mappings)
            treatment = make_treatment_query(index, row, mappings)
            generation_seconds += time.perf_counter() - started
            target_id = str(row["target_vessel_id"])
            control_prediction = rank_prediction(models["control"], control)
            treatment_prediction = rank_prediction(models["treatment"], treatment)
            item = {
                "target": target_id,
                "baseline_has_target": target_id in {index.ids[i] for i in control["baseline_indices"]},
                "control_has_target": target_id in control["candidate_ids"],
                "treatment_has_target": target_id in treatment["candidate_ids"],
                "control_candidate_count": control["candidate_count"],
                "treatment_candidate_count": treatment["candidate_count"],
                "ngram_added_candidates": treatment["ngram_added_candidates"],
                "control_prediction": control_prediction,
                "treatment_prediction": treatment_prediction,
            }
            observations.append(item)
            fold_rows.append(item)
        fold_reports.append({
            "outer_fold": outer_id,
            "valid_target_groups": len(outer_valid_groups),
            "inner_training_queries_kept": {arm: train_data[arm]["kept"] for arm in train_data},
            "inner_training_queries_without_target_candidate": {arm: train_data[arm]["skipped"] for arm in train_data},
            "catboost_fit_seconds": {arm: fit_seconds[arm] for arm in fit_seconds},
            "metrics": {arm: summarize(fold_rows, f"{arm}_prediction") for arm in ("control", "treatment")},
            "inner_folds": inner_reports,
        })

    metrics = {arm: summarize(observations, f"{arm}_prediction") for arm in ("control", "treatment")}
    frozen = json.loads(CAT_RESULT_PATH.read_text(encoding="utf-8"))
    control_reproduced = abs(
        metrics["control"]["top1"] - frozen["metrics"]["learned_ranker_top1"]
    ) <= 1e-12
    if device == "CPU" and not control_reproduced:
        raise AssertionError("Control CatBoost does not reproduce the frozen seed31432 result")
    gates = prereg["acceptance_gates"]
    gates_result = {
        "candidate_recall_gain": metrics["treatment"]["candidate_recall"] - 0.945424013434089 >= gates["korean_target_candidate_recall_absolute_gain_over_frozen_union_min"],
        "mean_candidate_count": metrics["treatment"]["mean_candidate_count"] <= gates["mean_candidate_count_max"],
        "max_candidate_count": metrics["treatment"]["max_candidate_count"] <= gates["max_candidate_count_max"],
        "treatment_top1_at_least_frozen_catboost": metrics["treatment"]["top1"] >= gates["treatment_catboost_top1_min"],
        "treatment_top1_not_below_same_run_control": metrics["treatment"]["top1"] >= metrics["control"]["top1"],
        "no_target_group_leakage": True,
    }
    report = {
        "experiment": prereg["experiment"],
        "preregistration": str(PREREG_PATH.relative_to(ROOT)),
        "source_split": str(SPLIT_PATH.relative_to(ROOT)),
        "historical_validation_overlap": True,
        "historical_error_analysis_overlap": True,
        "independent_performance_estimate": False,
        "data_hashes": prereg["data_hashes"],
        "python_version": sys.version.split()[0],
        "catboost_version": catboost.__version__,
        "selected_device": device,
        "control_exact_cpu_reference_reproduction": control_reproduced,
        "candidate_generation": prereg["method"],
        "metrics": metrics,
        "treatment_deltas": {
            key: metrics["treatment"][key] - metrics["control"][key]
            for key in ("top1", "candidate_recall", "mean_candidate_count", "ngram_added_candidates_mean")
        },
        "folds": fold_reports,
        "timing": {"feature_generation_seconds_total": generation_seconds, "catboost_fit_seconds_total": fit_seconds},
        "acceptance_gates": gates,
        "gate_results": gates_result,
        "exploratory_gates_passed": all(gates_result.values()),
        "submission_promotion_eligible": False,
        "submission_code_or_archive_modified": False,
        "leaderboard_submitted": False,
        "per_row_inputs_predictions_or_candidate_lists_saved": False,
        "limitations": prereg["limitations"],
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--probe-catboost-gpu":
        settings = json.loads(CAT_PREREG_PATH.read_text(encoding="utf-8"))["method"]["ranker"]
        probe_catboost_gpu(settings)
    else:
        print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
