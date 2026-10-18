"""GPU-auto fixed CatBoost-heavy normalized-rank ensemble under nested grouped OOF."""

from __future__ import annotations

import json
import math
import subprocess
import sys
import time
import argparse
from pathlib import Path

import catboost
import lightgbm as lgb
import numpy as np
import pandas as pd
from catboost import CatBoostRanker, Pool


ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = ROOT / "experiments" / "splits" / "agent_rank_ensemble_catboost70_nested_oof_seed31435.json"
SPLIT_PATH = ROOT / "experiments" / "splits" / "agent_syllable_transliteration_cv_seed31426.json"
LGBM_PREREG_PATH = ROOT / "experiments" / "splits" / "agent_learned_reranker_nested_oof_seed31430.json"
CAT_PREREG_PATH = ROOT / "experiments" / "splits" / "agent_catboost_ranker_nested_oof_seed31432.json"
LGBM_RESULT_PATH = (
    ROOT / "experiments" / "results" / "agent_learned_reranker_nested_oof_20261018"
    / "summary_lgbm_4_6_0.json"
)
CAT_RESULT_PATH = (
    ROOT / "experiments" / "results" / "agent_catboost_ranker_nested_oof_20261018"
    / "summary_catboost_1_2_8.json"
)
RESULT_PATH = (
    ROOT / "experiments" / "results" / "agent_rank_ensemble_catboost70_nested_oof_20261018"
    / "summary_rank_ensemble_catboost70.json"
)
DATA_ROOT = ROOT / "data" / "open"

sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "submit" / "model"))

from adaptive_lib import (  # noqa: E402
    FUZZY_SCORE_BONUS,
    AdaptiveCatalogIndex,
    base_features,
    is_korean,
    normalize_name,
    romanized_name,
)
from scripts.agent_learned_reranker_nested_oof import (  # noqa: E402
    FEATURE_NAMES,
    collect_training_queries,
    make_query_examples,
)
from scripts.agent_syllable_transliteration_probe import (  # noqa: E402
    file_sha256,
    learn_syllable_mappings,
)


def create_lightgbm_ranker(settings: dict, device_type: str) -> lgb.LGBMRanker:
    params = dict(
        objective="lambdarank",
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
        label_gain=list(settings["label_gain"]),
        device_type=device_type,
    )
    if device_type == "cpu":
        params["deterministic"] = True
        params["force_col_wise"] = True
    return lgb.LGBMRanker(**params)


def create_catboost_ranker(settings: dict, task_type: str) -> CatBoostRanker:
    return CatBoostRanker(
        loss_function=settings["loss_function"],
        iterations=int(settings["iterations"]),
        learning_rate=float(settings["learning_rate"]),
        depth=int(settings["depth"]),
        l2_leaf_reg=float(settings["l2_leaf_reg"]),
        random_seed=int(settings["random_seed"]),
        thread_count=int(settings["thread_count"]),
        task_type=task_type,
        verbose=False,
        allow_writing_files=False,
    )


def probe_lightgbm_gpu(lgb_settings: dict) -> dict:
    probe_x = pd.DataFrame([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]])
    probe_y = [1, 0, 1, 0]
    probe_groups = [2, 2]
    gpu_lgb_settings = dict(lgb_settings)
    gpu_lgb_settings.update({"n_estimators": 2, "num_leaves": 3, "max_depth": 2, "min_child_samples": 1})
    gpu_lgb = create_lightgbm_ranker(gpu_lgb_settings, device_type="gpu")
    gpu_lgb.fit(probe_x, probe_y, group=probe_groups)
    return {"backend": "OpenCL"}


def probe_catboost_gpu(cat_settings: dict) -> dict:
    probe_x = pd.DataFrame([[0.0, 1.0], [1.0, 0.0], [0.2, 0.8], [0.8, 0.2]])
    probe_y = [1, 0, 1, 0]
    probe_pool = Pool(data=probe_x, label=probe_y, group_id=[0, 0, 1, 1])
    probe_cat = create_catboost_ranker(cat_settings, task_type="GPU")
    probe_cat.set_params(iterations=2, depth=min(3, int(cat_settings["depth"])))
    probe_cat.fit(probe_pool)
    return {"backend": "CUDA"}


def detect_device_backends() -> dict:
    """Probe backends in child processes so native library aborts cannot kill this runner."""
    selected = {}
    for name, backend in (("lightgbm", "OpenCL"), ("catboost", "CUDA")):
        try:
            completed = subprocess.run(
                [sys.executable, str(Path(__file__).resolve()), "--probe-backend", name],
                capture_output=True,
                text=True,
                timeout=90,
                check=False,
            )
            if completed.returncode == 0:
                payload = json.loads(completed.stdout.strip().splitlines()[-1])
                if payload.get("backend") != backend:
                    raise ValueError("GPU probe returned an unexpected backend")
                selected[name] = {
                    "device": "gpu",
                    "backend": backend,
                    "fallback_reason": None,
                }
            else:
                detail = (completed.stderr or completed.stdout).strip().splitlines()
                reason = " | ".join(detail[-4:])[-400:]
                selected[name] = {
                    "device": "cpu",
                    "backend": "CPU",
                    "fallback_reason": f"GPU {backend} probe failed (exit {completed.returncode}): {reason}",
                }
        except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
            selected[name] = {
                "device": "cpu",
                "backend": "CPU",
                "fallback_reason": f"GPU {backend} probe failed: {type(exc).__name__}: {str(exc)[:400]}",
            }
    return selected


def normalized_rank_utilities(scores, candidate_ids: list[str]) -> dict[str, float]:
    if len(scores) != len(candidate_ids):
        raise AssertionError("Model scores and candidate IDs differ in length")
    if not candidate_ids:
        return {}
    values = [float(score) for score in scores]
    if not all(math.isfinite(value) for value in values):
        raise AssertionError("A ranker returned a non-finite prediction")
    order = sorted(range(len(candidate_ids)), key=lambda i: (-values[i], candidate_ids[i]))
    if len(order) == 1:
        return {candidate_ids[order[0]]: 1.0}
    denom = len(order) - 1
    return {
        candidate_ids[index]: (len(order) - rank - 1) / denom
        for rank, index in enumerate(order)
    }


def summarize_prediction(rows: list[dict], prediction_key: str) -> dict:
    frame = pd.DataFrame(rows)
    baseline_correct = frame["baseline_prediction"] == frame["target"]
    model_correct = frame[prediction_key] == frame["target"]
    return {
        "korean_rows": int(len(frame)),
        "baseline_top1": float(baseline_correct.mean()),
        "top1": float(model_correct.mean()),
        "top1_gain_over_baseline": float(model_correct.mean() - baseline_correct.mean()),
        "baseline_candidate_recall": float(frame["baseline_has_target"].mean()),
        "union_candidate_recall": float(frame["union_has_target"].mean()),
        "mean_union_candidates": float(frame["candidate_count"].mean()),
        "max_union_candidates": int(frame["candidate_count"].max()),
        "baseline_wrong_to_correct": int((~baseline_correct & model_correct).sum()),
        "baseline_correct_to_wrong": int((baseline_correct & ~model_correct).sum()),
        "prediction_changed_from_baseline": int(
            (frame["baseline_prediction"] != frame[prediction_key]).sum()
        ),
    }


def component_pair_counts(rows: list[dict], component_key: str, ensemble_key: str) -> dict:
    frame = pd.DataFrame(rows)
    component_correct = frame[component_key] == frame["target"]
    ensemble_correct = frame[ensemble_key] == frame["target"]
    return {
        "ensemble_correct_component_wrong": int((ensemble_correct & ~component_correct).sum()),
        "component_correct_ensemble_wrong": int((component_correct & ~ensemble_correct).sum()),
        "both_correct": int((component_correct & ensemble_correct).sum()),
        "both_wrong": int((~component_correct & ~ensemble_correct).sum()),
    }


def evaluate() -> dict:
    if RESULT_PATH.exists():
        raise FileExistsError(f"Refusing to overwrite result: {RESULT_PATH}")
    prereg = json.loads(PREREG_PATH.read_text(encoding="utf-8"))
    lgb_prereg = json.loads(LGBM_PREREG_PATH.read_text(encoding="utf-8"))
    cat_prereg = json.loads(CAT_PREREG_PATH.read_text(encoding="utf-8"))
    split = json.loads(SPLIT_PATH.read_text(encoding="utf-8"))
    if lgb.__version__ != "4.6.0" or catboost.__version__ != "1.2.8":
        raise AssertionError(
            f"Expected LightGBM 4.6.0 and CatBoost 1.2.8; got {lgb.__version__}, {catboost.__version__}"
        )
    expected_hashes = prereg["data_hashes"]
    if expected_hashes != lgb_prereg["data_hashes"] or expected_hashes != cat_prereg["data_hashes"]:
        raise AssertionError("Ensemble and component preregistrations use different input hashes")
    for filename, expected_hash in expected_hashes.items():
        if file_sha256(DATA_ROOT / filename) != expected_hash:
            raise AssertionError(f"Frozen input hash mismatch: {filename}")

    train = pd.read_csv(DATA_ROOT / "train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(DATA_ROOT / "vessel_catalog.csv", encoding="utf-8-sig")
    group_to_fold = {
        str(group_id): int(fold["fold"])
        for fold in split["folds"]
        for group_id in fold["target_group_ids"]
    }
    if set(group_to_fold) != set(train["target_vessel_id"].astype(str)):
        raise AssertionError("Frozen folds do not cover target groups exactly")

    index = AdaptiveCatalogIndex(catalog.reset_index(drop=True))
    observations: list[dict] = []
    fold_reports: list[dict] = []
    mapping_fit_seconds = 0.0
    feature_generation_seconds = 0.0
    lgb_fit_seconds = 0.0
    cat_fit_seconds = 0.0
    lgb_settings = lgb_prereg["ranker"]
    cat_settings = cat_prereg["method"]["ranker"]
    ensemble_weights = prereg["method"]["ensemble"]["weights"]
    if abs(sum(float(value) for value in ensemble_weights.values()) - 1.0) > 1e-12:
        raise AssertionError("Fixed ensemble weights must sum to 1.0")
    device_selection = detect_device_backends()

    for outer_fold in split["folds"]:
        outer_id = int(outer_fold["fold"])
        group_ids = train["target_vessel_id"].astype(str).map(group_to_fold)
        outer_train = train[group_ids.ne(outer_id)].copy()
        outer_valid = train[group_ids.eq(outer_id)].copy()
        outer_train_groups = set(outer_train["target_vessel_id"].astype(str))
        outer_valid_groups = set(outer_valid["target_vessel_id"].astype(str))
        if outer_train_groups & outer_valid_groups:
            raise AssertionError(f"Outer target-group leakage in fold {outer_id}")

        train_features: list[list[float]] = []
        train_labels: list[int] = []
        train_group_sizes: list[int] = []
        inner_reports: list[dict] = []
        inner_skipped = 0
        for inner_fold in split["folds"]:
            inner_id = int(inner_fold["fold"])
            if inner_id == outer_id:
                continue
            inner_group_ids = outer_train["target_vessel_id"].astype(str).map(group_to_fold)
            inner_valid = outer_train[inner_group_ids.eq(inner_id)].copy()
            inner_train = outer_train[inner_group_ids.ne(inner_id)].copy()
            inner_train_groups = set(inner_train["target_vessel_id"].astype(str))
            inner_valid_groups = set(inner_valid["target_vessel_id"].astype(str))
            if inner_train_groups & inner_valid_groups or inner_valid_groups & outer_valid_groups:
                raise AssertionError(f"Nested target-group leakage in outer {outer_id}, inner {inner_id}")

            started = time.perf_counter()
            mappings, _, _ = learn_syllable_mappings(
                inner_train, catalog, normalize_name, romanized_name
            )
            mapping_fit_seconds += time.perf_counter() - started
            started = time.perf_counter()
            features, labels, groups, kept, skipped = collect_training_queries(
                inner_valid, index, mappings
            )
            feature_generation_seconds += time.perf_counter() - started
            train_features.extend(features)
            train_labels.extend(labels)
            train_group_sizes.extend(groups)
            inner_skipped += skipped
            inner_reports.append({
                "inner_fold": inner_id,
                "mapping_fit_target_groups": len(inner_train_groups),
                "heldout_training_target_groups": len(inner_valid_groups),
                "ranker_training_queries_kept": kept,
                "ranker_training_queries_without_target_candidate": skipped,
            })

        if not train_group_sizes or not any(train_labels):
            raise AssertionError(f"No ranking examples for outer fold {outer_id}")
        if sum(train_group_sizes) != len(train_labels) or len(train_features) != len(train_labels):
            raise AssertionError(f"Query group sizes do not match features in fold {outer_id}")

        train_frame = pd.DataFrame(train_features, columns=FEATURE_NAMES)
        lgb_ranker = create_lightgbm_ranker(
            lgb_settings, device_type=device_selection["lightgbm"]["device"]
        )
        started = time.perf_counter()
        lgb_ranker.fit(train_frame, train_labels, group=train_group_sizes)
        lgb_fold_fit_seconds = time.perf_counter() - started
        lgb_fit_seconds += lgb_fold_fit_seconds

        cat_group_ids = [
            query_id
            for query_id, size in enumerate(train_group_sizes)
            for _ in range(size)
        ]
        cat_pool = Pool(
            data=train_frame,
            label=train_labels,
            group_id=cat_group_ids,
            feature_names=list(FEATURE_NAMES),
        )
        cat_ranker = create_catboost_ranker(
            cat_settings, task_type=device_selection["catboost"]["device"].upper()
        )
        started = time.perf_counter()
        cat_ranker.fit(cat_pool)
        cat_fold_fit_seconds = time.perf_counter() - started
        cat_fit_seconds += cat_fold_fit_seconds

        started = time.perf_counter()
        outer_mappings, _, _ = learn_syllable_mappings(
            outer_train, catalog, normalize_name, romanized_name
        )
        mapping_fit_seconds += time.perf_counter() - started
        fold_observations: list[dict] = []
        for row in outer_valid.to_dict("records"):
            if not is_korean(row.get("vessel_name")):
                continue
            query = make_query_examples(index, row, outer_mappings)
            baseline_ranked = []
            for candidate_index in query["baseline_indices"]:
                features = base_features(row, index.catalog.iloc[candidate_index])
                score = features["score"] + FUZZY_SCORE_BONUS * query[
                    "baseline_fuzzy_scores"
                ].get(candidate_index, 0.0)
                baseline_ranked.append((score, index.ids[candidate_index]))
            baseline_ranked.sort(key=lambda item: (-item[0], item[1]))
            baseline_prediction = baseline_ranked[0][1] if baseline_ranked else index.ids[0]

            candidate_frame = pd.DataFrame(query["features"], columns=FEATURE_NAMES)
            candidate_ids = query["candidate_ids"]
            lgb_utilities = normalized_rank_utilities(
                lgb_ranker.predict(candidate_frame), candidate_ids
            )
            cat_utilities = normalized_rank_utilities(
                cat_ranker.predict(candidate_frame), candidate_ids
            )
            ensemble_scores = {
                candidate_id: float(ensemble_weights["lightgbm"])
                * lgb_utilities[candidate_id]
                + float(ensemble_weights["catboost"]) * cat_utilities[candidate_id]
                for candidate_id in candidate_ids
            }
            ensemble_prediction = min(
                candidate_ids,
                key=lambda candidate_id: (-ensemble_scores[candidate_id], candidate_id),
            ) if candidate_ids else index.ids[0]
            lgb_prediction = min(
                candidate_ids,
                key=lambda candidate_id: (-lgb_utilities[candidate_id], candidate_id),
            ) if candidate_ids else index.ids[0]
            cat_prediction = min(
                candidate_ids,
                key=lambda candidate_id: (-cat_utilities[candidate_id], candidate_id),
            ) if candidate_ids else index.ids[0]
            item = {
                "target": query["target"],
                "baseline_prediction": baseline_prediction,
                "lightgbm_prediction": lgb_prediction,
                "catboost_prediction": cat_prediction,
                "ensemble_prediction": ensemble_prediction,
                "baseline_has_target": query["target"] in {
                    index.ids[i] for i in query["baseline_indices"]
                },
                "union_has_target": query["target"] in candidate_ids,
                "candidate_count": query["candidate_count"],
            }
            observations.append(item)
            fold_observations.append(item)
        feature_generation_seconds += time.perf_counter() - started

        fold_metrics = {
            "lightgbm": summarize_prediction(fold_observations, "lightgbm_prediction"),
            "catboost": summarize_prediction(fold_observations, "catboost_prediction"),
            "ensemble": summarize_prediction(fold_observations, "ensemble_prediction"),
        }
        fold_reports.append({
            "outer_fold": outer_id,
            "train_target_groups": len(outer_train_groups),
            "valid_target_groups": len(outer_valid_groups),
            "inner_oof_training_queries_kept": sum(
                item["ranker_training_queries_kept"] for item in inner_reports
            ),
            "inner_oof_training_queries_without_target_candidate": inner_skipped,
            "ranker_training_candidate_rows": len(train_labels),
            "ranker_training_positive_rows": int(sum(train_labels)),
            "ranker_training_group_count": len(train_group_sizes),
            "lightgbm_fit_seconds": lgb_fold_fit_seconds,
            "catboost_fit_seconds": cat_fold_fit_seconds,
            "metrics": fold_metrics,
            "inner_folds": inner_reports,
        })

    metrics = {
        name: summarize_prediction(observations, f"{name}_prediction")
        for name in ("lightgbm", "catboost", "ensemble")
    }
    frozen_lgb = json.loads(LGBM_RESULT_PATH.read_text(encoding="utf-8"))
    frozen_cat = json.loads(CAT_RESULT_PATH.read_text(encoding="utf-8"))
    reference_reproduction = {}
    for name, frozen in (("lightgbm", frozen_lgb), ("catboost", frozen_cat)):
        if device_selection[name]["device"] != "cpu":
            reference_reproduction[name] = {
                "exact_cpu_reference_check": False,
                "reason": "GPU arithmetic may change predictions relative to the frozen CPU run.",
            }
            continue
        observed = metrics[name]["top1"]
        expected = frozen["metrics"]["learned_ranker_top1"]
        if abs(observed - expected) > 1e-12:
            raise AssertionError(
                f"Reproduced {name} top-1 {observed:.12f} differs from frozen {expected:.12f}"
            )
        for fold in fold_reports:
            fold_id = fold["outer_fold"]
            observed_fold = fold["metrics"][name]["top1"]
            expected_fold = frozen["folds"][fold_id]["metrics"]["learned_ranker_top1"]
            if abs(observed_fold - expected_fold) > 1e-12:
                raise AssertionError(f"Reproduced {name} fold {fold_id} differs from frozen reference")
        reference_reproduction[name] = {"exact_cpu_reference_check": True}

    best_component = max(("lightgbm", "catboost"), key=lambda name: metrics[name]["top1"])
    ensemble_top1 = metrics["ensemble"]["top1"]
    component_deltas = {
        name: {
            "top1_delta_ensemble_minus_component": ensemble_top1 - metrics[name]["top1"],
            "weighted_score_contribution_delta": 0.7 * (ensemble_top1 - metrics[name]["top1"]),
            "folds_ensemble_better": sum(
                fold["metrics"]["ensemble"]["top1"] > fold["metrics"][name]["top1"]
                for fold in fold_reports
            ),
            "complementarity": component_pair_counts(
                observations, f"{name}_prediction", "ensemble_prediction"
            ),
        }
        for name in ("lightgbm", "catboost")
    }
    folds_better_than_each_fold_best = sum(
        fold["metrics"]["ensemble"]["top1"]
        > max(fold["metrics"]["lightgbm"]["top1"], fold["metrics"]["catboost"]["top1"])
        for fold in fold_reports
    )
    best_delta = ensemble_top1 - metrics[best_component]["top1"]
    gates = prereg["acceptance_gates"]
    gate_results = {
        "minimum_top1_gain_over_better_component": best_delta
        >= gates["minimum_korean_top1_gain_over_better_component"],
        "minimum_weighted_contribution_gain_over_better_component": 0.7 * best_delta
        >= gates["minimum_weighted_score_contribution_gain_over_better_component"],
        "minimum_fold_count_improved_over_better_component": folds_better_than_each_fold_best
        >= gates["minimum_outer_folds_improved_over_better_component"],
        "candidate_recall_not_below_frozen_union": all(
            abs(metrics[name]["union_candidate_recall"] - 0.945424013434089) <= 1e-12
            for name in metrics
        ),
        "no_target_group_leakage": True,
    }
    report = {
        "experiment": prereg["experiment"],
        "preregistration": str(PREREG_PATH.relative_to(ROOT)),
        "source_split": prereg["source_group_split"],
        "historical_validation_overlap": True,
        "historical_error_analysis_overlap": True,
        "independent_performance_estimate": False,
        "data_hashes": expected_hashes,
        "python_version": sys.version.split()[0],
        "lightgbm_version": lgb.__version__,
        "catboost_version": catboost.__version__,
        "device_selection_mode": "auto; each ranker independently uses GPU when its GPU training probe succeeds",
        "selected_devices": device_selection,
        "frozen_cpu_reference_reproduction": reference_reproduction,
        "features": list(FEATURE_NAMES),
        "ensemble_settings": prereg["method"]["ensemble"],
        "device_selection_mode": "auto; each ranker independently uses GPU when its GPU training probe succeeds",
        "selected_devices": device_selection,
        "frozen_cpu_reference_reproduction": reference_reproduction,
        "metrics": metrics,
        "best_component_by_overall_oof": best_component,
        "component_deltas": component_deltas,
        "folds_better_than_each_fold_best": folds_better_than_each_fold_best,
        "folds": fold_reports,
        "timing": {
            "mapping_fit_seconds_total": mapping_fit_seconds,
            "feature_generation_seconds_total": feature_generation_seconds,
            "lightgbm_fitting_seconds_total": lgb_fit_seconds,
            "catboost_fitting_seconds_total": cat_fit_seconds,
        },
        "acceptance_gates": gates,
        "gate_results": gate_results,
        "exploratory_gates_passed": all(gate_results.values()),
        "submission_promotion_eligible": False,
        "submission_package_modified": False,
        "leaderboard_submitted": False,
        "per_row_inputs_predictions_or_candidate_lists_saved": False,
        "limitations": prereg["limitations"],
    }
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--probe-only",
        action="store_true",
        help="Probe GPU training support for each ranker and exit without running OOF folds.",
    )
    parser.add_argument(
        "--probe-backend",
        choices=("lightgbm", "catboost"),
        help=argparse.SUPPRESS,
    )
    args = parser.parse_args()
    if args.probe_backend:
        lgb_prereg = json.loads(LGBM_PREREG_PATH.read_text(encoding="utf-8"))
        cat_prereg = json.loads(CAT_PREREG_PATH.read_text(encoding="utf-8"))
        if args.probe_backend == "lightgbm":
            result = probe_lightgbm_gpu(lgb_prereg["ranker"])
        else:
            result = probe_catboost_gpu(cat_prereg["method"]["ranker"])
        print(json.dumps(result))
    elif args.probe_only:
        print(json.dumps({"selected_devices": detect_device_backends()}, ensure_ascii=False, indent=2))
    else:
        print(json.dumps(evaluate(), ensure_ascii=False, indent=2))
