"""Pre-register and compare one non-Korean fuzzy retrieval rule against EXP-014.

The newly assigned group split can overlap historical experiments. It is a
prospective check of this fixed rule, not a never-used competition holdout.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import time
import types
import zipfile
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process


ROOT = Path(__file__).resolve().parents[1]
TRAIN = ROOT / "data" / "open" / "train.csv"
CATALOG = ROOT / "data" / "open" / "vessel_catalog.csv"
BASELINE_ZIP = ROOT / "submissions" / "final_exp014.zip"
SPLIT = ROOT / "experiments" / "splits" / "exp022_group_seed31425.json"
RESULT = ROOT / "experiments" / "results" / "exp022_non_korean_fuzzy_seed31425"
SEED = 31425
LIMIT = 40
CUTOFF = 50
MAX_CANDIDATES = 60
GATES = {
    "minimum_weighted_gain": 0.02,
    "minimum_overall_candidate_recall_gain": 0.10,
    "maximum_mean_candidates": 50.0,
    "maximum_candidates_per_row": 60,
    "maximum_projected_3000_seconds": 600.0,
    "korean_accuracy_must_not_fall": True,
}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def reserve() -> None:
    if SPLIT.exists():
        raise FileExistsError(f"Already reserved: {SPLIT}")
    train = pd.read_csv(TRAIN, encoding="utf-8-sig")
    groups = sorted(train["target_vessel_id"].astype(str).unique())
    random.Random(SEED).shuffle(groups)
    valid_groups = set(groups[: round(len(groups) * 0.2)])
    valid = train[train["target_vessel_id"].astype(str).isin(valid_groups)]
    if valid.empty or len(valid) == len(train):
        raise AssertionError("Invalid group allocation")
    payload = {
        "method": "pre_registered_target_group_partition",
        "historical_validation_overlap": True,
        "interpretation": "New assignment for fixed EXP-022 rule; not fully untouched data",
        "seed": SEED,
        "valid_group_count": len(valid_groups),
        "valid_row_count": len(valid),
        "valid_sample_ids": sorted(valid["sample_id"].astype(str).tolist()),
        "baseline_archive_sha256": sha256(BASELINE_ZIP),
        "train_sha256": sha256(TRAIN),
        "catalog_sha256": sha256(CATALOG),
        "rule": {
            "applies_to": "non-Korean vessel_name only",
            "scorer": "rapidfuzz.fuzz.ratio",
            "limit": LIMIT,
            "cutoff": CUTOFF,
            "merge": "fuzzy first, then EXP-014 exact-name/tonnage candidates",
            "non_korean_fuzzy_bonus": 0.0,
            "max_candidates": MAX_CANDIDATES,
        },
        "acceptance_gates": GATES,
    }
    SPLIT.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"reserved": str(SPLIT), "rows": len(valid), "groups": len(valid_groups), "gates": GATES}, ensure_ascii=False, indent=2))


def load_baseline_module() -> types.ModuleType:
    with zipfile.ZipFile(BASELINE_ZIP) as archive:
        names = set(archive.namelist())
        if names != {"model/adaptive_lib.py", "script.py", "requirements.txt"}:
            raise AssertionError(f"Unexpected baseline archive members: {names}")
        source = archive.read("model/adaptive_lib.py").decode("utf-8")
    module = types.ModuleType("exp022_frozen_adaptive_lib")
    exec(compile(source, str(BASELINE_ZIP) + ":model/adaptive_lib.py", "exec"), module.__dict__)
    return module


def improved_candidates(row: dict, index, module) -> tuple[list[int], dict[int, float]]:
    if module.is_korean(row.get("vessel_name")):
        return index.candidates_with_scores(row)
    query = module.normalize_name(row.get("vessel_name"))
    fuzzy_indices: list[int] = []
    if query:
        fuzzy_indices = [
            int(candidate_index)
            for _, _, candidate_index in process.extract(
                query,
                index.romanized_names,
                scorer=fuzz.ratio,
                limit=LIMIT,
                score_cutoff=CUTOFF,
            )
        ]
    base_indices = index.generate_base_candidates(row, MAX_CANDIDATES)
    merged: list[int] = []
    seen: set[int] = set()
    for candidate_index in fuzzy_indices + base_indices:
        if candidate_index not in seen:
            merged.append(candidate_index)
            seen.add(candidate_index)
        if len(merged) >= MAX_CANDIDATES:
            break
    return merged, {}


def evaluate(rows: list[dict], index, module, *, improved: bool) -> dict:
    predictions: list[str] = []
    covered: list[bool] = []
    candidate_counts: list[int] = []
    korean_flags: list[bool] = []
    correct: list[bool] = []
    started = time.perf_counter()
    for row in rows:
        korean = module.is_korean(row.get("vessel_name"))
        candidates, fuzzy_scores = (
            improved_candidates(row, index, module)
            if improved
            else index.candidates_with_scores(row)
        )
        ranked: list[tuple[float, str]] = []
        for candidate_index in candidates:
            features = module.base_features(row, index.catalog.iloc[candidate_index])
            score = float(features["score"]) + module.FUZZY_SCORE_BONUS * float(
                fuzzy_scores.get(candidate_index, 0.0)
            )
            ranked.append((score, index.ids[candidate_index]))
        ranked.sort(key=lambda item: (-item[0], item[1]))
        prediction = ranked[0][1] if ranked else index.ids[0]
        target = str(row["target_vessel_id"])
        predictions.append(prediction)
        covered.append(target in {index.ids[i] for i in candidates})
        candidate_counts.append(len(candidates))
        korean_flags.append(korean)
        correct.append(prediction == target)
    elapsed = time.perf_counter() - started
    if not any(korean_flags) or all(korean_flags):
        raise AssertionError("The group split must contain Korean and non-Korean rows")
    korean_count = sum(korean_flags)
    non_korean_count = len(rows) - korean_count
    korean_accuracy = sum(ok for ok, kr in zip(correct, korean_flags) if kr) / korean_count
    non_korean_accuracy = sum(ok for ok, kr in zip(correct, korean_flags) if not kr) / non_korean_count
    korean_recall = sum(hit for hit, kr in zip(covered, korean_flags) if kr) / korean_count
    non_korean_recall = sum(hit for hit, kr in zip(covered, korean_flags) if not kr) / non_korean_count
    return {
        "metrics": {
            "rows": len(rows),
            "korean_rows": korean_count,
            "non_korean_rows": non_korean_count,
            "overall_accuracy": sum(correct) / len(rows),
            "korean_accuracy": korean_accuracy,
            "non_korean_accuracy": non_korean_accuracy,
            "weighted": 0.7 * korean_accuracy + 0.3 * non_korean_accuracy,
            "overall_candidate_recall": sum(covered) / len(rows),
            "korean_candidate_recall": korean_recall,
            "non_korean_candidate_recall": non_korean_recall,
            "mean_candidates": sum(candidate_counts) / len(rows),
            "max_candidates": max(candidate_counts),
            "elapsed_seconds_excluding_load": elapsed,
            "projected_3000_seconds_excluding_load": elapsed * 3000 / len(rows),
        },
        "predictions": predictions,
        "correct": correct,
    }


def run_evaluation() -> None:
    if not SPLIT.exists():
        raise FileNotFoundError("Reserve the split and gates before evaluation")
    if RESULT.exists():
        raise FileExistsError(f"Already evaluated: {RESULT}")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    if split["seed"] != SEED or split["rule"]["limit"] != LIMIT or split["acceptance_gates"] != GATES:
        raise AssertionError("Pre-registered settings changed")
    for key, path in (("baseline_archive_sha256", BASELINE_ZIP), ("train_sha256", TRAIN), ("catalog_sha256", CATALOG)):
        if split[key] != sha256(path):
            raise AssertionError(f"Input changed since pre-registration: {path}")
    train = pd.read_csv(TRAIN, encoding="utf-8-sig")
    valid_ids = set(split["valid_sample_ids"])
    valid = train[train["sample_id"].astype(str).isin(valid_ids)]
    if len(valid) != len(valid_ids):
        raise AssertionError("Reserved sample IDs missing from train")
    train_groups = set(train.loc[~train["sample_id"].astype(str).isin(valid_ids), "target_vessel_id"].astype(str))
    valid_groups = set(valid["target_vessel_id"].astype(str))
    if train_groups & valid_groups:
        raise AssertionError("Target group leakage within EXP-022 split")
    catalog = pd.read_csv(CATALOG, encoding="utf-8-sig")
    module = load_baseline_module()
    index_started = time.perf_counter()
    index = module.AdaptiveCatalogIndex(catalog)
    index_seconds = time.perf_counter() - index_started
    rows = valid.to_dict("records")
    baseline = evaluate(rows, index, module, improved=False)
    variant = evaluate(rows, index, module, improved=True)
    base = baseline["metrics"]
    new = variant["metrics"]
    gains = sum(not a and b for a, b in zip(baseline["correct"], variant["correct"]))
    losses = sum(a and not b for a, b in zip(baseline["correct"], variant["correct"]))
    checks = {
        "weighted_gain": new["weighted"] - base["weighted"] >= GATES["minimum_weighted_gain"],
        "candidate_recall_gain": new["overall_candidate_recall"] - base["overall_candidate_recall"] >= GATES["minimum_overall_candidate_recall_gain"],
        "korean_accuracy_non_decrease": new["korean_accuracy"] >= base["korean_accuracy"],
        "mean_candidates": new["mean_candidates"] <= GATES["maximum_mean_candidates"],
        "max_candidates": new["max_candidates"] <= GATES["maximum_candidates_per_row"],
        "projected_3000_runtime": new["projected_3000_seconds_excluding_load"] + index_seconds <= GATES["maximum_projected_3000_seconds"],
    }
    summary = {
        "experiment": "EXP-022 fixed non-Korean fuzzy retrieval",
        "split": str(SPLIT.relative_to(ROOT)),
        "historical_validation_overlap": True,
        "index_seconds": index_seconds,
        "baseline": base,
        "variant": new,
        "delta": {key: new[key] - base[key] for key in ("weighted", "korean_accuracy", "non_korean_accuracy", "overall_candidate_recall", "korean_candidate_recall", "non_korean_candidate_recall", "mean_candidates", "max_candidates")},
        "prediction_changes": sum(a != b for a, b in zip(baseline["predictions"], variant["predictions"])),
        "corrected_rows": gains,
        "new_error_rows": losses,
        "gate_checks": checks,
        "gate_passed": all(checks.values()),
        "package_created": False,
        "limitation": "The newly assigned target-group split overlaps historically explored official train rows; it is not a fully unused holdout.",
    }
    RESULT.mkdir(parents=True)
    (RESULT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("reserve", "evaluate"))
    args = parser.parse_args()
    reserve() if args.mode == "reserve" else run_evaluation()
