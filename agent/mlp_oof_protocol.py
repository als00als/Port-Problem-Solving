"""Frozen target-group OOF assignment and paired MLP comparison statistics."""

from __future__ import annotations

from typing import Any
import unicodedata

import numpy as np


PROTOCOL_VERSION = "v11-mlp-oof-1"
LANGUAGES = ("korean", "non_korean")


def normalized_name_key(value: Any) -> str:
    if not isinstance(value, str):
        return ""
    text = unicodedata.normalize("NFKC", value).upper()
    return "".join(char for char in text if char.isalnum())


def build_group_oof_split(frame, seed: int = 20261019) -> dict[str, Any]:
    from sklearn.model_selection import StratifiedGroupKFold
    required = {"sample_id", "target_vessel_id", "is_korean"}
    if not required <= set(frame.columns):
        raise ValueError(f"분할 입력 컬럼 누락: {sorted(required - set(frame.columns))}")
    if frame["sample_id"].astype(str).duplicated().any():
        raise ValueError("sample_id가 중복됩니다")
    if frame["target_vessel_id"].isna().any():
        raise ValueError("target_vessel_id에 결측이 있습니다")
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
    y = frame["is_korean"].astype(int).to_numpy()
    groups = frame["target_vessel_id"].astype(str).to_numpy()
    rows = frame["sample_id"].astype(str).to_numpy()
    folds = []
    for fold_id, (train_idx, valid_idx) in enumerate(splitter.split(np.zeros(len(frame)), y, groups)):
        train_idx = np.sort(train_idx)
        valid_idx = np.sort(valid_idx)
        folds.append({
            "fold": fold_id,
            "train_sample_ids": rows[train_idx].tolist(),
            "valid_sample_ids": rows[valid_idx].tolist(),
            "train_target_group_ids": sorted(set(groups[train_idx])),
            "valid_target_group_ids": sorted(set(groups[valid_idx])),
        })
    result = {"experiment": "v11 fixed-feature MLP pairwise reranker", "protocol_version": PROTOCOL_VERSION,
              "seed": seed,
              "method": "five-fold StratifiedGroupKFold, target_vessel_id groups, stratified by Korean-name indicator",
              "folds": folds}
    validate_group_oof_split(frame, result)
    return result


def validate_group_oof_split(frame, split: dict[str, Any]) -> None:
    rows = frame["sample_id"].astype(str).tolist()
    groups = frame.set_index(frame["sample_id"].astype(str))["target_vessel_id"].astype(str).to_dict()
    expected_rows = set(rows)
    if len(split.get("folds", [])) != 5:
        raise ValueError("OOF 분할은 정확히 5개 fold여야 합니다")
    validation_rows: list[str] = []
    validation_groups: list[str] = []
    for expected_fold, fold in enumerate(split["folds"]):
        if fold.get("fold") != expected_fold:
            raise ValueError("fold 번호가 0부터 순서대로 있어야 합니다")
        fit_ids, valid_ids = fold["train_sample_ids"], fold["valid_sample_ids"]
        if len(fit_ids) != len(set(fit_ids)) or len(valid_ids) != len(set(valid_ids)):
            raise ValueError("fold 안에 중복 sample_id가 있습니다")
        if set(fit_ids) & set(valid_ids) or set(fit_ids) | set(valid_ids) != expected_rows:
            raise ValueError("각 fold는 공식 행 전체를 train/valid로 정확히 나눠야 합니다")
        fit_groups = {groups[row] for row in fit_ids}
        valid_groups = {groups[row] for row in valid_ids}
        if fit_groups & valid_groups:
            raise ValueError("target_vessel_id 그룹이 fold 안의 train/valid에 함께 있습니다")
        if sorted(fit_groups) != fold.get("train_target_group_ids"):
            raise ValueError("train target group 목록이 행 목록과 다릅니다")
        if sorted(valid_groups) != fold.get("valid_target_group_ids"):
            raise ValueError("valid target group 목록이 행 목록과 다릅니다")
        validation_rows.extend(valid_ids)
        validation_groups.extend(sorted(valid_groups))
    if len(validation_rows) != len(expected_rows) or set(validation_rows) != expected_rows:
        raise ValueError("모든 sample_id가 validation에서 정확히 한 번 나와야 합니다")
    if len(validation_groups) != len(set(validation_groups)) or set(validation_groups) != set(groups.values()):
        raise ValueError("모든 target group이 validation fold 하나에만 배정되어야 합니다")


def exact_mcnemar_pvalue(mlp_only_correct: int, baseline_only_correct: int) -> float:
    from scipy.stats import binomtest
    if (not isinstance(mlp_only_correct, int) or not isinstance(baseline_only_correct, int)
            or mlp_only_correct < 0 or baseline_only_correct < 0):
        raise ValueError("불일치 쌍 수는 0 이상의 정수여야 합니다")
    discordant = mlp_only_correct + baseline_only_correct
    if discordant == 0:
        return 1.0
    return float(binomtest(mlp_only_correct, discordant, p=0.5, alternative="two-sided").pvalue)


def holm_adjust(pvalues: list[float]) -> list[float]:
    if any(not 0.0 <= value <= 1.0 for value in pvalues):
        raise ValueError("p-value는 0과 1 사이여야 합니다")
    order = sorted(range(len(pvalues)), key=lambda index: (pvalues[index], index))
    adjusted = [0.0] * len(pvalues)
    running = 0.0
    count = len(pvalues)
    for rank, index in enumerate(order):
        running = max(running, min(1.0, (count - rank) * pvalues[index]))
        adjusted[index] = running
    return adjusted


def _counts(base: list[dict], alt: list[dict], mask: np.ndarray) -> dict[str, Any]:
    base_ok = np.asarray([bool(row["final_correct"]) for row in base], dtype=bool)[mask]
    alt_ok = np.asarray([bool(row["final_correct"]) for row in alt], dtype=bool)[mask]
    mlp_wins = int((~base_ok & alt_ok).sum())
    baseline_wins = int((base_ok & ~alt_ok).sum())
    n = int(mask.sum())
    return {"rows": n, "baseline_correct": int(base_ok.sum()), "mlp_correct": int(alt_ok.sum()),
            "mlp_only_correct": mlp_wins, "baseline_only_correct": baseline_wins,
            "net_correct_gain": mlp_wins - baseline_wins,
            "baseline_accuracy": float(base_ok.mean()) if n else 0.0,
            "mlp_accuracy": float(alt_ok.mean()) if n else 0.0,
            "exact_mcnemar_p": exact_mcnemar_pvalue(mlp_wins, baseline_wins)}


def summarize_paired_oof(baseline_cases: list[dict], mlp_cases: list[dict]) -> dict[str, Any]:
    if len(baseline_cases) != len(mlp_cases):
        raise ValueError("기준·MLP OOF 행 수가 다릅니다")
    for base, alt in zip(baseline_cases, mlp_cases):
        for key in ("sample_id", "truth_id", "is_korean", "seen_target", "query_name_seen_in_train"):
            if base.get(key) != alt.get(key):
                raise ValueError(f"대응 OOF 행이 다릅니다: {key}")
    korean = np.asarray([bool(row["is_korean"]) for row in baseline_cases])
    masks = {"all": np.ones(len(korean), dtype=bool), "korean": korean, "non_korean": ~korean}
    result = {name: _counts(baseline_cases, mlp_cases, mask) for name, mask in masks.items()}
    p_adjusted = holm_adjust([result[name]["exact_mcnemar_p"] for name in LANGUAGES])
    for name, adjusted in zip(LANGUAGES, p_adjusted):
        result[name]["holm_adjusted_p"] = adjusted
    strata = {}
    name_seen = np.asarray([bool(row["query_name_seen_in_train"]) for row in baseline_cases])
    for name, mask in (("train_name_seen", korean & name_seen),
                       ("train_name_unseen", korean & ~name_seen)):
        strata[name] = _counts(baseline_cases, mlp_cases, mask)
    k_base, k_mlp = result["korean"]["baseline_accuracy"], result["korean"]["mlp_accuracy"]
    n_base, n_mlp = result["non_korean"]["baseline_accuracy"], result["non_korean"]["mlp_accuracy"]
    weighted_base = 0.7 * k_base + 0.3 * n_base
    weighted_mlp = 0.7 * k_mlp + 0.3 * n_mlp
    language_net_positive = all(result[name]["net_correct_gain"] > 0 for name in LANGUAGES)
    language_tests_passed = all(result[name]["holm_adjusted_p"] < 0.05 for name in LANGUAGES)
    result["korean_name_strata"] = strata
    result["weighted_raw"] = {"baseline": weighted_base, "mlp": weighted_mlp,
                              "delta": weighted_mlp - weighted_base}
    result["decision"] = {"language_net_positive": language_net_positive,
                          "holm_exact_mcnemar_both_languages_p_lt_0_05": language_tests_passed,
                          "weighted_score_non_decrease": weighted_mlp >= weighted_base,
                          "mlp_improvement_signal": language_net_positive and language_tests_passed
                          and weighted_mlp >= weighted_base,
                          "protocol_version": PROTOCOL_VERSION}
    return result
