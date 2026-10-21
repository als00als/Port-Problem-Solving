"""Run the frozen v11 stress OOF protocol (v11-stress-oof-1) on the five fixed folds.

No retraining: each fold's frozen baseline LightGBM booster text is reloaded and only the
validation inputs are perturbed. Labels are joined after every prediction of a scenario is
final. Run with the competition environment:

    .\\.venv_competition_py311\\Scripts\\python.exe scripts\\run_v11_stress_oof.py

``--g0-only`` stops after the clean replay gate. The run folder is created only once every
pre-check passes, and an existing folder is never reused.
"""

from __future__ import annotations

# Import torch before numpy/pandas: loading it afterwards fails on this Windows
# environment with "WinError 1114 ... c10.dll" (see experiments/LOG.md).
try:  # pragma: no cover - environment-dependent import ordering
    import torch as _torch_preload  # noqa: F401
except Exception:
    _torch_preload = None

import argparse
from datetime import datetime
import hashlib
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import retrain_runtime as runtime  # noqa: E402
from agent import stress_perturbations as perturb  # noqa: E402
from agent.error_diagnostics import diagnose_rows  # noqa: E402
from agent.retrain_config import baseline_config  # noqa: E402
from scripts.v11_improvement_worker import (load_split, metrics,  # noqa: E402
                                            prediction_input_hashes, sha256_file, write_json)
from src.metric import calculate_score, is_korean  # noqa: E402

ml = runtime.ml

PROTOCOL_PATH = ROOT / "experiments" / "protocols" / "v11_stress_oof_v1.json"
PROTOCOL_SHA256 = "4fc3f8ba3fead033185470caf3b86452b32eb13e4db3df2515df22b3f11fc86a"
PROTOCOL_VERSION = "v11-stress-oof-1"
CONTROL = "S0_clean_replay"
CODE_FILES = (
    "agent/stress_perturbations.py", "agent/retrain_runtime.py", "agent/retrain_config.py",
    "agent/error_diagnostics.py", "agent/team_v11/source/match_lib.py",
    "agent/team_v11/source/train.py", "scripts/v11_improvement_worker.py",
    "scripts/run_v11_stress_oof.py",
)
INDEPENDENCE_ROWS = 5


def zip_snapshot() -> dict[str, dict[str, object]]:
    return {path.relative_to(ROOT).as_posix(): {"sha256": sha256_file(path), "size": path.stat().st_size}
            for path in sorted((ROOT / "submissions").glob("*.zip"))}


def source_hashes() -> dict[str, str]:
    return {path: sha256_file(ROOT / path) for path in CODE_FILES}


def load_protocol() -> dict:
    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(f"프로토콜 파일이 없습니다: {PROTOCOL_PATH}")
    actual = sha256_file(PROTOCOL_PATH)
    if actual != PROTOCOL_SHA256:
        raise ValueError(f"프로토콜 SHA-256이 다릅니다: 기대 {PROTOCOL_SHA256}, 실제 {actual}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    if protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("프로토콜 버전이 다릅니다")
    ids = [s["id"] for s in protocol["scenarios"]]
    if tuple(ids) != perturb.SCENARIOS:
        raise ValueError(f"구현한 시나리오 목록이 프로토콜과 다릅니다: {ids}")
    return protocol


def verify_frozen_inputs(protocol: dict) -> dict:
    frozen = protocol["frozen_inputs"]
    train_path = ROOT / "data" / "open" / "train.csv"
    catalog_path = ROOT / "data" / "open" / "vessel_catalog.csv"
    split_path = ROOT / frozen["split_file"]
    source = ROOT / frozen["source_run"]
    checks = {
        "train_csv": sha256_file(train_path) == frozen["train_csv_sha256"],
        "catalog_csv": sha256_file(catalog_path) == frozen["catalog_csv_sha256"],
        "split_file": sha256_file(split_path) == frozen["split_sha256"],
        "source_run_summary": sha256_file(source / "summary.json") == frozen["source_run_summary_sha256"],
    }
    folds = {}
    for fold in frozen["folds"]:
        model_path = source / fold["model_file"]
        cases_path = source / fold["clean_cases_file"]
        folds[fold["fold"]] = {
            "model_file": sha256_file(model_path) == fold["model_file_sha256"],
            "clean_cases_file": sha256_file(cases_path) == fold["clean_cases_sha256"],
        }
    failed = [k for k, ok in checks.items() if not ok]
    failed += [f"fold{f}_{k}" for f, inner in folds.items() for k, ok in inner.items() if not ok]
    if failed:
        raise ValueError("고정 입력 해시가 다릅니다: " + ", ".join(failed))
    config = baseline_config()
    if config["override_policy"] != "callsign_and_name":
        raise ValueError("override 정책이 callsign_and_name이 아닙니다")
    return {"input_hashes_verified": checks, "fold_hashes_verified": folds,
            "baseline_config": config}


def independence_check(model, inputs: pd.DataFrame, index, res, config, details) -> dict:
    """G2: at least five rows, Korean and non-Korean, identical alone and in a reversed subset."""
    korean = np.array([is_korean(value) for value in inputs["vessel_name"]])
    picked: list[int] = []
    for pool in (np.flatnonzero(korean), np.flatnonzero(~korean)):
        picked.extend(int(i) for i in pool[:2])
    for candidate in (0, len(inputs) // 2, len(inputs) - 1, 1, len(inputs) - 2):
        if len(set(picked)) >= INDEPENDENCE_ROWS:
            break
        if 0 <= candidate < len(inputs):
            picked.append(int(candidate))
    picked = sorted(set(picked))
    full = [details[i]["final_pred_id"] for i in picked]
    reverse = runtime.predict_details(model, inputs.iloc[picked[::-1]], index, res, config)
    singles = [runtime.predict_details(model, inputs.iloc[[i]], index, res, config)[0]["final_pred_id"]
               for i in picked]
    passed = ([d["final_pred_id"] for d in reverse][::-1] == full and singles == full)
    languages = sorted({bool(korean[i]) for i in picked})
    if not passed:
        raise ValueError("부분집합·단독 행 독립성 검사에 실패했습니다")
    return {"checked_rows": len(picked), "passed": True, "both_languages": languages == [False, True],
            "modes": ["reversed_subset", "single_row"]}


def fold_context(protocol: dict, fold_id: int):
    """Everything needed to predict a fold, with no retraining."""
    frozen = next(f for f in protocol["frozen_inputs"]["folds"] if f["fold"] == fold_id)
    split_name = frozen["split_name"]
    config = baseline_config()
    fit, valid, catalog, audit = load_split(split_name)
    index = ml.CatalogIndex(catalog)
    source = ROOT / protocol["frozen_inputs"]["source_run"]
    model_text = (source / frozen["model_file"]).read_text(encoding="utf-8")
    model = runtime.ReusedBooster(model_text)
    res = runtime.resources(fit, catalog, config)
    lookup_ids = set(res["callsign"].values())
    lookup_ids.update(v for votes in res["name_lookup"].values() for v in votes)
    if not lookup_ids <= set(fit["target_vessel_id"]):
        raise ValueError("조회표에 학습 외 target이 포함됐습니다")
    base_inputs = valid.drop(columns=["target_vessel_id"])
    clean_cases = json.loads((source / frozen["clean_cases_file"]).read_text(encoding="utf-8"))["cases"]
    return {"fold": fold_id, "split_name": split_name, "frozen": frozen, "config": config,
            "fit": fit, "valid": valid, "catalog": catalog, "index": index, "model": model,
            "res": res, "base_inputs": base_inputs, "clean_cases": clean_cases, "audit": audit,
            "train_callsigns": sorted(res["callsign"].keys()),
            "reloaded_model_sha256": runtime.model_sha256(model)}


def run_scenario(context: dict, scenario_id: str) -> dict:
    """Perturb, predict, check independence, then join labels and evaluate."""
    config, index, res, model = context["config"], context["index"], context["res"], context["model"]
    frame, audit, results = perturb.perturb_frame(
        scenario_id, context["base_inputs"],
        context["train_callsigns"] if scenario_id == "C2_callsign_collision_10pct" else None)
    started = time.monotonic()
    details = runtime.predict_details(model, frame, index, res, config)
    predict_seconds = time.monotonic() - started
    if not {d["final_pred_id"] for d in details} <= set(index.ids):
        raise AssertionError("예측 ID가 카탈로그 밖에 있습니다")
    independence = independence_check(model, frame, index, res, config, details)
    hashes = prediction_input_hashes(details)
    # Evaluation step: labels are joined only now, after every prediction above is final.
    valid = context["valid"]
    truth = valid["target_vessel_id"].astype(str).tolist()
    seen = valid["target_vessel_id"].isin(set(context["fit"]["target_vessel_id"])).to_numpy(bool)
    korean = np.array([is_korean(value) for value in valid["vessel_name"]])
    cases = diagnose_rows(details, truth, seen, korean, runtime.FEATURES, index.ids,
                          index.compact, index.id_to_index)
    perturbed_korean = np.array([is_korean(value) for value in frame["vessel_name"]])
    overrides = {source: int(sum(d["override_source"] == source for d in details))
                 for source in ("none", "name_lookup", "callsign")}
    collision = None
    if scenario_id == "C2_callsign_collision_10pct":
        table = set(context["train_callsigns"])
        changed = [i for i, r in enumerate(results) if "call_sign" in r["changes"]]
        collision = {
            "rows_with_injected_call_sign": len(changed),
            "injected_values_all_from_training_table":
                all(str(frame.iloc[i]["call_sign"]) in table for i in changed),
            "injected_equals_own_truth_rows":
                int(sum(context["res"]["callsign"].get(frame.iloc[i]["call_sign"]) == truth[i]
                        for i in changed)),
            "override_fired_rows": int(sum(details[i]["override_source"] == "callsign" for i in changed)),
        }
    return {"scenario_id": scenario_id, "fold": context["fold"], "split_name": context["split_name"],
            "perturbation_audit": audit, "prediction_input_hashes": hashes,
            "metrics": metrics(cases, valid), "override_counts": overrides,
            "override_changed_and_wrong": int(sum(c["override_changed"] and not c["final_correct"]
                                                  for c in cases)),
            "korean_indicator_changed_rows": int((perturbed_korean != korean).sum()),
            "row_independence": independence, "collision_audit": collision,
            "predict_seconds": round(predict_seconds, 2),
            "cases": cases, "perturbation_results": results}


def g0_fold_check(context: dict, scenario: dict) -> dict:
    frozen = context["frozen"]
    stored = {c["sample_id"]: c["final_pred_id"] for c in context["clean_cases"]}
    replay = {c["sample_id"]: c["final_pred_id"] for c in scenario["cases"]}
    matched = sum(1 for key, value in replay.items() if stored.get(key) == value)
    mismatched = sorted(key for key, value in replay.items() if stored.get(key) != value)
    return {
        "fold": context["fold"],
        "candidates_sha256_matches": scenario["prediction_input_hashes"]["candidates_sha256"]
            == frozen["clean_candidates_sha256"],
        "features_sha256_matches": scenario["prediction_input_hashes"]["features_sha256"]
            == frozen["clean_features_sha256"],
        "rows": len(replay), "final_pred_matched": matched,
        "final_pred_mismatched": len(mismatched),
        "all_rows_match": len(mismatched) == 0 and len(replay) == len(stored),
        "clean_correct_recorded": frozen["clean_correct"],
        "clean_correct_replayed": f"{scenario['metrics']['all']['correct']}/{scenario['metrics']['all']['rows']}",
        "reloaded_model_sha256": context["reloaded_model_sha256"],
        "model_file_sha256": frozen["model_file_sha256"],
    }


def aggregate(per_fold: list[dict], control: list[dict] | None) -> dict:
    """Pool the five folds into one 12,222-row OOF readout for one scenario."""
    cases = [case for fold in per_fold for case in fold["cases"]]
    truth = [c["truth_id"] for c in cases]
    final = [c["final_pred_id"] for c in cases]
    korean_flags = [c["is_korean"] for c in cases]
    names = ["가" if flag else "A" for flag in korean_flags]
    official = calculate_score(truth, final, names)
    weighted_raw = 0.7 * official["korean"] + 0.3 * official["non_korean"]

    def group(mask):
        rows = [c for c, keep in zip(cases, mask) if keep]
        correct = sum(c["final_correct"] for c in rows)
        hits = sum(c["truth_in_candidates"] for c in rows)
        return {"rows": len(rows), "correct": correct,
                "accuracy": (correct / len(rows)) if rows else None,
                "candidate_recall": (hits / len(rows)) if rows else None,
                "mean_candidates": (sum(c["candidate_count"] for c in rows) / len(rows)) if rows else None}

    out = {
        "rows": len(cases),
        "all": group([True] * len(cases)), "korean": group(korean_flags),
        "non_korean": group([not f for f in korean_flags]),
        "official_weighted_rounded": official["weighted"], "weighted_raw": weighted_raw,
        "official_korean_accuracy": official["korean"],
        "official_non_korean_accuracy": official["non_korean"],
        "rule_eligible_rows": sum(f["perturbation_audit"]["rule_eligible_rows"] for f in per_fold),
        "changed_rows": sum(f["perturbation_audit"]["changed_rows"] for f in per_fold),
        "not_applicable": all(f["perturbation_audit"]["not_applicable"] for f in per_fold),
        "column_change_counts": {},
        "override_counts": {k: sum(f["override_counts"][k] for f in per_fold)
                            for k in ("none", "name_lookup", "callsign")},
        "override_changed_and_wrong": sum(f["override_changed_and_wrong"] for f in per_fold),
        "korean_indicator_changed_rows": sum(f["korean_indicator_changed_rows"] for f in per_fold),
        "predict_seconds": round(sum(f["predict_seconds"] for f in per_fold), 2),
    }
    for fold in per_fold:
        for column, count in fold["perturbation_audit"]["column_change_counts"].items():
            out["column_change_counts"][column] = out["column_change_counts"].get(column, 0) + count
    out["column_change_counts"] = dict(sorted(out["column_change_counts"].items()))
    if control is not None:
        base = {c["sample_id"]: c for fold in control for c in fold["cases"]}
        newly_wrong, newly_correct, changed = [], [], 0
        for case in cases:
            before = base[case["sample_id"]]
            changed += before["final_pred_id"] != case["final_pred_id"]
            if before["final_correct"] and not case["final_correct"]:
                newly_wrong.append(case)
            elif not before["final_correct"] and case["final_correct"]:
                newly_correct.append(case)

        def by_language(rows):
            return {"total": len(rows),
                    "korean": sum(r["is_korean"] for r in rows),
                    "non_korean": sum(not r["is_korean"] for r in rows)}

        def rank_bucket(case):
            rank = case["truth_rank"]
            if not case["truth_in_candidates"] or rank is None:
                return "missing"
            return "2_5" if rank <= 5 else "6_plus"

        error_types: dict[str, int] = {}
        buckets = {"missing": 0, "2_5": 0, "6_plus": 0}
        for case in newly_wrong:
            key = case["primary_error_type"] or "unclassified"
            error_types[key] = error_types.get(key, 0) + 1
            buckets[rank_bucket(case)] += 1
        out["paired_vs_control"] = {
            "changed_predictions": changed,
            "newly_wrong": by_language(newly_wrong), "newly_correct": by_language(newly_correct),
            "newly_wrong_primary_error_types": dict(sorted(error_types.items(),
                                                           key=lambda kv: (-kv[1], kv[0]))),
            "newly_wrong_truth_rank_buckets": buckets,
        }
    return out


def deltas(scenario: dict, control: dict) -> dict:
    out = {}
    for key in ("weighted_raw", "official_korean_accuracy", "official_non_korean_accuracy"):
        out[key] = scenario[key] - control[key]
    for group in ("all", "korean", "non_korean"):
        for key in ("accuracy", "candidate_recall"):
            before, after = control[group][key], scenario[group][key]
            out[f"{group}_{key}"] = None if before is None or after is None else after - before
        out[f"{group}_correct"] = scenario[group]["correct"] - control[group]["correct"]
    return out


def strip_rows(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k not in ("cases", "perturbation_results")}


def local_rows(payload: dict) -> dict:
    return {"scenario_id": payload["scenario_id"], "fold": payload["fold"],
            "rows": [{"sample_id": case["sample_id"],
                      "changes": {k: (None if isinstance(v, float) and np.isnan(v) else v)
                                  for k, v in result["changes"].items()},
                      "operations": result["operations"],
                      "final_pred_id": case["final_pred_id"],
                      "final_correct": case["final_correct"],
                      "primary_error_type": case["primary_error_type"]}
                     for case, result in zip(payload["cases"], payload["perturbation_results"])]}


def summary_markdown(summary: dict) -> str:
    lines = ["# v11 스트레스 OOF 민감도 (탐색용)", "",
             f"- 실행 ID: `{summary['run_id']}`",
             f"- 프로토콜: `{summary['protocol_version']}` (SHA-256 `{summary['protocol_sha256']}`)",
             "- 재사용 분할의 탐색적 민감도 순위이며 리더보드 추정치나 모델 선택 근거가 아니다.", "",
             "| 시나리오 | 전체 | 한글 | 비한글 | 가중 원점수 | Δ vs S0 | 새 오답 | 새 정답 | 후보 포함률 |",
             "|---|---|---|---|---:|---:|---:|---:|---:|"]
    for row in summary["scenarios"]:
        agg, delta = row["aggregate"], row.get("delta")
        paired = agg.get("paired_vs_control", {})
        lines.append(
            f"| `{row['scenario_id']}` "
            f"| {agg['all']['correct']}/{agg['all']['rows']} "
            f"| {agg['korean']['correct']}/{agg['korean']['rows']} "
            f"| {agg['non_korean']['correct']}/{agg['non_korean']['rows']} "
            f"| {agg['weighted_raw']:.6f} "
            f"| {'-' if delta is None else format(delta['weighted_raw'], '+.6f')} "
            f"| {paired.get('newly_wrong', {}).get('total', '-')} "
            f"| {paired.get('newly_correct', {}).get('total', '-')} "
            f"| {agg['all']['candidate_recall']:.6f} |")
    lines += ["", "## 게이트", "",
              f"- G0 깨끗한 재현: {summary['gates']['G0_clean_replay']['passed']}",
              f"- G1 섭동 감사: 미적용 시나리오 {summary['gates']['G1_perturbation_audit']['not_applicable']}",
              f"- G2 행 독립성: {summary['gates']['G2_row_independence']['passed']}",
              f"- G3 C1 예측 불변 기대: {summary['gates']['G3_sanity_expectation']['passed']}", ""]
    return "\n".join(lines) + "\n"


def run(*, g0_only: bool = False) -> Path:
    protocol = load_protocol()
    verification = verify_frozen_inputs(protocol)
    before = source_hashes()
    zips_before = zip_snapshot()
    scenarios = [s["id"] for s in protocol["scenarios"]]
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "experiments" / "results" / f"agent_v11_stress_oof_{stamp}_{uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    local = run_dir / "local_only"
    local.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "pre_run_intent.json", {
        "run_id": run_dir.name, "protocol_version": PROTOCOL_VERSION,
        "protocol_file": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
        "protocol_sha256": PROTOCOL_SHA256,
        "frozen_inputs_verified": verification["input_hashes_verified"],
        "fold_hashes_verified": {str(k): v for k, v in verification["fold_hashes_verified"].items()},
        "baseline_config": verification["baseline_config"],
        "source_sha256": before, "zip_sha256_before": zips_before,
        "scenarios": scenarios, "g0_only": g0_only,
        "perturbation_salts": perturb.SALTS, "protocol_readings": perturb.READINGS,
        "retraining": False, "leaderboard_submission": False,
        "pre_registered_before_any_stress_prediction": True,
    })
    print(f"실행 폴더: {run_dir}", flush=True)

    control_folds: list[dict] = []
    g0 = {"per_fold": [], "passed": False}
    for fold_id in range(5):
        print(f"G0 fold{fold_id}: 모델 재사용·조회표 재구성·S0 재현", flush=True)
        context = fold_context(protocol, fold_id)
        payload = run_scenario(context, CONTROL)
        check = g0_fold_check(context, payload)
        g0["per_fold"].append(check)
        write_json(run_dir / f"stress_{CONTROL}_fold{fold_id}.json", strip_rows(payload) | {"g0": check})
        write_json(local / f"rows_{CONTROL}_fold{fold_id}.json", local_rows(payload))
        control_folds.append(payload)
        print(f"  해시 일치 {check['candidates_sha256_matches']}/{check['features_sha256_matches']}, "
              f"예측 일치 {check['final_pred_matched']}/{check['rows']}", flush=True)
    g0["rows"] = sum(c["rows"] for c in g0["per_fold"])
    g0["final_pred_matched"] = sum(c["final_pred_matched"] for c in g0["per_fold"])
    g0["passed"] = all(c["all_rows_match"] and c["candidates_sha256_matches"]
                       and c["features_sha256_matches"] for c in g0["per_fold"])
    write_json(run_dir / "g0_clean_replay.json", g0)
    if not g0["passed"]:
        write_json(run_dir / "completion_verification.json", {
            "run_id": run_dir.name, "completed": False, "reason": "G0_clean_replay_failed",
            "gates": {"G0_clean_replay": g0}, "stress_scenarios_run": [],
            "source_sha256_unchanged": source_hashes() == before,
            "zip_sha256_unchanged": zip_snapshot() == zips_before,
            "leaderboard_submission": False})
        raise SystemExit(f"G0 실패: 스트레스 시나리오를 실행하지 않았습니다. {run_dir}")

    control_aggregate = aggregate(control_folds, None)
    rows = [{"scenario_id": CONTROL, "aggregate": control_aggregate, "delta": None,
             "per_fold": [strip_rows(f) for f in control_folds]}]
    if not g0_only:
        for scenario_id in scenarios:
            if scenario_id == CONTROL:
                continue
            folds = []
            for fold_id in range(5):
                print(f"{scenario_id} fold{fold_id}", flush=True)
                context = fold_context(protocol, fold_id)
                payload = run_scenario(context, scenario_id)
                write_json(run_dir / f"stress_{scenario_id}_fold{fold_id}.json", strip_rows(payload))
                write_json(local / f"rows_{scenario_id}_fold{fold_id}.json", local_rows(payload))
                folds.append(payload)
            agg = aggregate(folds, control_folds)
            rows.append({"scenario_id": scenario_id, "aggregate": agg,
                         "delta": deltas(agg, control_aggregate),
                         "per_fold": [strip_rows(f) for f in folds]})
            print(f"  가중 원점수 {agg['weighted_raw']:.6f} "
                  f"(Δ {agg['weighted_raw'] - control_aggregate['weighted_raw']:+.6f})", flush=True)

    ranked = sorted((r for r in rows if r["delta"] is not None),
                    key=lambda r: r["delta"]["weighted_raw"])
    gates = {
        "G0_clean_replay": g0,
        "G1_perturbation_audit": {
            "per_scenario": {r["scenario_id"]: {
                "rule_eligible_rows": r["aggregate"]["rule_eligible_rows"],
                "changed_rows": r["aggregate"]["changed_rows"],
                "column_change_counts": r["aggregate"]["column_change_counts"],
                "not_applicable": r["aggregate"]["not_applicable"]} for r in rows},
            "not_applicable": [r["scenario_id"] for r in rows
                               if r["aggregate"]["not_applicable"]],
        },
        "G2_row_independence": {
            "passed": all(f["row_independence"]["passed"] and f["row_independence"]["both_languages"]
                          for r in rows for f in r["per_fold"]),
            "min_checked_rows": min((f["row_independence"]["checked_rows"]
                                     for r in rows for f in r["per_fold"]), default=0),
        },
        "G3_sanity_expectation": {
            "scenario": "C1_callsign_removed",
            "changed_predictions": next((r["aggregate"]["paired_vs_control"]["changed_predictions"]
                                         for r in rows if r["scenario_id"] == "C1_callsign_removed"), None),
            "expected": 0,
            "passed": next((r["aggregate"]["paired_vs_control"]["changed_predictions"] == 0
                            for r in rows if r["scenario_id"] == "C1_callsign_removed"), None),
            "note": "C1이 예측을 바꾸면 다른 call_sign 결과를 해석하기 전에 원인을 확인한다",
        },
    }
    summary = {
        "run_id": run_dir.name, "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256, "retraining": False,
        "official_rows": control_aggregate["rows"],
        "control": CONTROL, "g0_only": g0_only,
        "scenarios": rows, "gates": gates,
        "sensitivity_ranking": [{"scenario_id": r["scenario_id"],
                                 "delta_weighted_raw": r["delta"]["weighted_raw"],
                                 "delta_korean_accuracy": r["delta"]["official_korean_accuracy"],
                                 "delta_non_korean_accuracy": r["delta"]["official_non_korean_accuracy"],
                                 "newly_wrong": r["aggregate"]["paired_vs_control"]["newly_wrong"]["total"]}
                                for r in ranked],
        "interpretation": protocol["interpretation"]["primary_readout"],
        "limits": protocol["interpretation"]["limits"],
        "independence_note": "All official target groups were exposed in earlier validation and error "
                             "analysis; this is an exploratory sensitivity ranking, not a leaderboard "
                             "estimate or a model-selection basis.",
        "leaderboard_submission": False,
    }
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    after = source_hashes()
    write_json(run_dir / "completion_verification.json", {
        "run_id": run_dir.name, "completed": True,
        "scenarios_run": [r["scenario_id"] for r in rows],
        "gates": {k: (v if k != "G0_clean_replay" else {"passed": v["passed"], "rows": v["rows"],
                                                        "final_pred_matched": v["final_pred_matched"]})
                  for k, v in gates.items()},
        "source_sha256_unchanged": after == before,
        "source_sha256_changed_files": sorted(k for k in before if before[k] != after.get(k)),
        "zip_sha256_unchanged": zip_snapshot() == zips_before,
        "retraining": False, "leaderboard_submission": False,
        "row_level_files": "experiments/results/<run>/local_only (never sent to an LLM, never committed)",
    })
    print(f"완료: {run_dir / 'summary.json'}", flush=True)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--g0-only", action="store_true", help="G0 깨끗한 재현만 수행하고 멈춥니다")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(run(g0_only=args.g0_only))
