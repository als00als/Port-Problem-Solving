"""Run the frozen override gate protocol (v11-override-gate-1).

No retraining. The five group-OOF fold models and the row_seed42 model are reloaded from
their frozen booster text after a hash check, ``predict_details`` runs once per scenario,
and the four fixed policies are applied as post-processing on those same candidate scores.

    .\\.venv_competition_py311\\Scripts\\python.exe scripts\\run_v11_override_gate.py

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
import json
from pathlib import Path
import sys
import time
from uuid import uuid4

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent import override_gate as gate  # noqa: E402
from agent import retrain_runtime as runtime  # noqa: E402
from agent import stress_c3 as c3  # noqa: E402
from agent import stress_perturbations as perturb  # noqa: E402
from agent.retrain_config import baseline_config  # noqa: E402
from scripts.v11_improvement_worker import (load_split, prediction_input_hashes,  # noqa: E402
                                            sha256_file, write_json)
from src.metric import calculate_score, is_korean  # noqa: E402

ml = runtime.ml

PROTOCOL_PATH = ROOT / "experiments" / "protocols" / "v11_override_gate_v1.json"
PROTOCOL_SHA256 = "73d4c464f63d050e8f68dbdadab64e9ac013b3ff2a24f6ef8b968755764735b1"
PROTOCOL_VERSION = "v11-override-gate-1"
STRESS_PATH = ROOT / "experiments" / "protocols" / "v11_stress_oof_v1.json"
STRESS_SHA256 = "4fc3f8ba3fead033185470caf3b86452b32eb13e4db3df2515df22b3f11fc86a"

GROUP_SCENARIOS = ("E1_group_clean", "E2_group_C2", "E2_group_C3")
ROW_SCENARIOS = ("E3_row_clean", "E3_row_N2", "E3_row_N5", "E3_row_T1")
ROW_PERTURBATION = {"E3_row_N2": "N2_char_edit_one", "E3_row_N5": "N5_token_order",
                    "E3_row_T1": "T1_tonnage_both_missing"}
HARM_SCENARIOS = ("E2_group_C2", "E2_group_C3")
BENEFIT_SCENARIOS = ("E3_row_N2", "E3_row_N5", "E3_row_T1")
CODE_FILES = (
    "agent/override_gate.py", "agent/stress_c3.py", "agent/stress_perturbations.py",
    "agent/retrain_runtime.py", "agent/retrain_config.py",
    "agent/team_v11/source/match_lib.py", "agent/team_v11/source/train.py",
    "scripts/v11_improvement_worker.py", "scripts/run_v11_override_gate.py",
)
INDEPENDENCE_ROWS = 5


def zip_snapshot() -> dict[str, dict[str, object]]:
    return {path.relative_to(ROOT).as_posix(): {"sha256": sha256_file(path), "size": path.stat().st_size}
            for path in sorted((ROOT / "submissions").glob("*.zip"))}


def source_hashes() -> dict[str, str]:
    return {path: sha256_file(ROOT / path) for path in CODE_FILES}


def load_protocols() -> tuple[dict, dict]:
    for path, want in ((PROTOCOL_PATH, PROTOCOL_SHA256), (STRESS_PATH, STRESS_SHA256)):
        if not path.is_file():
            raise FileNotFoundError(f"프로토콜 파일이 없습니다: {path}")
        actual = sha256_file(path)
        if actual != want:
            raise ValueError(f"{path.name} SHA-256이 다릅니다: 기대 {want}, 실제 {actual}")
    protocol = json.loads(PROTOCOL_PATH.read_text(encoding="utf-8"))
    stress = json.loads(STRESS_PATH.read_text(encoding="utf-8"))
    if protocol["protocol_version"] != PROTOCOL_VERSION:
        raise ValueError("프로토콜 버전이 다릅니다")
    if protocol["frozen_inputs"]["stress_protocol"]["sha256"] != STRESS_SHA256:
        raise ValueError("참조된 스트레스 프로토콜 해시가 다릅니다")
    declared = set(protocol["scenarios"])
    if declared != set(GROUP_SCENARIOS) | set(ROW_SCENARIOS):
        raise ValueError(f"시나리오 목록 불일치: {sorted(declared)}")
    return protocol, stress


def verify_frozen_inputs(protocol: dict, stress: dict) -> dict:
    frozen = protocol["frozen_inputs"]
    checks = {
        "train_csv": sha256_file(ROOT / "data/open/train.csv") == frozen["train_csv_sha256"],
        "catalog_csv": sha256_file(ROOT / "data/open/vessel_catalog.csv") == frozen["catalog_csv_sha256"],
        "stress_run_summary": sha256_file(ROOT / frozen["stress_run_summary"]["file"])
            == frozen["stress_run_summary"]["sha256"],
    }
    source = ROOT / stress["frozen_inputs"]["source_run"]
    folds = {}
    for fold in stress["frozen_inputs"]["folds"]:
        folds[str(fold["fold"])] = {
            "model_file": sha256_file(source / fold["model_file"]) == fold["model_file_sha256"],
            "clean_cases_file": sha256_file(source / fold["clean_cases_file"]) == fold["clean_cases_sha256"],
        }
    row = frozen["row_split"]
    row_source = ROOT / row["source_run"]
    row_checks = {
        "split_file": sha256_file(ROOT / row["split_file"]) == row["split_sha256"],
        "model_file": sha256_file(row_source / row["model_file"]) == row["model_file_sha256"],
        "clean_cases_file": sha256_file(row_source / row["clean_cases_file"]) == row["clean_cases_sha256"],
        "diagnose_file": sha256_file(row_source / row["diagnose_file"]) == row["diagnose_sha256"],
    }
    failed = [k for k, ok in checks.items() if not ok]
    failed += [f"group_fold{f}_{k}" for f, inner in folds.items() for k, ok in inner.items() if not ok]
    failed += [f"row_{k}" for k, ok in row_checks.items() if not ok]
    if failed:
        raise ValueError("고정 입력 해시가 다릅니다: " + ", ".join(failed))
    config = baseline_config()
    if config["override_policy"] != "callsign_and_name":
        raise ValueError("기준 override 정책이 callsign_and_name이 아닙니다")
    return {"input_hashes_verified": checks, "group_fold_hashes_verified": folds,
            "row_split_hashes_verified": row_checks, "baseline_config": config}


def context_from(split_name: str, model_path: Path):
    """Reload one fold without retraining; lookup tables come from its training rows only."""
    config = baseline_config()
    fit, valid, catalog, audit = load_split(split_name)
    index = ml.CatalogIndex(catalog)
    model = runtime.ReusedBooster(model_path.read_text(encoding="utf-8"))
    res = runtime.resources(fit, catalog, config)
    lookup_ids = set(res["callsign"].values())
    lookup_ids.update(v for votes in res["name_lookup"].values() for v in votes)
    if not lookup_ids <= set(fit["target_vessel_id"]):
        raise ValueError("조회표에 학습 외 target이 포함됐습니다")
    return {"split_name": split_name, "config": config, "fit": fit, "valid": valid,
            "catalog": catalog, "index": index, "model": model, "res": res,
            "base_inputs": valid.drop(columns=["target_vessel_id"]), "audit": audit,
            "train_callsigns": sorted(res["callsign"].keys()),
            "reloaded_model_sha256": runtime.model_sha256(model)}


def predict(context: dict, frame: pd.DataFrame) -> tuple[list[dict], float]:
    started = time.monotonic()
    details = runtime.predict_details(context["model"], frame, context["index"],
                                      context["res"], context["config"])
    seconds = time.monotonic() - started
    if not {d["final_pred_id"] for d in details} <= set(context["index"].ids):
        raise AssertionError("예측 ID가 카탈로그 밖에 있습니다")
    return details, seconds


def independence_check(context: dict, frame: pd.DataFrame, details: list[dict],
                       columns: dict[str, int]) -> dict:
    """G2: every policy must decide identically alone and in a reversed subset."""
    korean = np.array([is_korean(value) for value in frame["vessel_name"]])
    picked: list[int] = []
    for pool in (np.flatnonzero(korean), np.flatnonzero(~korean)):
        picked.extend(int(i) for i in pool[:2])
    for candidate in (0, len(frame) // 2, len(frame) - 1, 1, len(frame) - 2):
        if len(set(picked)) >= INDEPENDENCE_ROWS:
            break
        if 0 <= candidate < len(frame):
            picked.append(int(candidate))
    picked = sorted(set(picked))
    reverse, _ = predict(context, frame.iloc[picked[::-1]])
    singles = [predict(context, frame.iloc[[i]])[0][0] for i in picked]
    index = context["index"]
    for policy in gate.POLICIES:
        full = [gate.apply_policy(details[i], policy, index, columns)["final_pred_id"] for i in picked]
        back = [gate.apply_policy(row, policy, index, columns)["final_pred_id"] for row in reverse][::-1]
        alone = [gate.apply_policy(row, policy, index, columns)["final_pred_id"] for row in singles]
        if back != full or alone != full:
            raise ValueError(f"행 독립성 검사 실패: {policy}")
    return {"checked_rows": len(picked), "passed": True, "policies": list(gate.POLICIES),
            "both_languages": sorted({bool(korean[i]) for i in picked}) == [False, True],
            "modes": ["reversed_subset", "single_row"]}


def evaluate(context: dict, details: list[dict], columns: dict[str, int]) -> dict:
    """Per-policy counts. Labels are joined only here, after predictions are final."""
    valid = context["valid"]
    truth = valid["target_vessel_id"].astype(str).tolist()
    korean = [bool(is_korean(value)) for value in valid["vessel_name"]]
    out = {}
    for policy in gate.POLICIES:
        rows = gate.decide_rows(details, policy, context["index"], columns)
        finals = [row["final_pred_id"] for row in rows]
        official = calculate_score(truth, finals, valid["vessel_name"])
        correct = [final == want for final, want in zip(finals, truth)]
        model_correct = [row["model_pred_id"] == want for row, want in zip(rows, truth)]
        applied = {source: 0 for source in gate.SOURCES}
        available = {source: 0 for source in gate.SOURCES}
        blocked = {reason: 0 for reason in gate.BLOCK_REASONS}
        applied_wrong = blocked_would_be_correct = rescued = harm = 0
        for row, want, ok, model_ok in zip(rows, truth, correct, model_correct):
            for source in gate.SOURCES:
                decision = row["decisions"][source]
                if not decision["available"]:
                    continue
                available[source] += 1
                if decision["applied"]:
                    applied[source] += 1
                else:
                    if decision["reason"] in blocked:
                        blocked[decision["reason"]] += 1
                    if decision["lookup_id"] == want:
                        blocked_would_be_correct += 1
            if row["override_source"] != "none":
                applied_wrong += not ok
                rescued += (not model_ok) and ok
                harm += model_ok and (not ok)
        out[policy] = {
            "rows": len(rows), "correct": int(sum(correct)),
            "korean_rows": int(sum(korean)), "korean_correct": int(sum(
                ok for ok, flag in zip(correct, korean) if flag)),
            "non_korean_rows": int(sum(not flag for flag in korean)),
            "non_korean_correct": int(sum(ok for ok, flag in zip(correct, korean) if not flag)),
            "weighted_raw": 0.7 * official["korean"] + 0.3 * official["non_korean"],
            "official_weighted_rounded": official["weighted"],
            "korean_accuracy": official["korean"], "non_korean_accuracy": official["non_korean"],
            "model_only_correct": int(sum(model_correct)),
            "lookups_available": available, "lookups_applied": applied,
            "blocked_by_reason": blocked, "applied_wrong": applied_wrong,
            "blocked_would_be_correct": blocked_would_be_correct,
            "rescued": rescued, "harm": harm,
            "final_pred_sha256": runtime.array_digest(np.array(finals, dtype=object).astype("U32")),
        }
    return out


def combine(per_fold: list[dict]) -> dict:
    """Pool folds for one scenario, per policy."""
    out = {}
    for policy in gate.POLICIES:
        parts = [fold[policy] for fold in per_fold]
        totals = {key: int(sum(p[key] for p in parts)) for key in
                  ("rows", "correct", "korean_rows", "korean_correct", "non_korean_rows",
                   "non_korean_correct", "model_only_correct", "applied_wrong",
                   "blocked_would_be_correct", "rescued", "harm")}
        totals["lookups_available"] = {s: int(sum(p["lookups_available"][s] for p in parts))
                                       for s in gate.SOURCES}
        totals["lookups_applied"] = {s: int(sum(p["lookups_applied"][s] for p in parts))
                                     for s in gate.SOURCES}
        totals["blocked_by_reason"] = {r: int(sum(p["blocked_by_reason"][r] for p in parts))
                                       for r in gate.BLOCK_REASONS}
        korean_accuracy = (totals["korean_correct"] / totals["korean_rows"]
                           if totals["korean_rows"] else 0.0)
        non_korean_accuracy = (totals["non_korean_correct"] / totals["non_korean_rows"]
                               if totals["non_korean_rows"] else 0.0)
        totals["korean_accuracy"] = korean_accuracy
        totals["non_korean_accuracy"] = non_korean_accuracy
        totals["weighted_raw"] = 0.7 * korean_accuracy + 0.3 * non_korean_accuracy
        totals["accuracy"] = totals["correct"] / totals["rows"] if totals["rows"] else None
        out[policy] = totals
    return out


def g0_check(label: str, details: list[dict], stored: list[dict], expect_candidates: str,
             expect_features: str) -> dict:
    hashes = prediction_input_hashes(details)
    by_id = {case["sample_id"]: case["final_pred_id"] for case in stored}
    replay = {d["sample_id"]: d["final_pred_id"] for d in details}
    mismatched = [key for key, value in replay.items() if by_id.get(key) != value]
    return {"label": label, "rows": len(replay),
            "candidates_sha256_matches": hashes["candidates_sha256"] == expect_candidates,
            "features_sha256_matches": hashes["features_sha256"] == expect_features,
            "final_pred_matched": len(replay) - len(mismatched),
            "final_pred_mismatched": len(mismatched),
            "all_rows_match": not mismatched and len(replay) == len(by_id)}


def e1_policies_identical(per_fold: list[dict], policy: str | None = None) -> bool:
    """G1: every policy's per-row final predictions must equal P0's, fold by fold."""
    checked = gate.POLICIES if policy is None else (policy,)
    return all(fold[name]["final_pred_sha256"] == fold["P0_always"]["final_pred_sha256"]
               for fold in per_fold for name in checked)


def summary_markdown(summary: dict) -> str:
    lines = ["# v11 override 게이트 비교 (탐색용)", "",
             f"- 실행 ID: `{summary['run_id']}`",
             f"- 프로토콜: `{summary['protocol_version']}` (SHA-256 `{summary['protocol_sha256']}`)",
             "- 재학습 없음. H·B는 선택한 충돌 비율·잡음 강도에 의존하며 리더보드 변화 예측이 아니다.", "",
             "| 시나리오 | 정책 | 전체 | 한글 | 비한글 | 적용 | 차단 | applied_wrong | blocked_would_be_correct | rescued | harm |",
             "|---|---|---|---|---|---:|---:|---:|---:|---:|---:|"]
    for scenario, policies in summary["scenarios"].items():
        for policy, m in policies.items():
            applied = sum(m["lookups_applied"].values())
            blocked = sum(m["blocked_by_reason"].values())
            lines.append(
                f"| `{scenario}` | `{policy}` | {m['correct']}/{m['rows']} "
                f"| {m['korean_correct']}/{m['korean_rows']} "
                f"| {m['non_korean_correct']}/{m['non_korean_rows']} "
                f"| {applied} | {blocked} | {m['applied_wrong']} "
                f"| {m['blocked_would_be_correct']} | {m['rescued']} | {m['harm']} |")
    lines += ["", "## 정책별 H / B / net", "", "| 정책 | H (harm) | B (rescued) | net | 적격 |",
              "|---|---:|---:|---:|---|"]
    for policy in gate.POLICIES:
        totals = summary["policy_totals"][policy]
        lines.append(f"| `{policy}` | {totals['H']} | {totals['B']} | {totals['net']} "
                     f"| {summary['eligibility'][policy]['eligible']} |")
    lines += ["", f"- 순위: {' > '.join(summary['ranking']) or '(없음)'}",
              f"- 권고 진단 대상: {', '.join(summary['recommended']) or '(없음)'}", "",
              "## 게이트", "",
              f"- G0: {summary['gates']['G0_clean_replay']['passed']}",
              f"- G1 (E1 정책 동일): {summary['gates']['G1_policy_equivalence_E1']['passed']}",
              f"- G2 (행 독립성): {summary['gates']['G2_row_independence']['passed']}",
              f"- G3 (섭동 감사): 미적용 {summary['gates']['G3_perturbation_audit']['not_applicable']}", ""]
    return "\n".join(lines) + "\n"


def run(*, g0_only: bool = False) -> Path:
    protocol, stress = load_protocols()
    verification = verify_frozen_inputs(protocol, stress)
    constants = gate.check_constants(protocol)
    columns = gate.feature_columns(runtime.FEATURES)
    before = source_hashes()
    zips_before = zip_snapshot()
    group_source = ROOT / stress["frozen_inputs"]["source_run"]
    row_frozen = protocol["frozen_inputs"]["row_split"]
    row_source = ROOT / row_frozen["source_run"]
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "experiments" / "results" / f"agent_v11_override_gate_{stamp}_{uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    local = run_dir / "local_only"
    local.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "pre_run_intent.json", {
        "run_id": run_dir.name, "protocol_version": PROTOCOL_VERSION,
        "protocol_file": PROTOCOL_PATH.relative_to(ROOT).as_posix(),
        "protocol_sha256": PROTOCOL_SHA256,
        "stress_protocol_sha256": STRESS_SHA256,
        "frozen_inputs_verified": verification["input_hashes_verified"],
        "group_fold_hashes_verified": verification["group_fold_hashes_verified"],
        "row_split_hashes_verified": verification["row_split_hashes_verified"],
        "baseline_config": verification["baseline_config"],
        "gate_constants": constants, "gate_feature_columns": columns,
        "c3": {"scenario_id": c3.SCENARIO_ID, "rate": c3.RATE,
               "protocol_string_for_hash": "v11-stress-oof-1", "selection_salt": c3.SELECTION_SALT,
               "rank_range": [c3.RANK_LOW, c3.RANK_HIGH]},
        "scenarios": list(GROUP_SCENARIOS) + list(ROW_SCENARIOS),
        "source_sha256": before, "zip_sha256_before": zips_before,
        "retraining": False, "leaderboard_submission": False, "g0_only": g0_only,
        "pre_registered_before_any_gate_prediction": True,
    })
    print(f"실행 폴더: {run_dir}", flush=True)

    scenarios: dict[str, list[dict]] = {name: [] for name in GROUP_SCENARIOS + ROW_SCENARIOS}
    audits: dict[str, list[dict]] = {name: [] for name in GROUP_SCENARIOS + ROW_SCENARIOS}
    independence: list[dict] = []
    g0 = {"per_split": [], "passed": False}
    c3_columns: dict[int, list] = {}

    # --- pass 1: clean replay (G0), E1 and the C3 plan, which needs the clean ranking ---
    for fold in stress["frozen_inputs"]["folds"]:
        fold_id = fold["fold"]
        print(f"G0 group fold{fold_id}", flush=True)
        context = context_from(fold["split_name"], group_source / fold["model_file"])
        details, seconds = predict(context, context["base_inputs"])
        stored = json.loads((group_source / fold["clean_cases_file"]).read_text(encoding="utf-8"))["cases"]
        check = g0_check(f"group_fold{fold_id}", details, stored,
                         fold["clean_candidates_sha256"], fold["clean_features_sha256"])
        g0["per_split"].append(check)
        print(f"  해시 {check['candidates_sha256_matches']}/{check['features_sha256_matches']}, "
              f"예측 {check['final_pred_matched']}/{check['rows']}", flush=True)
        scenarios["E1_group_clean"].append(evaluate(context, details, columns))
        audits["E1_group_clean"].append({"scenario_id": "E1_group_clean", "rows": len(details),
                                         "changed_rows": 0, "not_applicable": False,
                                         "predict_seconds": round(seconds, 2)})
        independence.append({"scenario": "E1_group_clean", "split": fold["split_name"]}
                            | independence_check(context, context["base_inputs"], details, columns))
        frame3, audit3, plans = c3.perturb_frame(context["base_inputs"], details,
                                                 context["res"]["callsign"], context["index"].ids)
        c3_columns[fold_id] = list(frame3["call_sign"])
        audits["E2_group_C3"].append(audit3)
        write_json(local / f"c3_plan_group_fold{fold_id}.json",
                   {"scenario_id": c3.SCENARIO_ID, "fold": fold_id,
                    "rows": [{k: p[k] for k in ("sample_id", "selected", "eligible",
                                                "injected_rank", "injected_vessel_id", "changes")}
                             for p in plans]})
        del details

    print("G0 row_seed42", flush=True)
    row_context = context_from("row_seed42", row_source / row_frozen["model_file"])
    row_details, row_seconds = predict(row_context, row_context["base_inputs"])
    row_stored = json.loads((row_source / row_frozen["clean_cases_file"]).read_text(encoding="utf-8"))["cases"]
    row_check = g0_check("row_seed42", row_details, row_stored,
                         row_frozen["clean_candidates_sha256"], row_frozen["clean_features_sha256"])
    g0["per_split"].append(row_check)
    print(f"  해시 {row_check['candidates_sha256_matches']}/{row_check['features_sha256_matches']}, "
          f"예측 {row_check['final_pred_matched']}/{row_check['rows']}", flush=True)
    scenarios["E3_row_clean"].append(evaluate(row_context, row_details, columns))
    audits["E3_row_clean"].append({"scenario_id": "E3_row_clean", "rows": len(row_details),
                                   "changed_rows": 0, "not_applicable": False,
                                   "predict_seconds": round(row_seconds, 2)})
    independence.append({"scenario": "E3_row_clean", "split": "row_seed42"}
                        | independence_check(row_context, row_context["base_inputs"], row_details, columns))
    del row_details

    g0["rows"] = sum(c["rows"] for c in g0["per_split"])
    g0["final_pred_matched"] = sum(c["final_pred_matched"] for c in g0["per_split"])
    g0["passed"] = all(c["all_rows_match"] and c["candidates_sha256_matches"]
                       and c["features_sha256_matches"] for c in g0["per_split"])
    write_json(run_dir / "g0_clean_replay.json", g0)
    if not g0["passed"]:
        write_json(run_dir / "completion_verification.json", {
            "run_id": run_dir.name, "completed": False, "reason": "G0_clean_replay_failed",
            "gates": {"G0_clean_replay": g0}, "scenarios_run": [],
            "source_sha256_unchanged": source_hashes() == before,
            "zip_sha256_unchanged": zip_snapshot() == zips_before,
            "leaderboard_submission": False})
        raise SystemExit(f"G0 실패: 게이트 시나리오를 실행하지 않았습니다. {run_dir}")

    # G1 is checked on the E1 numbers before any perturbed scenario is interpreted.
    e1 = combine(scenarios["E1_group_clean"])
    e1_identical = e1_policies_identical(scenarios["E1_group_clean"])
    g1 = {"scenario": "E1_group_clean", "passed": bool(e1_identical),
          "per_policy_correct": {policy: e1[policy]["correct"] for policy in gate.POLICIES},
          "note": "네 정책의 행별 최종 예측 해시가 fold마다 같아야 한다"}
    write_json(run_dir / "g1_policy_equivalence.json", g1)
    if not g1["passed"]:
        write_json(run_dir / "completion_verification.json", {
            "run_id": run_dir.name, "completed": False, "reason": "G1_policy_equivalence_failed",
            "gates": {"G0_clean_replay": g0, "G1_policy_equivalence_E1": g1},
            "scenarios_run": ["E1_group_clean"],
            "source_sha256_unchanged": source_hashes() == before,
            "zip_sha256_unchanged": zip_snapshot() == zips_before,
            "leaderboard_submission": False})
        raise SystemExit(f"G1 실패: E1에서 정책이 동일하지 않습니다. {run_dir}")

    # --- pass 2: perturbed scenarios ---
    if not g0_only:
        for fold in stress["frozen_inputs"]["folds"]:
            fold_id = fold["fold"]
            context = context_from(fold["split_name"], group_source / fold["model_file"])
            print(f"E2_group_C2 fold{fold_id}", flush=True)
            frame, audit, _ = perturb.perturb_frame("C2_callsign_collision_10pct",
                                                    context["base_inputs"], context["train_callsigns"])
            details, seconds = predict(context, frame)
            scenarios["E2_group_C2"].append(evaluate(context, details, columns))
            audits["E2_group_C2"].append(audit | {"predict_seconds": round(seconds, 2)})
            independence.append({"scenario": "E2_group_C2", "split": fold["split_name"]}
                                | independence_check(context, frame, details, columns))
            del details
            print(f"E2_group_C3 fold{fold_id}", flush=True)
            frame3 = context["base_inputs"].copy()
            frame3["call_sign"] = pd.Series(c3_columns[fold_id], index=frame3.index, dtype=object)
            details3, seconds3 = predict(context, frame3)
            scenarios["E2_group_C3"].append(evaluate(context, details3, columns))
            audits["E2_group_C3"][fold_id]["predict_seconds"] = round(seconds3, 2)
            independence.append({"scenario": "E2_group_C3", "split": fold["split_name"]}
                                | independence_check(context, frame3, details3, columns))
            del details3
        for scenario in BENEFIT_SCENARIOS:
            print(f"{scenario} row_seed42", flush=True)
            frame, audit, _ = perturb.perturb_frame(ROW_PERTURBATION[scenario],
                                                    row_context["base_inputs"])
            details, seconds = predict(row_context, frame)
            scenarios[scenario].append(evaluate(row_context, details, columns))
            audits[scenario].append(audit | {"predict_seconds": round(seconds, 2)})
            independence.append({"scenario": scenario, "split": "row_seed42"}
                                | independence_check(row_context, frame, details, columns))
            del details

    pooled = {name: combine(folds) for name, folds in scenarios.items() if folds}
    policy_totals = {}
    for policy in gate.POLICIES:
        harm = sum(pooled[s][policy]["harm"] for s in HARM_SCENARIOS if s in pooled)
        benefit = sum(pooled[s][policy]["rescued"] for s in BENEFIT_SCENARIOS if s in pooled)
        policy_totals[policy] = {"H": int(harm), "B": int(benefit), "net": int(benefit - harm)}
    row_clean = pooled.get("E3_row_clean", {})
    base_row_correct = row_clean.get("P0_always", {}).get("correct")
    gates = {
        "G0_clean_replay": g0,
        "G1_policy_equivalence_E1": g1,
        "G2_row_independence": {
            "passed": all(item["passed"] and item["both_languages"] for item in independence),
            "checks": len(independence),
            "min_checked_rows": min((item["checked_rows"] for item in independence), default=0),
        },
        "G3_perturbation_audit": {
            "per_scenario": {name: {"rows": sum(a["rows"] for a in items),
                                    "rule_eligible_rows": sum(a.get("rule_eligible_rows", a["rows"])
                                                              for a in items),
                                    "changed_rows": sum(a["changed_rows"] for a in items),
                                    "not_applicable": all(a.get("not_applicable", False)
                                                          for a in items)}
                             for name, items in audits.items() if items},
            "not_applicable": [name for name, items in audits.items()
                               if items and all(a.get("not_applicable", False) for a in items)],
            "c3_injected_rank_counts": {str(i): audits["E2_group_C3"][i].get("injected_rank_counts", {})
                                        for i in range(len(audits["E2_group_C3"]))},
        },
    }
    eligibility = {}
    for policy in gate.POLICIES:
        identical = e1_policies_identical(scenarios["E1_group_clean"], policy)
        keeps_rows = (base_row_correct is None
                      or row_clean.get(policy, {}).get("correct", -1) >= base_row_correct)
        ok = bool(g0["passed"] and g1["passed"] and gates["G2_row_independence"]["passed"]
                  and not gates["G3_perturbation_audit"]["not_applicable"]
                  and identical and keeps_rows)
        eligibility[policy] = {"eligible": ok, "e1_identical_to_p0": bool(identical),
                               "row_clean_loses_no_row": bool(keeps_rows),
                               "row_clean_correct": row_clean.get(policy, {}).get("correct")}
    nets = {policy: policy_totals[policy]["net"] for policy in gate.POLICIES}
    ranking = gate.rank_policies(nets, {p: eligibility[p]["eligible"] for p in gate.POLICIES})
    summary = {
        "run_id": run_dir.name, "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256, "retraining": False, "g0_only": g0_only,
        "gate_constants": constants,
        "scenarios": pooled, "per_fold": scenarios, "perturbation_audits": audits,
        "policy_totals": policy_totals, "eligibility": eligibility,
        "ranking": ranking, "recommended": gate.recommended(ranking),
        "gates": gates, "row_independence": independence,
        "decision_rule": protocol["decision_rule"],
        "interpretation_limits": protocol["interpretation_limits"],
        "caveat": protocol["decision_rule"]["caveat"],
        "leaderboard_submission": False,
    }
    write_json(run_dir / "summary.json", summary)
    (run_dir / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    for name, policies in pooled.items():
        write_json(run_dir / f"gate_{name}.json",
                   {"scenario_id": name, "policies": policies,
                    "perturbation_audit": audits.get(name)})
    after = source_hashes()
    write_json(run_dir / "completion_verification.json", {
        "run_id": run_dir.name, "completed": True,
        "scenarios_run": sorted(pooled),
        "gates": {"G0_clean_replay": {"passed": g0["passed"], "rows": g0["rows"],
                                      "final_pred_matched": g0["final_pred_matched"]},
                  "G1_policy_equivalence_E1": {"passed": g1["passed"]},
                  "G2_row_independence": gates["G2_row_independence"],
                  "G3_perturbation_audit": {"not_applicable": gates["G3_perturbation_audit"]["not_applicable"]}},
        "source_sha256_unchanged": after == before,
        "source_sha256_changed_files": sorted(k for k in before if before[k] != after.get(k)),
        "zip_sha256_unchanged": zip_snapshot() == zips_before,
        "retraining": False, "new_zip_created": False, "leaderboard_submission": False,
        "row_level_files": "experiments/results/<run>/local_only (never sent to an LLM, never committed)",
    })
    print(f"완료: {run_dir / 'summary.json'}", flush=True)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--g0-only", action="store_true", help="G0·G1까지만 수행하고 멈춥니다")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(run(g0_only=args.g0_only))
