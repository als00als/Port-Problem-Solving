"""Run the frozen Korean augmentation protocol (v11-korean-augment-1).

A0 reuses the frozen fold models; A1-A4 retrain per fold with one perturbed copy of every
Korean training query. Evaluation covers the clean inputs, three matched scenarios
(report only) and six held-out scenarios (the decision).

    .\\.venv_competition_py311\\Scripts\\python.exe scripts\\run_v11_korean_augment.py

``--gates-only`` stops after G0 and G0b. The run folder is created only once every
pre-check passes, and an existing folder is never reused. Expect 3-4 hours for a full run.
"""

from __future__ import annotations

# Import torch before numpy/pandas on this Windows environment (see experiments/LOG.md).
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
from agent import korean_augment as aug  # noqa: E402
from agent import retrain_runtime as runtime  # noqa: E402
from agent import stress_heldout as heldout  # noqa: E402
from agent import stress_perturbations as perturb  # noqa: E402
from agent.error_diagnostics import diagnose_rows  # noqa: E402
from agent.retrain_config import baseline_config  # noqa: E402
from scripts.v11_improvement_worker import (load_split, prediction_input_hashes,  # noqa: E402
                                            sha256_file, write_json)
from src.metric import calculate_score, is_korean  # noqa: E402

ml = runtime.ml

PROTOCOL_PATH = ROOT / "experiments" / "protocols" / "v11_korean_augment_v1.json"
PROTOCOL_SHA256 = "b3bf281ea0acb37f4c1a30d8ec7586476294ce7d5677e74e51fcb944f458f85c"
PROTOCOL_VERSION = "v11-korean-augment-1"
STRESS_PATH = ROOT / "experiments" / "protocols" / "v11_stress_oof_v1.json"
STRESS_SHA256 = "4fc3f8ba3fead033185470caf3b86452b32eb13e4db3df2515df22b3f11fc86a"

CLEAN = "E0_clean"
MATCHED = ("T1", "N1", "N2_delete_korean")
HELD_OUT = ("T2", "T2b", "T4", "N3", "N6", "N7")
# scenario id -> (module, stress/held-out scenario name)
SOURCES = {
    "T1": ("stress", "T1_tonnage_both_missing"),
    "N1": ("stress", "N1_spacing"),
    "N2_delete_korean": ("stress", "N2_char_edit_one"),
    "T2": ("stress", "T2_tonnage_noise_20pct"),
    "N3": ("stress", "N3_affix"),
    "T2b": ("heldout", "T2b_tonnage_noise_30pct"),
    "T4": ("heldout", "T4_gross_only_missing"),
    "N6": ("heldout", "N6_syllable_duplicate"),
    "N7": ("heldout", "N7_korean_transpose"),
}
KOREAN_DELETE_LABEL = "n2_korean_delete_syllable"
GAIN_THRESHOLD = 24
NON_KOREAN_TOLERANCE = -3
CODE_FILES = (
    "agent/korean_augment.py", "agent/stress_heldout.py", "agent/stress_perturbations.py",
    "agent/retrain_runtime.py", "agent/retrain_config.py", "agent/error_diagnostics.py",
    "agent/team_v11/source/match_lib.py", "agent/team_v11/source/train.py",
    "scripts/v11_improvement_worker.py", "scripts/run_v11_korean_augment.py",
)
INDEPENDENCE_ROWS = 5


def zip_snapshot() -> dict[str, dict[str, object]]:
    return {p.relative_to(ROOT).as_posix(): {"sha256": sha256_file(p), "size": p.stat().st_size}
            for p in sorted((ROOT / "submissions").glob("*.zip"))}


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
    declared = set(protocol["augmentation"]["variants"])
    if declared != set(aug.VARIANTS):
        raise ValueError(f"변형 목록 불일치: {sorted(declared)} vs {sorted(aug.VARIANTS)}")
    matched = set(protocol["evaluation"]["matched_scenarios_report_only"])
    if matched != set(MATCHED):
        raise ValueError(f"matched 시나리오 불일치: {sorted(matched)}")
    held = set(protocol["evaluation"]["held_out_scenarios_decision"])
    expected_held = {"T2", "T2b_tonnage_noise_30pct", "T4_gross_only_missing", "N3",
                     "N6_syllable_duplicate", "N7_korean_transpose"}
    if held != expected_held:
        raise ValueError(f"held-out 시나리오 불일치: {sorted(held)}")
    rule = protocol["decision_rule"]["robustness_gain"]
    if str(GAIN_THRESHOLD) not in rule:
        raise ValueError("robustness_gain 임계값이 코드 상수와 다릅니다")
    if str(NON_KOREAN_TOLERANCE) not in protocol["gates"]["G2_clean_non_regression"]:
        raise ValueError("G2 비한글 허용치가 코드 상수와 다릅니다")
    factors = protocol["augmentation"]["variants"]["A2_tonnage_noise"]
    for value in aug.A2_FACTORS:
        if str(value) not in factors:
            raise ValueError(f"A2 계수 {value}가 프로토콜 문구에 없습니다")
    return protocol, stress


def verify_frozen_inputs(protocol: dict, stress: dict) -> dict:
    frozen = protocol["frozen_inputs"]
    checks = {
        "train_csv": sha256_file(ROOT / "data/open/train.csv") == frozen["train_csv_sha256"],
        "catalog_csv": sha256_file(ROOT / "data/open/vessel_catalog.csv") == frozen["catalog_csv_sha256"],
        "split_file": sha256_file(ROOT / frozen["split_file"]) == frozen["split_sha256"],
    }
    source = ROOT / frozen["a0_source_run"]
    folds = {}
    for fold in stress["frozen_inputs"]["folds"]:
        key = str(fold["fold"])
        folds[key] = {
            "model_file": sha256_file(source / fold["model_file"]) == fold["model_file_sha256"],
            "clean_cases_file": sha256_file(source / fold["clean_cases_file"]) == fold["clean_cases_sha256"],
        }
    failed = [k for k, ok in checks.items() if not ok]
    failed += [f"fold{f}_{k}" for f, inner in folds.items() for k, ok in inner.items() if not ok]
    if failed:
        raise ValueError("고정 입력 해시가 다릅니다: " + ", ".join(failed))
    config = baseline_config()
    if config["override_policy"] != "callsign_and_name" or config["ranker_family"] != "lightgbm_v11":
        raise ValueError("기준 설정이 프로토콜과 다릅니다")
    return {"input_hashes_verified": checks, "a0_fold_hashes_verified": folds,
            "baseline_config": config}


def fold_context(stress: dict, fold_id: int):
    frozen = next(f for f in stress["frozen_inputs"]["folds"] if f["fold"] == fold_id)
    config = baseline_config()
    fit, valid, catalog, audit = load_split(frozen["split_name"])
    index = ml.CatalogIndex(catalog)
    source = ROOT / stress["frozen_inputs"]["source_run"]
    a0 = runtime.ReusedBooster((source / frozen["model_file"]).read_text(encoding="utf-8"))
    res = runtime.resources(fit, catalog, config)
    return {"fold": fold_id, "frozen": frozen, "config": config, "fit": fit, "valid": valid,
            "catalog": catalog, "index": index, "a0": a0, "res": res,
            "base_inputs": valid.drop(columns=["target_vessel_id"]), "audit": audit,
            "valid_sample_ids": [str(v) for v in valid["sample_id"]],
            "clean_cases": json.loads((source / frozen["clean_cases_file"])
                                      .read_text(encoding="utf-8"))["cases"]}


def perturbed_frame(scenario: str, base: pd.DataFrame):
    kind, name = SOURCES[scenario]
    if kind == "stress":
        return perturb.perturb_frame(name, base)
    return heldout.perturb_frame(name, base)


def predict(context: dict, model, frame: pd.DataFrame):
    started = time.monotonic()
    details = runtime.predict_details(model, frame, context["index"], context["res"],
                                      context["config"])
    return details, time.monotonic() - started


def independence_check(context: dict, model, frame: pd.DataFrame, details: list[dict]) -> dict:
    korean = np.array([is_korean(v) for v in frame["vessel_name"]])
    picked: list[int] = []
    for pool in (np.flatnonzero(korean), np.flatnonzero(~korean)):
        picked.extend(int(i) for i in pool[:2])
    for candidate in (0, len(frame) // 2, len(frame) - 1, 1, len(frame) - 2):
        if len(set(picked)) >= INDEPENDENCE_ROWS:
            break
        if 0 <= candidate < len(frame):
            picked.append(int(candidate))
    picked = sorted(set(picked))
    full = [details[i]["final_pred_id"] for i in picked]
    reverse, _ = predict(context, model, frame.iloc[picked[::-1]])
    singles = [predict(context, model, frame.iloc[[i]])[0][0]["final_pred_id"] for i in picked]
    if [d["final_pred_id"] for d in reverse][::-1] != full or singles != full:
        raise ValueError("행 독립성 검사에 실패했습니다")
    return {"checked_rows": len(picked), "passed": True,
            "both_languages": sorted({bool(korean[i]) for i in picked}) == [False, True],
            "modes": ["reversed_subset", "single_row"]}


def evaluate(context: dict, details: list[dict], subset: list[bool] | None = None) -> dict:
    """Join labels only here. ``subset`` restricts the reported rows (N2_delete_korean)."""
    valid = context["valid"]
    index = context["index"]
    truth = valid["target_vessel_id"].astype(str).tolist()
    seen = valid["target_vessel_id"].isin(set(context["fit"]["target_vessel_id"])).to_numpy(bool)
    korean = np.array([is_korean(v) for v in valid["vessel_name"]])
    cases = diagnose_rows(details, truth, seen, korean, runtime.FEATURES, index.ids,
                          index.compact, index.id_to_index)
    keep = [True] * len(cases) if subset is None else list(subset)
    rows = [c for c, flag in zip(cases, keep) if flag]
    names = [n for n, flag in zip(valid["vessel_name"], keep) if flag]
    official = calculate_score([c["truth_id"] for c in rows],
                               [c["final_pred_id"] for c in rows], names)
    def group(flag_fn):
        picked = [c for c in rows if flag_fn(c)]
        return {"rows": len(picked), "correct": int(sum(c["final_correct"] for c in picked)),
                "candidate_hits": int(sum(c["truth_in_candidates"] for c in picked))}
    return {
        "rows": len(rows), "all": group(lambda c: True),
        "korean": group(lambda c: c["is_korean"]),
        "non_korean": group(lambda c: not c["is_korean"]),
        "weighted_raw": 0.7 * official["korean"] + 0.3 * official["non_korean"],
        "korean_accuracy": official["korean"], "non_korean_accuracy": official["non_korean"],
        "cases": {c["sample_id"]: (c["final_correct"], c["is_korean"], c["primary_error_type"])
                  for c in rows},
    }


def pair_vs_a0(variant_eval: dict, a0_eval: dict) -> dict:
    newly_wrong, newly_correct, errors = [], [], {}
    for sample_id, (ok, korean, error) in variant_eval["cases"].items():
        before = a0_eval["cases"].get(sample_id)
        if before is None:
            continue
        was_ok = before[0]
        if was_ok and not ok:
            newly_wrong.append((korean, error))
            key = error or "unclassified"
            errors[key] = errors.get(key, 0) + 1
        elif not was_ok and ok:
            newly_correct.append((korean, error))
    def split(items):
        return {"total": len(items), "korean": sum(1 for k, _ in items if k),
                "non_korean": sum(1 for k, _ in items if not k)}
    return {
        "newly_wrong": split(newly_wrong), "newly_correct": split(newly_correct),
        "newly_wrong_primary_error_types": dict(sorted(errors.items(),
                                                       key=lambda kv: (-kv[1], kv[0]))),
        "korean_net_correct": variant_eval["korean"]["correct"] - a0_eval["korean"]["correct"],
        "non_korean_net_correct": (variant_eval["non_korean"]["correct"]
                                   - a0_eval["non_korean"]["correct"]),
        "candidate_recall_equal": (variant_eval["all"]["candidate_hits"]
                                   == a0_eval["all"]["candidate_hits"]),
    }


def combine(per_fold: list[dict]) -> dict:
    out = {"rows": sum(f["rows"] for f in per_fold)}
    for group in ("all", "korean", "non_korean"):
        out[group] = {k: int(sum(f[group][k] for f in per_fold))
                      for k in ("rows", "correct", "candidate_hits")}
    korean = out["korean"]["correct"] / out["korean"]["rows"] if out["korean"]["rows"] else 0.0
    other = (out["non_korean"]["correct"] / out["non_korean"]["rows"]
             if out["non_korean"]["rows"] else 0.0)
    out["korean_accuracy"], out["non_korean_accuracy"] = korean, other
    out["weighted_raw"] = 0.7 * korean + 0.3 * other
    out["cases"] = {k: v for f in per_fold for k, v in f["cases"].items()}
    return out


def strip_cases(payload: dict) -> dict:
    return {k: v for k, v in payload.items() if k != "cases"}


def g0_check(context: dict, details: list[dict]) -> dict:
    frozen = context["frozen"]
    hashes = prediction_input_hashes(details)
    stored = {c["sample_id"]: c["final_pred_id"] for c in context["clean_cases"]}
    replay = {d["sample_id"]: d["final_pred_id"] for d in details}
    bad = [k for k, v in replay.items() if stored.get(k) != v]
    return {"fold": context["fold"], "rows": len(replay),
            "candidates_sha256_matches": hashes["candidates_sha256"] == frozen["clean_candidates_sha256"],
            "features_sha256_matches": hashes["features_sha256"] == frozen["clean_features_sha256"],
            "final_pred_matched": len(replay) - len(bad), "final_pred_mismatched": len(bad),
            "all_rows_match": not bad and len(replay) == len(stored)}


def summary_markdown(summary: dict) -> str:
    lines = ["# v11 한글 강건성 증강 (탐색용)", "",
             f"- 실행 ID: `{summary['run_id']}`",
             f"- 프로토콜: `{summary['protocol_version']}` (SHA-256 `{summary['protocol_sha256']}`)",
             "- 이득 크기를 리더보드 차이와 비교하지 않는다. 탐색적 결과다.", "",
             "## held-out 시나리오 (판정)", "",
             "| 변형 | " + " | ".join(HELD_OUT) + " | 합계 |", "|---" * (len(HELD_OUT) + 2) + "|"]
    for variant in aug.VARIANTS[1:]:
        if variant not in summary["variants"]:
            continue
        row = summary["variants"][variant]
        cells = [str(row["scenarios"][s]["paired"]["korean_net_correct"]) for s in HELD_OUT]
        lines.append(f"| `{variant}` | " + " | ".join(cells)
                     + f" | **{row['held_out_korean_gain']}** |")
    lines += ["", "## matched 시나리오 (보고용)", "",
              "| 변형 | " + " | ".join(MATCHED) + " |", "|---" * (len(MATCHED) + 1) + "|"]
    for variant in aug.VARIANTS[1:]:
        if variant not in summary["variants"]:
            continue
        row = summary["variants"][variant]
        cells = [str(row["scenarios"][s]["paired"]["korean_net_correct"]) for s in MATCHED]
        lines.append(f"| `{variant}` | " + " | ".join(cells) + " |")
    lines += ["", "## E0_clean 과 게이트", "",
              "| 변형 | 전체 | 한글 | 비한글 | 한글 순증 | 비한글 순증 | G2 |",
              "|---|---|---|---|---:|---:|---|"]
    for variant in aug.VARIANTS:
        if variant not in summary["variants"]:
            continue
        row = summary["variants"][variant]
        clean = row["scenarios"][CLEAN]
        agg, paired = clean["aggregate"], clean.get("paired")
        lines.append(
            f"| `{variant}` | {agg['all']['correct']}/{agg['all']['rows']} "
            f"| {agg['korean']['correct']}/{agg['korean']['rows']} "
            f"| {agg['non_korean']['correct']}/{agg['non_korean']['rows']} "
            f"| {'-' if paired is None else paired['korean_net_correct']} "
            f"| {'-' if paired is None else paired['non_korean_net_correct']} "
            f"| {row['gates']['G2'] if variant != 'A0_none' else '기준'} |")
    lines += ["", f"- 권고 변형: **{summary['recommended'] or '없음 (검증된 음성 결과)'}**",
              f"- G0 {summary['gates']['G0_a0_replay']['passed']} / "
              f"G0b {summary['gates']['G0b_training_parity']['passed']} / "
              f"G3 {summary['gates']['G3_row_independence']['passed']}", ""]
    return "\n".join(lines) + "\n"


def run(*, gates_only: bool = False) -> Path:
    protocol, stress = load_protocols()
    verification = verify_frozen_inputs(protocol, stress)
    frozen = protocol["frozen_inputs"]
    before = source_hashes()
    zips_before = zip_snapshot()
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "experiments" / "results" / f"agent_v11_korean_augment_{stamp}_{uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    local = run_dir / "local_only"
    local.mkdir(parents=True, exist_ok=False)
    write_json(run_dir / "pre_run_intent.json", {
        "run_id": run_dir.name, "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256, "stress_protocol_sha256": STRESS_SHA256,
        "frozen_inputs_verified": verification["input_hashes_verified"],
        "a0_fold_hashes_verified": verification["a0_fold_hashes_verified"],
        "baseline_config": verification["baseline_config"],
        "variants": list(aug.VARIANTS), "clean": CLEAN,
        "matched_report_only": list(MATCHED), "held_out_decision": list(HELD_OUT),
        "augmentation_hash_protocol_string": aug.PROTOCOL_VERSION,
        "evaluation_hash_protocol_string": perturb.PROTOCOL_VERSION,
        "a2_factors": list(aug.A2_FACTORS), "gain_threshold": GAIN_THRESHOLD,
        "non_korean_tolerance": NON_KOREAN_TOLERANCE,
        "source_sha256": before, "zip_sha256_before": zips_before,
        "retraining": "A1-A4 only; A0 reuses frozen models",
        "leaderboard_submission": False, "new_zip": False, "gates_only": gates_only,
        "pre_registered_before_any_augmented_training": True,
    })
    print(f"실행 폴더: {run_dir}", flush=True)

    contexts = {}
    g0 = {"per_fold": [], "passed": False}
    a0_evals: dict[str, list[dict]] = {}
    a0_audits: dict[str, list[dict]] = {}
    independence: list[dict] = []

    # --- G0: A0 clean replay -------------------------------------------------------------
    for fold_id in range(5):
        print(f"G0 fold{fold_id}: A0 깨끗한 재현", flush=True)
        context = fold_context(stress, fold_id)
        contexts[fold_id] = context
        details, seconds = predict(context, context["a0"], context["base_inputs"])
        check = g0_check(context, details)
        g0["per_fold"].append(check)
        print(f"  해시 {check['candidates_sha256_matches']}/{check['features_sha256_matches']}, "
              f"예측 {check['final_pred_matched']}/{check['rows']}", flush=True)
        a0_evals.setdefault(CLEAN, []).append(evaluate(context, details))
        a0_audits.setdefault(CLEAN, []).append({"scenario_id": CLEAN, "rows": len(details),
                                                "changed_rows": 0, "not_applicable": False,
                                                "predict_seconds": round(seconds, 2)})
        independence.append({"variant": "A0_none", "scenario": CLEAN, "fold": fold_id}
                            | independence_check(context, context["a0"], context["base_inputs"], details))
    g0["rows"] = sum(c["rows"] for c in g0["per_fold"])
    g0["final_pred_matched"] = sum(c["final_pred_matched"] for c in g0["per_fold"])
    g0["passed"] = all(c["all_rows_match"] and c["candidates_sha256_matches"]
                       and c["features_sha256_matches"] for c in g0["per_fold"])
    write_json(run_dir / "g0_replay.json", g0)

    # --- G0b: training parity with augmentation off --------------------------------------
    g0b = {"per_fold": {}, "fold0_model": None, "passed": False}
    if g0["passed"]:
        for fold_id in range(5):
            print(f"G0b fold{fold_id}: 증강 끈 학습 행렬 해시", flush=True)
            context = contexts[fold_id]
            with runtime.v11_globals(context["config"]):
                x, y, groups, _, stats = runtime.training_matrix(
                    context["fit"], context["catalog"], context["index"], context["config"])
            got = runtime.array_digest(x, y, groups)
            want = frozen["a0_training_matrix_sha256"][str(fold_id)]
            g0b["per_fold"][str(fold_id)] = {"matrix_sha256": got, "matches": got == want,
                                             "rows": int(len(y))}
            print(f"  행렬 해시 일치 {got == want}", flush=True)
            del x, y, groups
        print("G0b fold0: 증강 끈 재학습 (모델 해시)", flush=True)
        context = contexts[0]
        started = time.monotonic()
        model, _, stats = runtime.fit(context["fit"], context["catalog"], context["index"],
                                      context["config"])
        got = runtime.model_sha256(model)
        want = frozen["a0_model_sha256"]["0"]
        g0b["fold0_model"] = {"model_sha256": got, "matches": got == want,
                              "training_seconds": round(time.monotonic() - started, 1),
                              "augmentation_recorded": "augmentation" in stats}
        print(f"  모델 해시 일치 {got == want}", flush=True)
        g0b["passed"] = (all(v["matches"] for v in g0b["per_fold"].values())
                         and g0b["fold0_model"]["matches"])
    write_json(run_dir / "g0b_training_parity.json", g0b)

    if not (g0["passed"] and g0b["passed"]):
        write_json(run_dir / "completion_verification.json", {
            "run_id": run_dir.name, "completed": False,
            "reason": "G0_failed" if not g0["passed"] else "G0b_failed",
            "gates": {"G0_a0_replay": g0, "G0b_training_parity": g0b},
            "variants_trained": [], "source_sha256_unchanged": source_hashes() == before,
            "zip_sha256_unchanged": zip_snapshot() == zips_before,
            "leaderboard_submission": False, "new_zip_created": False})
        raise SystemExit(f"G0/G0b 실패: 변형을 하나도 학습하지 않았습니다. {run_dir}")

    # --- A0 evaluation on every scenario -------------------------------------------------
    for scenario in MATCHED + HELD_OUT:
        for fold_id in range(5):
            context = contexts[fold_id]
            print(f"A0 {scenario} fold{fold_id}", flush=True)
            frame, audit, results = perturbed_frame(scenario, context["base_inputs"])
            details, seconds = predict(context, context["a0"], frame)
            subset = None
            if scenario == "N2_delete_korean":
                subset = [KOREAN_DELETE_LABEL in r["operations"] for r in results]
                audit = dict(audit, reported_rows=int(sum(subset)),
                             reported_filter=KOREAN_DELETE_LABEL)
            a0_evals.setdefault(scenario, []).append(evaluate(context, details, subset))
            a0_audits.setdefault(scenario, []).append(dict(audit, predict_seconds=round(seconds, 2)))
            del details
    a0_pooled = {s: combine(v) for s, v in a0_evals.items()}
    variants: dict[str, dict] = {"A0_none": {
        "scenarios": {s: {"aggregate": strip_cases(a0_pooled[s]), "paired": None,
                          "audit": a0_audits[s]} for s in a0_pooled},
        "gates": {"G2": "기준"}, "held_out_korean_gain": 0,
        "augmentation": None, "retrained": False}}
    for scenario, pooled in a0_pooled.items():
        write_json(run_dir / f"eval_A0_none_{scenario}.json",
                   {"variant": "A0_none", "scenario_id": scenario,
                    "aggregate": strip_cases(pooled), "audit": a0_audits[scenario]})

    # --- A1-A4 ---------------------------------------------------------------------------
    if not gates_only:
        for variant in aug.VARIANTS[1:]:
            fold_evals: dict[str, list[dict]] = {}
            fold_audits: dict[str, list[dict]] = {}
            augment_audits = []
            for fold_id in range(5):
                context = contexts[fold_id]
                print(f"{variant} fold{fold_id}: 증강 학습", flush=True)
                augmenter = aug.make_augmenter(variant, context["valid_sample_ids"])
                started = time.monotonic()
                model, _, stats = runtime.fit(context["fit"], context["catalog"],
                                              context["index"], context["config"],
                                              augment=augmenter)
                merged = aug.merge_audits(stats["augmentation"])
                merged.update({"fold": fold_id, "training_seconds": round(time.monotonic() - started, 1),
                               "training_matrix_rows": stats["training_matrix_rows"],
                               "training_matrix_sha256": stats["training_matrix_sha256"],
                               "model_sha256": runtime.model_sha256(model),
                               "a0_training_matrix_sha256":
                                   frozen["a0_training_matrix_sha256"][str(fold_id)]})
                merged["matrix_differs_from_a0"] = (merged["training_matrix_sha256"]
                                                    != merged["a0_training_matrix_sha256"])
                augment_audits.append(merged)
                (local / f"model_{variant}_fold{fold_id}.txt").write_text(
                    model.booster_.model_to_string(), encoding="utf-8")
                print(f"  복사본 {merged['copies_added']}건(변경 {merged['copies_changed']}), "
                      f"{merged['training_seconds']}s", flush=True)
                for scenario in (CLEAN,) + MATCHED + HELD_OUT:
                    if scenario == CLEAN:
                        frame, audit, results = context["base_inputs"], {
                            "scenario_id": CLEAN, "rows": len(context["base_inputs"]),
                            "changed_rows": 0, "not_applicable": False}, None
                    else:
                        frame, audit, results = perturbed_frame(scenario, context["base_inputs"])
                    details, seconds = predict(context, model, frame)
                    subset = None
                    if scenario == "N2_delete_korean":
                        subset = [KOREAN_DELETE_LABEL in r["operations"] for r in results]
                        audit = dict(audit, reported_rows=int(sum(subset)),
                                     reported_filter=KOREAN_DELETE_LABEL)
                    fold_evals.setdefault(scenario, []).append(evaluate(context, details, subset))
                    fold_audits.setdefault(scenario, []).append(
                        dict(audit, predict_seconds=round(seconds, 2)))
                    if scenario == CLEAN:
                        independence.append({"variant": variant, "scenario": CLEAN, "fold": fold_id}
                                            | independence_check(context, model, frame, details))
                    del details
                del model
            pooled = {s: combine(v) for s, v in fold_evals.items()}
            scenarios = {}
            for scenario, value in pooled.items():
                paired = pair_vs_a0(value, a0_pooled[scenario])
                scenarios[scenario] = {"aggregate": strip_cases(value), "paired": paired,
                                       "audit": fold_audits[scenario]}
                write_json(run_dir / f"eval_{variant}_{scenario}.json",
                           {"variant": variant, "scenario_id": scenario,
                            "aggregate": strip_cases(value), "paired": paired,
                            "audit": fold_audits[scenario]})
            clean = scenarios[CLEAN]["paired"]
            g2 = bool(clean["korean_net_correct"] >= 0
                      and clean["non_korean_net_correct"] >= NON_KOREAN_TOLERANCE
                      and clean["candidate_recall_equal"])
            gain = sum(scenarios[s]["paired"]["korean_net_correct"] for s in HELD_OUT)
            negative = [s for s in HELD_OUT if scenarios[s]["paired"]["korean_net_correct"] < 0]
            variants[variant] = {
                "scenarios": scenarios, "augmentation": augment_audits,
                "held_out_korean_gain": int(gain),
                "held_out_negative_scenarios": negative,
                "meets_robustness_gain": bool(gain >= GAIN_THRESHOLD and not negative),
                "gates": {"G2": g2}, "retrained": True}
            print(f"  {variant}: held-out 한글 순증 합계 {gain}, G2 {g2}", flush=True)

    g3_passed = all(i["passed"] and i["both_languages"] for i in independence)
    gates = {
        "G0_a0_replay": g0, "G0b_training_parity": g0b,
        "G1_augmentation_audit": {
            "per_variant": {v: {"copies_added": sum(a["copies_added"] for a in d["augmentation"]),
                                "copies_changed": sum(a["copies_changed"] for a in d["augmentation"]),
                                "korean_prep_rows": sum(a["korean_prep_rows"] for a in d["augmentation"]),
                                "outer_validation_ids_present":
                                    sum(a["outer_validation_ids_present"] for a in d["augmentation"]),
                                "labels_preserved": all(a["labels_preserved"] for a in d["augmentation"]),
                                "matrix_differs_from_a0": all(a["matrix_differs_from_a0"]
                                                              for a in d["augmentation"])}
                            for v, d in variants.items() if d.get("augmentation")},
        },
        "G2_clean_non_regression": {v: d["gates"]["G2"] for v, d in variants.items()
                                    if v != "A0_none"},
        "G3_row_independence": {"passed": g3_passed, "checks": len(independence),
                                "min_checked_rows": min((i["checked_rows"] for i in independence),
                                                        default=0)},
    }
    eligible = [v for v in aug.VARIANTS[1:] if v in variants
                and variants[v]["gates"]["G2"] and g0["passed"] and g0b["passed"] and g3_passed]
    qualified = [v for v in eligible if variants[v]["meets_robustness_gain"]]
    order = {"A1_tonnage_missing": 0, "A2_tonnage_noise": 1, "A3_name_noise": 2, "A4_mixed": 3}
    qualified.sort(key=lambda v: (-variants[v]["held_out_korean_gain"], order[v]))
    summary = {
        "run_id": run_dir.name, "protocol_version": PROTOCOL_VERSION,
        "protocol_sha256": PROTOCOL_SHA256, "gates_only": gates_only,
        "clean": CLEAN, "matched_report_only": list(MATCHED), "held_out_decision": list(HELD_OUT),
        "variants": variants, "gates": gates, "row_independence": independence,
        "eligible": eligible, "qualified": qualified,
        "recommended": qualified[0] if qualified else None,
        "gain_threshold": GAIN_THRESHOLD, "non_korean_tolerance": NON_KOREAN_TOLERANCE,
        "decision_rule": protocol["decision_rule"],
        "interpretation_limits": protocol["interpretation_limits"],
        "leaderboard_submission": False, "new_zip_created": False,
    }
    write_json(run_dir / "summary.json", summary)
    write_json(run_dir / "augmentation_audit.json",
               {v: d["augmentation"] for v, d in variants.items() if d.get("augmentation")})
    (run_dir / "summary.md").write_text(summary_markdown(summary), encoding="utf-8")
    after = source_hashes()
    write_json(run_dir / "completion_verification.json", {
        "run_id": run_dir.name, "completed": True,
        "variants_trained": [v for v, d in variants.items() if d["retrained"]],
        "gates": {"G0": g0["passed"], "G0b": g0b["passed"],
                  "G2": gates["G2_clean_non_regression"], "G3": g3_passed},
        "recommended": summary["recommended"],
        "source_sha256_unchanged": after == before,
        "source_sha256_changed_files": sorted(k for k in before if before[k] != after.get(k)),
        "zip_sha256_unchanged": zip_snapshot() == zips_before,
        "leaderboard_submission": False, "new_zip_created": False,
        "row_level_files": "experiments/results/<run>/local_only (never sent to an LLM, never committed)",
    })
    print(f"완료: {run_dir / 'summary.json'}", flush=True)
    return run_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gates-only", action="store_true", help="G0·G0b까지만 수행하고 멈춥니다")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(run(gates_only=args.gates_only))
