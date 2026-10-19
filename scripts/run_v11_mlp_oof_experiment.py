"""Run one frozen five-fold OOF comparison of v11 and the fixed CPU MLP."""

from __future__ import annotations

from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from uuid import uuid4

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.mlp_oof_protocol import (PROTOCOL_VERSION, summarize_paired_oof,
                                    validate_group_oof_split)  # noqa: E402
from agent.retrain_config import canonical_name  # noqa: E402
from src.metric import is_korean  # noqa: E402

PYTHON = ROOT / ".venv_competition_py311" / "Scripts" / "python.exe"
WORKER = ROOT / "scripts" / "v11_improvement_worker.py"
SPLIT_FILE = ROOT / "experiments" / "splits" / "agent_v11_mlp_listwise_group5_seed20261019.json"
SPLITS = tuple(f"mlp_oof_fold{i}" for i in range(5))
KNOB, VALUE = "ranker_family", "mlp_listwise_v1_cpu"
SETTING = canonical_name(KNOB, VALUE)
CODE_FILES = (
    "agent/mlp_oof_protocol.py", "agent/retrain_config.py", "agent/retrain_runtime.py",
    "agent/error_diagnostics.py", "agent/team_v11/source/match_lib.py",
    "agent/team_v11/source/train.py", "scripts/v11_improvement_worker.py",
    "scripts/freeze_v11_mlp_oof_split.py", "scripts/run_v11_mlp_oof_experiment.py",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def zip_snapshot() -> dict[str, dict[str, object]]:
    return {path.relative_to(ROOT).as_posix(): {"sha256": sha256_file(path), "size": path.stat().st_size}
            for path in sorted((ROOT / "submissions").glob("*.zip"))}


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def diagnosis_output_path(run_dir: Path, split_name: str) -> Path:
    """Use the filename required by the worker's retrain stage."""
    return run_dir / f"diagnose_{split_name}.json"


def worker(args: list[str], output: Path) -> dict:
    command = [str(PYTHON), str(WORKER), *args, "--output", str(output)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    env.pop("OPENAI_API_KEY", None)
    process = subprocess.Popen(command, cwd=ROOT, env=env, text=True, encoding="utf-8",
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert process.stdout is not None
    for line in process.stdout:
        print("  " + line.rstrip(), flush=True)
    code = process.wait()
    if code:
        raise RuntimeError(f"worker 실패(exit {code}): {' '.join(args[:4])}")
    return json.loads(output.read_text(encoding="utf-8"))


def load_cases(run_dir: Path, relative_path: str) -> list[dict]:
    path = ROOT / relative_path
    if not path.is_relative_to(run_dir):
        raise ValueError("행별 사례는 해당 실행의 local_only에만 저장해야 합니다")
    return json.loads(path.read_text(encoding="utf-8"))["cases"]


def run() -> Path:
    if not PYTHON.is_file():
        raise FileNotFoundError(f"대회 Python 환경을 찾을 수 없습니다: {PYTHON}")
    if not SPLIT_FILE.is_file():
        raise FileNotFoundError("먼저 scripts/freeze_v11_mlp_oof_split.py로 OOF 분할·판정 기준을 고정하세요")
    split = json.loads(SPLIT_FILE.read_text(encoding="utf-8"))
    if split.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError("고정 OOF protocol 버전이 다릅니다")
    import pandas as pd
    train_path = ROOT / "data" / "open" / "train.csv"
    train = pd.read_csv(train_path, encoding="utf-8-sig")
    validate_group_oof_split(train.assign(is_korean=train["vessel_name"].map(is_korean)), split)
    source_hashes = {path: sha256_file(ROOT / path) for path in CODE_FILES}
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "experiments" / "results" / f"agent_v11_mlp_oof_{timestamp}_{uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    zips_before = zip_snapshot()
    train_hash = sha256_file(train_path)
    catalog_hash = sha256_file(ROOT / "data" / "open" / "vessel_catalog.csv")
    if train_hash != split["train_sha256"] or catalog_hash != split["catalog_sha256"]:
        raise ValueError("고정 protocol의 공식 데이터 해시와 현재 입력이 다릅니다")
    intent = {
        "run_id": run_dir.name, "protocol_version": PROTOCOL_VERSION,
        "split_file": SPLIT_FILE.relative_to(ROOT).as_posix(), "split_sha256": sha256_file(SPLIT_FILE),
        "source_sha256": source_hashes,
        "input_sha256": {"train.csv": train_hash, "vessel_catalog.csv": catalog_hash},
        "ranker": {"knob": KNOB, "value": VALUE, "profile": split["ranker_profile"]},
        "unchanged_conditions": ["official candidate generation", "ordered candidates", "v11 features",
                                 "lookup overrides", "outer folds", "validation rows"],
        "decision_gates": split["acceptance_gates"],
        "split_exposure": split["independence_note"], "llm_calls": 0,
        "leaderboard_submission": False, "zip_sha256_before": zips_before,
        "pre_registered_before_model_predictions": True,
    }
    write_json(run_dir / "pre_run_intent.json", intent)
    per_fold = []
    baseline_cases_all, mlp_cases_all = [], []
    for split_name in SPLITS:
        fold_id = int(split_name[-1])
        print(f"{split_name}: 분할 감사·기준 v11 재학습·MLP 비교", flush=True)
        audit = worker(["audit", "--split", split_name], run_dir / f"audit_{split_name}.json")
        baseline = worker(["diagnose", "--split", split_name, "--run-dir", str(run_dir)],
                          diagnosis_output_path(run_dir, split_name))
        alternative = worker(["retrain", "--split", split_name, "--run-dir", str(run_dir),
                              "--knob", KNOB, "--value", VALUE],
                             run_dir / f"mlp_{split_name}.json")
        base_cases = load_cases(run_dir, baseline["local_case_file"])
        mlp_cases = load_cases(run_dir, alternative["local_case_file"])
        baseline_cases_all.extend(base_cases)
        mlp_cases_all.extend(mlp_cases)
        per_fold.append({
            "fold": fold_id, "split_name": split_name, "audit": audit,
            "baseline": baseline["metrics"], "mlp": alternative["metrics"],
            "delta": alternative["delta"], "row_changes": alternative["row_changes"],
            "timing_seconds": {"baseline_training": baseline["timing_seconds"]["training"],
                               "mlp_training": alternative["timing_seconds"]["training"],
                               "baseline_validation_prediction": baseline["timing_seconds"]["validation_prediction"],
                               "mlp_validation_prediction": alternative["timing_seconds"]["validation_prediction"]},
            "training_matrix_sha256": {"baseline": baseline["training_stats"]["training_matrix_sha256"],
                                       "mlp": alternative["training_stats"]["training_matrix_sha256"]},
            "prediction_input_hashes": {"baseline": baseline["prediction_input_hashes"],
                                        "mlp": alternative["prediction_input_hashes"]},
            "checks": {"baseline": baseline["checks"], "mlp": alternative["checks"],
                       "behavior_check": alternative["behavior_check"],
                       "model_sha256": alternative["model_sha256"],
                       "model_artifact": alternative["local_model_file"]},
            "local_case_files": {"baseline": baseline["local_case_file"],
                                 "mlp": alternative["local_case_file"]},
        })
    paired = summarize_paired_oof(baseline_cases_all, mlp_cases_all)
    expected_ids = set(train["sample_id"].astype(str))
    oof_ids = [case["sample_id"] for case in baseline_cases_all]
    if len(oof_ids) != len(set(oof_ids)) or set(oof_ids) != expected_ids:
        raise AssertionError("OOF 결과가 공식 train 각 행을 정확히 한 번씩 포함하지 않습니다")
    if len(mlp_cases_all) != len(baseline_cases_all):
        raise AssertionError("baseline과 MLP OOF 행 수가 다릅니다")
    if any(fold["training_matrix_sha256"]["baseline"] != fold["training_matrix_sha256"]["mlp"]
           or fold["prediction_input_hashes"]["baseline"] != fold["prediction_input_hashes"]["mlp"]
           for fold in per_fold):
        raise AssertionError("baseline과 MLP의 학습 또는 검증 입력이 다릅니다")
    candidate_recall_equal = all(abs(fold["baseline"]["all"]["candidate_recall"]
                                     - fold["mlp"]["all"]["candidate_recall"]) < 1e-15
                                 for fold in per_fold)
    target_visibility = {}
    for subset in ("seen_target", "unseen_target"):
        target_visibility[subset] = {}
        for family in ("baseline", "mlp"):
            rows = sum(fold[family][subset]["rows"] for fold in per_fold)
            correct = sum(fold[family][subset]["correct"] for fold in per_fold)
            target_visibility[subset][family] = {
                "rows": rows, "correct": correct,
                "accuracy": correct / rows if rows else None,
            }
    outcome = {**paired["decision"], "candidate_recall_equal": candidate_recall_equal,
               "accepted": bool(paired["decision"]["mlp_improvement_signal"] and candidate_recall_equal)}
    summary = {
        "run_id": run_dir.name, "protocol_version": PROTOCOL_VERSION,
        "setting": SETTING, "knob": KNOB, "value": VALUE,
        "official_rows": len(train), "folds": per_fold,
        "oof_row_coverage": {"rows": len(oof_ids), "unique_rows": len(set(oof_ids)),
                             "each_official_row_once": True},
        "paired_oof": paired, "candidate_recall_equal": candidate_recall_equal,
        "target_visibility_oof": target_visibility,
        "decision": outcome,
        "timing_seconds_total": {
            "baseline_training": sum(f["timing_seconds"]["baseline_training"] for f in per_fold),
            "mlp_training": sum(f["timing_seconds"]["mlp_training"] for f in per_fold),
            "baseline_validation_prediction": sum(f["timing_seconds"]["baseline_validation_prediction"]
                                                   for f in per_fold),
            "mlp_validation_prediction": sum(f["timing_seconds"]["mlp_validation_prediction"]
                                              for f in per_fold),
        },
        "independence_note": split["independence_note"],
        "leaderboard_submission": False,
    }
    write_json(run_dir / "summary.json", summary)
    sources_after = {path: sha256_file(ROOT / path) for path in CODE_FILES}
    zips_after = zip_snapshot()
    checks = {
        "protocol_pre_registered_before_predictions": True,
        "official_inputs_only": train_hash == split["train_sha256"] and catalog_hash == split["catalog_sha256"],
        "five_fold_target_group_disjointness": all(f["audit"]["target_group_overlap"] == 0 for f in per_fold),
        "each_official_row_oof_once": summary["oof_row_coverage"]["each_official_row_once"],
        "same_training_and_validation_inputs": all(
            f["training_matrix_sha256"]["baseline"] == f["training_matrix_sha256"]["mlp"]
            and f["prediction_input_hashes"]["baseline"] == f["prediction_input_hashes"]["mlp"]
            for f in per_fold),
        "validation_labels_removed_before_prediction": all(
            f["checks"]["mlp"]["validation_label_removed_before_prediction"] for f in per_fold),
        "sample_row_independence": all(f["checks"]["mlp"]["row_independence"]["passed"] for f in per_fold),
        "saved_mlp_roundtrip": all(f["checks"]["mlp"].get("model_save_reload_prediction_equal", False)
                                    for f in per_fold),
        "source_hashes_unchanged": source_hashes == sources_after,
        "zip_hashes_unchanged": zips_before == zips_after,
        "leaderboard_submission": False,
    }
    write_json(run_dir / "completion_verification.json", {
        "run_id": run_dir.name, "checks": checks,
        "source_sha256_before": source_hashes, "source_sha256_after": sources_after,
        "zip_sha256_before": zips_before, "zip_sha256_after": zips_after,
        "decision": outcome, "paired_oof": paired,
    })
    if not all(value for key, value in checks.items()
               if key not in {"leaderboard_submission"}):
        raise AssertionError(f"완료 검증 실패: {[key for key, value in checks.items() if value is False]}")
    print(f"완료: {run_dir / 'summary.json'}", flush=True)
    return run_dir


if __name__ == "__main__":
    print(run())
