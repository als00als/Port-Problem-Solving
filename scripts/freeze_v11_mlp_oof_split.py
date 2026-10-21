"""Freeze the MLP OOF split and decision protocol before model evaluation."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.mlp_oof_protocol import CUDA_PROTOCOL_VERSION, PROTOCOL_VERSION, build_group_oof_split  # noqa: E402
from agent.retrain_runtime import MLP_CUDA_PROFILE, MLP_PROFILE  # noqa: E402
from src.metric import is_korean  # noqa: E402


TRAIN_PATH = ROOT / "data" / "open" / "train.csv"
CATALOG_PATH = ROOT / "data" / "open" / "vessel_catalog.csv"
SPLIT_PATH = ROOT / "experiments" / "splits" / "agent_v11_mlp_listwise_group5_seed20261019.json"
CUDA_SPLIT_PATH = ROOT / "experiments" / "splits" / "agent_v11_mlp_listwise_cuda_group5_seed20261019.json"
FOLD_KEYS = ("train_sample_ids", "valid_sample_ids", "train_target_group_ids", "valid_target_group_ids")


def split_environment() -> dict:
    """Record the library versions that decide the fold assignment."""
    import numpy
    import sklearn
    return {"python": sys.version.split()[0], "numpy": numpy.__version__,
            "pandas": pd.__version__, "scikit_learn": sklearn.__version__}


def fold_digest(result: dict) -> str:
    """Hash only the fold membership so metadata edits stay distinguishable from fold changes."""
    payload = [[fold[key] for key in FOLD_KEYS] for fold in result["folds"]]
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False,
                                     sort_keys=True).encode("utf-8")).hexdigest()


def compare_to(existing: Path) -> dict:
    """Rebuild the partition here and report whether an existing frozen file has the same folds."""
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    train["is_korean"] = train["vessel_name"].map(is_korean).astype(int)
    rebuilt = build_group_oof_split(train, seed=20261019)
    frozen = json.loads(existing.read_text(encoding="utf-8"))
    label = (existing.relative_to(ROOT).as_posix()
             if existing.is_relative_to(ROOT) else existing.as_posix())
    return {"existing_file": label,
            "existing_file_sha256": sha256_file(existing),
            "existing_fold_digest": fold_digest(frozen),
            "rebuilt_fold_digest": fold_digest(rebuilt),
            "folds_identical": fold_digest(frozen) == fold_digest(rebuilt),
            "existing_environment": frozen.get("environment"),
            "rebuilt_environment": split_environment()}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


CUDA_SELECTION_POLICY = ("CPU OOF is the primary analysis; CUDA is a secondary device check "
                         "and cannot be chosen for a better score.")


def freeze_cuda_from_cpu(source: Path, output: Path = CUDA_SPLIT_PATH) -> dict:
    """Derive the CUDA protocol from the pre-registered CPU protocol's own folds.

    Rebuilding the partition here can produce different folds than the environment that
    froze the CPU protocol, so the folds are copied verbatim instead of recomputed.
    """
    if output.exists():
        raise FileExistsError(f"기존 split/protocol 파일은 덮어쓰지 않습니다: {output}")
    cpu = json.loads(source.read_text(encoding="utf-8"))
    if cpu.get("protocol_version") != PROTOCOL_VERSION:
        raise ValueError(f"원본은 CPU 프로토콜 {PROTOCOL_VERSION}이어야 합니다: {cpu.get('protocol_version')}")
    if cpu.get("split_method") != "target_vessel_id_group_oof":
        raise ValueError("원본 split_method가 target_vessel_id_group_oof가 아닙니다")
    if cpu.get("train_sha256") != sha256_file(TRAIN_PATH) or cpu.get("catalog_sha256") != sha256_file(CATALOG_PATH):
        raise ValueError("원본 프로토콜의 공식 데이터 해시가 현재 입력과 다릅니다")
    source_digest = fold_digest(cpu)
    result = dict(cpu)
    result.update({
        "protocol_version": CUDA_PROTOCOL_VERSION,
        "analysis_role": "secondary_device_check",
        "prior_cpu_partial_results_reported_before_cuda_freeze": True,
        "prior_partial_results_reported_before_freeze": True,
        "refrozen_after_partial_results_seen": False,
        "selection_policy": CUDA_SELECTION_POLICY,
        "ranker_profile": dict(MLP_CUDA_PROFILE),
        "mlp": "same full v11 training candidate matrix/features; train-only scaler; query-listwise "
               "softmax; CUDA PyTorch; no pretrained weights",
        "folds_copied_verbatim_from_cpu_protocol": True,
        "derived_from": {
            "file": (source.relative_to(ROOT).as_posix() if source.is_relative_to(ROOT)
                     else source.as_posix()),
            "sha256": sha256_file(source),
            "fold_digest": source_digest,
            "protocol_version": cpu.get("protocol_version"),
            "experiment_label": cpu.get("experiment"),
        },
        "environment": split_environment(),
    })
    if fold_digest(result) != source_digest:
        raise ValueError("fold를 그대로 옮기지 못했습니다")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


def freeze(output: Path = SPLIT_PATH, *, gpu: bool = False,
           refrozen_after_exposure: bool = False) -> dict:
    if output.exists():
        raise FileExistsError(f"기존 split/protocol 파일은 덮어쓰지 않습니다: {output}")
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    catalog_hash = sha256_file(CATALOG_PATH)
    train_hash = sha256_file(TRAIN_PATH)
    train["is_korean"] = train["vessel_name"].map(is_korean).astype(int)
    result = build_group_oof_split(train, seed=20261019)
    result.update({
        "protocol_version": CUDA_PROTOCOL_VERSION if gpu else PROTOCOL_VERSION,
        "split_method": "target_vessel_id_group_oof",
        "train_rows": int(len(train)),
        "target_groups": int(train["target_vessel_id"].nunique()),
        "data_sources": ["data/open/train.csv", "data/open/vessel_catalog.csv"],
        "train_sha256": train_hash,
        "catalog_sha256": catalog_hash,
        "historical_validation_overlap": True,
        "historical_error_analysis_overlap": True,
        "independence_note": "All official target groups were previously exposed to validation or error analysis; this new OOF partition remains exploratory, not an independent performance estimate.",
        "analysis_role": ("secondary_device_check" if gpu else
                          "primary_cpu_oof_refrozen_after_exposure" if refrozen_after_exposure
                          else "primary_cpu_oof"),
        "prior_cpu_partial_results_reported_before_cuda_freeze": bool(gpu),
        "prior_partial_results_reported_before_freeze": bool(gpu or refrozen_after_exposure),
        "refrozen_after_partial_results_seen": bool(refrozen_after_exposure),
        "selection_policy": ("CPU OOF is the primary analysis; CUDA is a secondary device check and cannot be chosen for a better score."
                             if gpu else
                             "Re-frozen after partial CPU fold results were already reported; treat this OOF result as exploratory, not a pre-registered estimate."
                             if refrozen_after_exposure else
                             "CPU OOF is the primary analysis; do not select a better-scoring device after comparison."),
        "environment": split_environment(),
        "ranker_profile": dict(MLP_CUDA_PROFILE if gpu else MLP_PROFILE),
        "baseline": "team v11 LightGBM, retrained separately on each outer training fold",
        "mlp": "same full v11 training candidate matrix/features; train-only scaler; query-listwise softmax; "
               + ("CUDA PyTorch" if gpu else "CPU PyTorch") + "; no pretrained weights",
        "outer_validation": "Each target_vessel_id group appears in exactly one validation fold and is excluded from that fold's training and lookup resources.",
        "inner_training_features": "Existing five-fold target-group cross-fit inside the outer training partition.",
        "acceptance_gates": {
            "mlp_only_correct_exceeds_baseline_only_correct_korean": True,
            "mlp_only_correct_exceeds_baseline_only_correct_non_korean": True,
            "two_sided_exact_mcnemar_holm_adjusted_p_below": 0.05,
            "holm_family": ["korean", "non_korean"],
            "official_weighted_oof_score_non_decrease": True,
            "candidate_recall_equal_to_baseline": True,
            "submission_promotion_from_this_oof": False,
            "transformer_followup_only_if_all_gates_pass": True,
        },
    })
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", action="store_true", help="별도 CUDA MLP 프로토콜을 고정합니다")
    parser.add_argument("--refrozen-after-exposure", action="store_true",
                        help="이미 보고된 fold 결과를 본 뒤 다시 고정하는 경우 메타데이터에 표시합니다")
    parser.add_argument("--compare-to", type=Path,
                        help="쓰기 없이 기존 고정 파일의 fold가 여기서 재계산한 fold와 같은지 대조합니다")
    parser.add_argument("--from-cpu-split", type=Path,
                        help="--gpu 전용: 사전 고정된 CPU 프로토콜의 fold를 그대로 옮겨 CUDA 프로토콜을 만듭니다")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    if args.compare_to is not None:
        print(json.dumps(compare_to(args.compare_to), ensure_ascii=False, indent=2))
        raise SystemExit(0)
    output = CUDA_SPLIT_PATH if args.gpu else SPLIT_PATH
    if args.from_cpu_split is not None:
        if not args.gpu or args.refrozen_after_exposure:
            parser.error("--from-cpu-split은 --gpu와만 함께 사용합니다")
        frozen = freeze_cuda_from_cpu(args.from_cpu_split, output)
        print(f"CPU fold 복사: {frozen['derived_from']['file']} ({frozen['derived_from']['sha256'][:16]}…)")
    else:
        frozen = freeze(output, gpu=args.gpu, refrozen_after_exposure=args.refrozen_after_exposure)
    print(f"고정 OOF 분할: {output}")
    print(f"행 {frozen['train_rows']}, target 그룹 {frozen['target_groups']}, fold {len(frozen['folds'])}")
    print(f"analysis_role {frozen['analysis_role']}, fold digest {fold_digest(frozen)}")
    print(f"환경 {frozen['environment']}")
