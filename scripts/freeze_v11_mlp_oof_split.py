"""Freeze the MLP OOF split and decision protocol before model evaluation."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.mlp_oof_protocol import PROTOCOL_VERSION, build_group_oof_split  # noqa: E402
from agent.retrain_runtime import MLP_PROFILE  # noqa: E402
from src.metric import is_korean  # noqa: E402


TRAIN_PATH = ROOT / "data" / "open" / "train.csv"
CATALOG_PATH = ROOT / "data" / "open" / "vessel_catalog.csv"
SPLIT_PATH = ROOT / "experiments" / "splits" / "agent_v11_mlp_listwise_group5_seed20261019.json"
def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def freeze(output: Path = SPLIT_PATH) -> dict:
    if output.exists():
        raise FileExistsError(f"기존 split/protocol 파일은 덮어쓰지 않습니다: {output}")
    train = pd.read_csv(TRAIN_PATH, encoding="utf-8-sig")
    catalog_hash = sha256_file(CATALOG_PATH)
    train_hash = sha256_file(TRAIN_PATH)
    train["is_korean"] = train["vessel_name"].map(is_korean).astype(int)
    result = build_group_oof_split(train, seed=20261019)
    result.update({
        "protocol_version": PROTOCOL_VERSION,
        "split_method": "target_vessel_id_group_oof",
        "train_rows": int(len(train)),
        "target_groups": int(train["target_vessel_id"].nunique()),
        "data_sources": ["data/open/train.csv", "data/open/vessel_catalog.csv"],
        "train_sha256": train_hash,
        "catalog_sha256": catalog_hash,
        "historical_validation_overlap": True,
        "historical_error_analysis_overlap": True,
        "independence_note": "All official target groups were previously exposed to validation or error analysis; this new OOF partition remains exploratory, not an independent performance estimate.",
        "ranker_profile": dict(MLP_PROFILE),
        "baseline": "team v11 LightGBM, retrained separately on each outer training fold",
        "mlp": "same full v11 training candidate matrix/features; train-only scaler; query-listwise softmax; CPU PyTorch; no pretrained weights",
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
    frozen = freeze()
    print(f"고정 OOF 분할: {SPLIT_PATH}")
    print(f"행 {frozen['train_rows']}, target 그룹 {frozen['target_groups']}, fold {len(frozen['folds'])}")
