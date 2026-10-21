"""Compare v11 Hangul candidate recall with its catalog vocabulary enabled.

This is a retrieval probe, not an independent performance estimate or a
submission builder. Validation vessel IDs are excluded from resource fitting.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
import pandas as pd
from rapidfuzz import fuzz, process
from sklearn.model_selection import StratifiedGroupKFold


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.team_v11.source import match_lib as ml  # noqa: E402


SEED = 42
OUTPUT = ROOT / "experiments" / "results" / "team_v11_hangul_candidates_20261019" / "summary.json"


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    train_path = ROOT / "data" / "open" / "train.csv"
    catalog_path = ROOT / "data" / "open" / "vessel_catalog.csv"
    train = pd.read_csv(train_path, encoding="utf-8-sig")
    catalog = pd.read_csv(catalog_path, encoding="utf-8-sig")
    index = ml.CatalogIndex(catalog)
    hangul = train["vessel_name"].map(ml.is_hangul).to_numpy(bool)
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)

    totals = {
        "hangul_rows": 0,
        "baseline_hits": 0,
        "vocab_hits": 0,
        "newly_recovered": 0,
        "lost_hits": 0,
        "baseline_candidates": 0,
        "vocab_candidates": 0,
        "baseline_seconds": 0.0,
        "vocab_seconds": 0.0,
        "unique_name_pairs": 0,
        "base_name_top1": 0,
        "vocab_name_top1": 0,
        "either_name_top1": 0,
    }
    folds = []
    original_setting = ml.USE_VOCAB
    try:
        for fold, (fit_idx, valid_idx) in enumerate(
            splitter.split(train, hangul, train["target_vessel_id"])
        ):
            fit = train.iloc[fit_idx]
            valid = train.iloc[valid_idx]
            valid = valid.loc[valid["vessel_name"].map(ml.is_hangul)]
            assert not set(fit["target_vessel_id"]) & set(valid["target_vessel_id"])
            resources = ml.build_resources(fit, catalog)
            resources["token_dict"] = {}
            resources["whole_dict"] = {}
            truth = [index.id_to_index[value] for value in valid["target_vessel_id"]]
            fold_result = {"fold": fold, "hangul_rows": len(valid)}
            hit_by_setting = {}
            queries_by_setting = {}
            for setting, enabled in (("baseline", False), ("vocab", True)):
                ml.USE_VOCAB = enabled
                start = time.monotonic()
                queries = [ml.prepare_query(row, resources) for row in valid.to_dict("records")]
                queries_by_setting[setting] = queries
                candidates, _ = ml.candidates_and_features(queries, index, resources)
                elapsed = time.monotonic() - start
                hits = np.array(
                    [target in candidate for target, candidate in zip(truth, candidates)],
                    dtype=bool,
                )
                hit_by_setting[setting] = hits
                count = sum(len(candidate) for candidate in candidates)
                fold_result[f"{setting}_hits"] = int(hits.sum())
                fold_result[f"{setting}_mean_candidates"] = round(count / len(valid), 2)
                fold_result[f"{setting}_seconds"] = round(elapsed, 2)
                totals[f"{setting}_hits"] += int(hits.sum())
                totals[f"{setting}_candidates"] += count
                totals[f"{setting}_seconds"] += elapsed
            base = hit_by_setting["baseline"]
            vocab = hit_by_setting["vocab"]
            recovered = int((~base & vocab).sum())
            lost = int((base & ~vocab).sum())
            fold_result["newly_recovered"] = recovered
            fold_result["lost_hits"] = lost
            totals["newly_recovered"] += recovered
            totals["lost_hits"] += lost
            totals["hangul_rows"] += len(valid)

            # The ranker is intentionally excluded. This measures whether
            # name forms alone put the truth at a tie-favorable top rank.
            unique_positions = valid.reset_index(drop=True).drop_duplicates(
                ["vessel_name", "target_vessel_id"]
            ).index
            base_top1 = vocab_top1 = either_top1 = 0
            for position in unique_positions:
                base_query = queries_by_setting["baseline"][position]
                vocab_query = queries_by_setting["vocab"][position]
                forms = list(dict.fromkeys([
                    base_query["compact"], base_query["rom_compact"],
                    *base_query["variants"], *base_query["learned"],
                ]))
                base_scores = process.cdist(
                    forms, index.compact, scorer=fuzz.ratio, dtype=np.uint8, workers=-1
                ).max(axis=0)
                target = truth[position]
                base_win = bool(base_scores[target] == base_scores.max())
                vocab_forms = vocab_query["vocab_forms"]
                vocab_win = False
                if vocab_forms:
                    vocab_scores = process.cdist(
                        vocab_forms, index.compact, scorer=fuzz.ratio,
                        dtype=np.uint8, workers=-1,
                    ).max(axis=0)
                    vocab_win = bool(vocab_scores[target] == vocab_scores.max())
                base_top1 += base_win
                vocab_top1 += vocab_win
                either_top1 += base_win or vocab_win
            fold_result["unique_name_pairs"] = len(unique_positions)
            fold_result["base_name_top1"] = base_top1
            fold_result["vocab_name_top1"] = vocab_top1
            fold_result["either_name_top1"] = either_top1
            totals["unique_name_pairs"] += len(unique_positions)
            totals["base_name_top1"] += base_top1
            totals["vocab_name_top1"] += vocab_top1
            totals["either_name_top1"] += either_top1
            folds.append(fold_result)
            print(json.dumps(fold_result, ensure_ascii=False), flush=True)
    finally:
        ml.USE_VOCAB = original_setting

    n = totals["hangul_rows"]
    assert n == int(hangul.sum())
    report = {
        "purpose": "Hangul candidate retrieval and name-only ranking: existing v11 versus USE_VOCAB=True",
        "method": "5-fold StratifiedGroupKFold by target_vessel_id; fold-fit resources; no validation labels passed to retrieval",
        "limitations": "All existing target groups were exposed in previous experiments; exploratory only. No ranker was retrained, and this is not a leaderboard or accuracy estimate.",
        "seed": SEED,
        "train_sha256": file_hash(train_path),
        "catalog_sha256": file_hash(catalog_path),
        "source_match_lib_sha256": file_hash(Path(ml.__file__)),
        "folds": folds,
        "totals": {
            **{key: value for key, value in totals.items() if not key.endswith("_seconds")},
            "baseline_seconds": round(totals["baseline_seconds"], 2),
            "vocab_seconds": round(totals["vocab_seconds"], 2),
            "baseline_recall": round(totals["baseline_hits"] / n, 6),
            "vocab_recall": round(totals["vocab_hits"] / n, 6),
            "baseline_mean_candidates": round(totals["baseline_candidates"] / n, 2),
            "vocab_mean_candidates": round(totals["vocab_candidates"] / n, 2),
            "base_name_top1_rate": round(totals["base_name_top1"] / totals["unique_name_pairs"], 6),
            "vocab_name_top1_rate": round(totals["vocab_name_top1"] / totals["unique_name_pairs"], 6),
            "either_name_top1_rate": round(totals["either_name_top1"] / totals["unique_name_pairs"], 6),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print("summary:", OUTPUT, flush=True)
    print(json.dumps(report["totals"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
