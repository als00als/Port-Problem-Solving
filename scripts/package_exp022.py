"""Build and exercise the EXP-022 candidate from the frozen EXP-014 ZIP."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
import time
import types
import zipfile
from pathlib import Path

import pandas as pd

from scripts.exp022_non_korean_fuzzy_gate import (
    BASELINE_ZIP,
    CATALOG,
    GATES,
    RESULT,
    ROOT,
    SEED,
    SPLIT,
    TRAIN,
    evaluate,
    load_baseline_module,
    sha256,
)


OUTPUT_ZIP = ROOT / "submissions" / "final_exp022_non_korean_fuzzy.zip"
PACKAGE_REPORT = RESULT / "package_verification.json"
EXPECTED_MEMBERS = {"model/adaptive_lib.py", "script.py", "requirements.txt"}


def build_variant_library() -> bytes:
    with zipfile.ZipFile(BASELINE_ZIP) as archive:
        source = archive.read("model/adaptive_lib.py").decode("utf-8")
    source = source.replace("\r\n", "\n").replace("\r", "\n")
    start_marker = "    def candidates_with_scores(self, row: dict) -> tuple[list[int], dict[int, float]]:\n"
    end_marker = "    def predict_row(self, row: dict) -> str:\n"
    if source.count(start_marker) != 1 or source.count(end_marker) != 1:
        raise AssertionError("EXP-014 method markers changed")
    start = source.index(start_marker)
    end = source.index(end_marker, start)
    method = '''    def candidates_with_scores(self, row: dict) -> tuple[list[int], dict[int, float]]:
        """EXP-022: search Latin names fuzzily while preserving Korean ranking."""
        max_candidates = MAX_CANDIDATES
        base_indices = self.generate_base_candidates(row, max_candidates)
        korean_name = is_korean(row.get("vessel_name"))
        query_key = (
            romanized_name(row.get("vessel_name"))
            if korean_name
            else normalize_name(row.get("vessel_name"))
        )
        fuzzy_indices: list[int] = []
        fuzzy_scores: dict[int, float] = {}
        if query_key:
            matches = process.extract(
                query_key,
                self.romanized_names,
                scorer=fuzz.ratio,
                limit=FUZZY_LIMIT,
                score_cutoff=FUZZY_CUTOFF,
            )
            for _, score, index in matches:
                index = int(index)
                fuzzy_indices.append(index)
                if korean_name:
                    fuzzy_scores[index] = float(score) / 100.0

        merged: list[int] = []
        seen: set[int] = set()
        for index in fuzzy_indices + base_indices:
            index = int(index)
            if index not in seen:
                merged.append(index)
                seen.add(index)
            if len(merged) >= max_candidates:
                break
        return merged, fuzzy_scores

'''
    result = source[:start] + method + source[end:]
    compile(result, "model/adaptive_lib.py", "exec")
    return result.encode("utf-8")


def create() -> None:
    summary = json.loads((RESULT / "summary.json").read_text(encoding="utf-8"))
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    if not summary["gate_passed"] or not all(summary["gate_checks"].values()):
        raise AssertionError("EXP-022 acceptance gates did not pass")
    if split["acceptance_gates"] != GATES or split["baseline_archive_sha256"] != sha256(BASELINE_ZIP):
        raise AssertionError("Pre-registered experiment inputs changed")
    if OUTPUT_ZIP.exists():
        raise FileExistsError(f"Refusing to overwrite: {OUTPUT_ZIP}")
    model = build_variant_library()
    with zipfile.ZipFile(BASELINE_ZIP) as baseline:
        if set(baseline.namelist()) != EXPECTED_MEMBERS:
            raise AssertionError("EXP-014 archive structure changed")
        script = baseline.read("script.py").replace(b"EXP-014", b"EXP-022")
        requirements = baseline.read("requirements.txt")
    with zipfile.ZipFile(OUTPUT_ZIP, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("model/adaptive_lib.py", model)
        archive.writestr("script.py", script)
        archive.writestr("requirements.txt", requirements)
    print(json.dumps({"archive": str(OUTPUT_ZIP), "sha256": sha256(OUTPUT_ZIP), "model_sha256": hashlib.sha256(model).hexdigest()}, indent=2))


def load_variant_module() -> types.ModuleType:
    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        source = archive.read("model/adaptive_lib.py").decode("utf-8")
    module = types.ModuleType("exp022_packaged_adaptive_lib")
    exec(compile(source, str(OUTPUT_ZIP), "exec"), module.__dict__)
    return module


def run_archive(root: Path, rows: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    rows.to_csv(root / "data" / "test.csv", index=False, encoding="utf-8-sig")
    started = time.perf_counter()
    completed = subprocess.run(
        [sys.executable, str(root / "script.py")],
        cwd=root,
        text=True,
        capture_output=True,
        timeout=600,
        check=False,
    )
    elapsed = time.perf_counter() - started
    if completed.returncode:
        raise RuntimeError(f"Packaged script failed: {completed.stderr[-1500:]}")
    output = pd.read_csv(root / "output" / "submission.csv", encoding="utf-8-sig")
    if output.columns.tolist() != ["sample_id", "target_vessel_id"]:
        raise AssertionError("Submission columns changed")
    if len(output) != len(rows) or output["sample_id"].astype(str).tolist() != rows["sample_id"].astype(str).tolist():
        raise AssertionError("Submission row count or order changed")
    if output["target_vessel_id"].isna().any():
        raise AssertionError("Submission has missing predictions")
    return output, elapsed


def verify() -> None:
    if PACKAGE_REPORT.exists():
        raise FileExistsError(f"Verification already recorded: {PACKAGE_REPORT}")
    if not OUTPUT_ZIP.exists():
        raise FileNotFoundError(OUTPUT_ZIP)
    with zipfile.ZipFile(OUTPUT_ZIP) as archive:
        if set(archive.namelist()) != EXPECTED_MEMBERS or archive.testzip() is not None:
            raise AssertionError("Archive structure or integrity failed")
        if archive.read("model/adaptive_lib.py") != build_variant_library():
            raise AssertionError("Packaged model differs from frozen candidate source")
        if b"http" in archive.read("script.py").lower():
            raise AssertionError("Submission entry point unexpectedly references a URL")
    split = json.loads(SPLIT.read_text(encoding="utf-8"))
    train = pd.read_csv(TRAIN, encoding="utf-8-sig")
    catalog = pd.read_csv(CATALOG, encoding="utf-8-sig")
    valid_ids = set(split["valid_sample_ids"])
    valid = train[train["sample_id"].astype(str).isin(valid_ids)]
    extra = train[~train["sample_id"].astype(str).isin(valid_ids)].head(3000 - len(valid))
    input_rows = pd.concat([valid, extra], ignore_index=True).drop(columns=["target_vessel_id"])
    if len(input_rows) != 3000:
        raise AssertionError("Expected exactly 3,000 simulated input rows")
    with tempfile.TemporaryDirectory(prefix="exp022_package_") as directory:
        root = Path(directory)
        with zipfile.ZipFile(OUTPUT_ZIP) as archive:
            archive.extractall(root)
        (root / "data").mkdir()
        catalog.to_csv(root / "data" / "vessel_catalog.csv", index=False, encoding="utf-8-sig")
        first, first_seconds = run_archive(root, input_rows)
        second, second_seconds = run_archive(root, input_rows)
        if not first.equals(second):
            raise AssertionError("The same 3,000 input rows produced different predictions")
        subset = input_rows.iloc[:10].sample(frac=1.0, random_state=SEED).reset_index(drop=True)
        shuffled, _ = run_archive(root, subset)
        expected = dict(zip(first["sample_id"].astype(str), first["target_vessel_id"].astype(str)))
        if any(expected[str(sid)] != str(pred) for sid, pred in shuffled.itertuples(index=False, name=None)):
            raise AssertionError("Shuffled subset prediction changed")
        single, _ = run_archive(root, input_rows.iloc[[0]])
        if str(single.iloc[0]["target_vessel_id"]) != expected[str(input_rows.iloc[0]["sample_id"])]:
            raise AssertionError("Single-row prediction changed")

    if not set(first["target_vessel_id"].astype(str)).issubset(set(catalog["vessel_id"].astype(str))):
        raise AssertionError("Output contains ID absent from official catalog")
    # Compare packaged results with the independently evaluated EXP-022 rule.
    baseline_module = load_baseline_module()
    baseline_index = baseline_module.AdaptiveCatalogIndex(catalog)
    probe = evaluate(valid.to_dict("records"), baseline_index, baseline_module, improved=True)
    if first["target_vessel_id"].astype(str).iloc[: len(valid)].tolist() != probe["predictions"]:
        raise AssertionError("Packaged predictions differ from the EXP-022 evaluation rule")
    reported = json.loads((RESULT / "summary.json").read_text(encoding="utf-8"))
    if abs(probe["metrics"]["weighted"] - reported["variant"]["weighted"]) > 1e-12:
        raise AssertionError("Recomputed packaged weighted score differs from experiment")
    gate_runtime = max(first_seconds, second_seconds) <= GATES["maximum_projected_3000_seconds"]
    report = {
        "archive": str(OUTPUT_ZIP.relative_to(ROOT)),
        "archive_sha256": sha256(OUTPUT_ZIP),
        "archive_members": sorted(EXPECTED_MEMBERS),
        "input_rows": len(input_rows),
        "output_rows": len(first),
        "output_columns": first.columns.tolist(),
        "all_ids_in_catalog": True,
        "same_input_repeatable": True,
        "shuffled_subset_and_single_row_independent": True,
        "packaged_predictions_match_experiment": True,
        "run_seconds": [first_seconds, second_seconds],
        "runtime_gate_passed": gate_runtime,
    }
    PACKAGE_REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    reported["package_created"] = True
    reported["package_verified"] = gate_runtime
    (RESULT / "summary.json").write_text(json.dumps(reported, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not gate_runtime:
        raise AssertionError("Packaged 3,000-row runtime exceeded pre-registered limit")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=("create", "verify"))
    args = parser.parse_args()
    create() if args.mode == "create" else verify()
