"""Keep frozen CatBoost artifacts and restore only EXP-022 non-Korean search."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SUBMISSIONS = ROOT / "submissions"
CATBOOST = SUBMISSIONS / "catboost_ngram_negatives.zip"
LAMBDA = SUBMISSIONS / "candidate_lambdarank.zip"
EXP022 = SUBMISSIONS / "final_exp022.zip"
OUTPUT = SUBMISSIONS / "candidate_catboost_ngram_exp022_non_korean.zip"
REPORT = ROOT / "experiments/results/agent_catboost_exp022_non_korean_package_20261018/package_verification.json"
PYTHON = ROOT / ".venv_competition_py311/Scripts/python.exe"
EXPECTED = {
    CATBOOST: "b5f3226be731104f5338b28951c2860f729137da922234becdef05cc4c272226",
    LAMBDA: "39cd14a82f06a3441981b71678baf677a1f384bac8c8cca4a533fd90446d0362",
    EXP022: "fd7833eba4f0c417c226ca8054a828b3f1963195cd64331a63267f761d874255",
}
ADAPTIVE = "model/adaptive_lib.py"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_sources() -> None:
    if sys.version_info[:2] != (3, 11):
        raise AssertionError("Package verification requires Python 3.11")
    for path, digest in EXPECTED.items():
        if sha256(path) != digest:
            raise AssertionError(f"Frozen ZIP changed: {path.name}")


def create() -> None:
    check_sources()
    if OUTPUT.exists() or REPORT.exists():
        raise FileExistsError("New candidate or verification report already exists")
    with zipfile.ZipFile(CATBOOST) as current, zipfile.ZipFile(EXP022) as original:
        if current.testzip() is not None or original.testzip() is not None:
            raise AssertionError("Source archive CRC failure")
        if current.read(ADAPTIVE) == original.read(ADAPTIVE):
            raise AssertionError("Expected a source-code difference")
        with zipfile.ZipFile(OUTPUT, "x", compression=zipfile.ZIP_DEFLATED) as candidate:
            for name in current.namelist():
                candidate.writestr(name, original.read(name) if name == ADAPTIVE else current.read(name))
    print(json.dumps({"archive": str(OUTPUT), "sha256": sha256(OUTPUT)}, ensure_ascii=False))


def run_archive(root: Path, frame: pd.DataFrame):
    frame.to_csv(root / "data/test.csv", index=False, encoding="utf-8-sig")
    env = os.environ.copy()
    env["HF_HUB_OFFLINE"] = "1"
    env["TRANSFORMERS_OFFLINE"] = "1"
    start = time.perf_counter()
    completed = subprocess.run(
        [str(PYTHON), str(root / "script.py")], cwd=root, env=env,
        capture_output=True, text=True, timeout=600, check=False,
    )
    elapsed = time.perf_counter() - start
    if completed.returncode:
        raise RuntimeError(completed.stderr[-3000:])
    output = pd.read_csv(root / "output/submission.csv", encoding="utf-8-sig")
    if output.columns.tolist() != ["sample_id", "target_vessel_id"]:
        raise AssertionError("Unexpected submission columns")
    if len(output) != len(frame) or output["sample_id"].astype(str).tolist() != frame["sample_id"].astype(str).tolist():
        raise AssertionError("Submission row count or order changed")
    if output["target_vessel_id"].isna().any():
        raise AssertionError("Missing prediction")
    return output, elapsed


def verify() -> None:
    check_sources()
    if REPORT.exists():
        raise FileExistsError("Verification report already exists")
    with zipfile.ZipFile(CATBOOST) as old, zipfile.ZipFile(EXP022) as exp022, zipfile.ZipFile(OUTPUT) as new:
        if set(new.namelist()) != set(old.namelist()) or len(new.namelist()) != len(set(new.namelist())):
            raise AssertionError("Archive members changed")
        if new.testzip() is not None:
            raise AssertionError("Archive CRC failure")
        changed = [name for name in old.namelist() if new.read(name) != old.read(name)]
        if changed != [ADAPTIVE] or new.read(ADAPTIVE) != exp022.read(ADAPTIVE):
            raise AssertionError("More than EXP-022 adaptive_lib changed")
        member_hashes = {name: hashlib.sha256(new.read(name)).hexdigest() for name in new.namelist()}

    train = pd.read_csv(ROOT / "data/open/train.csv", encoding="utf-8-sig")
    catalog = pd.read_csv(ROOT / "data/open/vessel_catalog.csv", encoding="utf-8-sig")
    rows = train.head(3000).drop(columns=["target_vessel_id"])
    korean = rows["vessel_name"].astype(str).str.contains("[가-힣]", regex=True)
    if korean.sum() == 0 or (~korean).sum() == 0 or not rows["sample_id"].is_unique:
        raise AssertionError("Functional sample must contain both name groups")
    with tempfile.TemporaryDirectory(prefix="catboost_exp022_compare_") as directory:
        root = Path(directory)
        old_root, new_root = root / "old", root / "new"
        for archive_path, run_root in ((CATBOOST, old_root), (OUTPUT, new_root)):
            run_root.mkdir()
            with zipfile.ZipFile(archive_path) as archive:
                archive.extractall(run_root)
            (run_root / "data").mkdir()
            catalog.to_csv(run_root / "data/vessel_catalog.csv", index=False, encoding="utf-8-sig")
        old_output, old_seconds = run_archive(old_root, rows)
        new_output, first_seconds = run_archive(new_root, rows)
        repeat_output, second_seconds = run_archive(new_root, rows)
        if not new_output.equals(repeat_output):
            raise AssertionError("Repeated new-package output differs")
        old_pred = old_output["target_vessel_id"].astype(str)
        new_pred = new_output["target_vessel_id"].astype(str)
        if not old_pred[korean.to_numpy()].equals(new_pred[korean.to_numpy()]):
            raise AssertionError("Korean predictions changed")
        non_korean_changes = int((old_pred[~korean.to_numpy()] != new_pred[~korean.to_numpy()]).sum())
        if non_korean_changes == 0:
            raise AssertionError("Restored non-Korean search did not change any diagnostic prediction")
        expected = dict(zip(rows["sample_id"].astype(str), new_pred))
        mixed = pd.concat([rows[korean].head(5), rows[~korean].head(5)])
        shuffled, _ = run_archive(new_root, mixed.sample(frac=1, random_state=31438))
        if any(expected[str(row.sample_id)] != str(row.target_vessel_id) for row in shuffled.itertuples(index=False)):
            raise AssertionError("Shuffled subset changed a prediction")
        for single in (rows[korean].head(1), rows[~korean].head(1)):
            predicted, _ = run_archive(new_root, single)
            if expected[str(single.iloc[0]["sample_id"])] != str(predicted.iloc[0]["target_vessel_id"]):
                raise AssertionError("Single-row prediction changed")
    if not set(new_pred).issubset(set(catalog["vessel_id"].astype(str))):
        raise AssertionError("Prediction absent from official catalog")
    if max(first_seconds, second_seconds) > 600:
        raise AssertionError("3000-row runtime exceeds 600 seconds")
    report = {
        "new_archive": str(OUTPUT.relative_to(ROOT)), "new_sha256": sha256(OUTPUT),
        "source_catboost_sha256": sha256(CATBOOST), "source_exp022_sha256": sha256(EXP022),
        "only_changed_member": ADAPTIVE, "member_sha256": member_hashes,
        "functional_rows": len(rows), "korean_rows": int(korean.sum()),
        "korean_predictions_unchanged": True,
        "non_korean_rows": int((~korean).sum()),
        "non_korean_predictions_changed": non_korean_changes,
        "repeatable": True, "shuffled_and_single_row_independent": True,
        "all_predictions_in_official_catalog": True,
        "old_3000_row_seconds": old_seconds,
        "new_3000_row_seconds": [first_seconds, second_seconds],
        "leaderboard_submitted": False,
        "performance_estimate": False,
        "historical_group_exposure": True,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if len(sys.argv) != 2 or sys.argv[1] not in ("create", "verify"):
        raise SystemExit("Usage: python scripts/package_catboost_exp022_non_korean.py create|verify")
    create() if sys.argv[1] == "create" else verify()
