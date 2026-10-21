"""Build the language-split diagnostic ZIP from the deployed v11 submission.

Purpose: the official score is ``0.7*K + 0.3*N``. P0 and P1 both scored the same Public
value, so the single number cannot say which language loses the points. This package is
P0 with one change: rows whose ``vessel_name`` contains a Hangul syllable are predicted as
one fixed catalog id that is never a training target, so the Public score of this package
is ``0.3*N`` (plus whatever tiny amount the fixed id happens to get right).

No retraining and no local artifacts: ``model/lgb_ranker.txt``, ``model/match_lib.py``,
``model/resources.pkl`` and ``requirements.txt`` are copied byte for byte from the deployed
ZIP. ``script.py`` keeps the deployed text, including its CWD-relative data/output paths,
with exactly one step inserted after the predictions are final.

    .\\.venv_competition_py311\\Scripts\\python.exe scripts\\make_submit_v11_language_probe.py \\
        --out submissions\\team_v11_language_probe_korean_fixed.zip

This builds and verifies a package. It never submits anything.
"""

from __future__ import annotations

import argparse
import difflib
import hashlib
import json
import pickle
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ZIP = ROOT / "submissions" / "team_v11_merged.zip"
SOURCE_ZIP_SHA256 = "33dc878b5f4987011eb217f85527d960630225bd80c5d7c2ab60e64448420ab2"
METADATA = ROOT / "team_v11_artifacts" / "v11" / "training_metadata.json"
TRAIN_CSV = ROOT / "data" / "open" / "train.csv"
CATALOG_CSV = ROOT / "data" / "open" / "vessel_catalog.csv"
MEMBERS = ("model/lgb_ranker.txt", "model/match_lib.py", "model/resources.pkl",
           "script.py", "requirements.txt")
COPIED = ("model/lgb_ranker.txt", "model/match_lib.py", "model/resources.pkl",
          "requirements.txt")
PROBE_MEMBER = "model/probe_config.json"
FIXED_ID_RULE = ("lexicographically smallest vessel_id in data/open/vessel_catalog.csv "
                 "that is not any target_vessel_id in data/open/train.csv")

ANCHOR = '''    result = pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": predictions})
'''
INSERTED = '''    result = pd.DataFrame({"sample_id": test["sample_id"], "target_vessel_id": predictions})
    result = _apply_language_probe(result, test, index)
'''
HELPER = '''

def _is_korean(value) -> bool:
    """Same rule as the official metric: any Hangul syllable in the name."""
    return isinstance(value, str) and any("\\uac00" <= char <= "\\ud7a3" for char in value)


def _apply_language_probe(result, test, index):
    """Diagnostic step: Korean-named rows get one fixed catalog id.

    See model/probe_config.json. Nothing about the evaluation rows is printed.
    """
    with open(BASE / "model" / "probe_config.json", encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("mode") != "korean_fixed_wrong":
        raise ValueError("Unsupported probe mode")
    fixed = str(config["fixed_vessel_id"])
    if fixed not in index.id_to_index:
        raise ValueError("Fixed probe id is not in the catalog")
    korean = pd.Series([_is_korean(value) for value in test["vessel_name"]],
                       index=result.index)
    result.loc[korean, "target_vessel_id"] = fixed
    return result
'''
IMPORT_ANCHOR = '''from pathlib import Path
import pickle
import sys
'''
IMPORT_NEW = '''from pathlib import Path
import json
import pickle
import sys
'''
RUN_ANCHOR = '''

def run(
'''


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fixed_vessel_id() -> dict:
    """The diagnostic id, computed from the official data at build time."""
    catalog = pd.read_csv(CATALOG_CSV, encoding="utf-8-sig")
    train = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
    targets = set(train["target_vessel_id"].astype(str))
    ids = sorted(catalog["vessel_id"].astype(str))
    free = [value for value in ids if value not in targets]
    if not free:
        raise ValueError("train target이 아닌 catalog vessel_id가 없습니다")
    chosen = free[0]
    return {"fixed_vessel_id": chosen, "rule": FIXED_ID_RULE,
            "catalog_vessel_ids": len(ids), "train_target_ids": len(targets),
            "eligible_ids": len(free), "is_train_target": chosen in targets,
            "in_catalog": chosen in set(ids),
            "train_sha256": sha256_file(TRAIN_CSV),
            "catalog_sha256": sha256_file(CATALOG_CSV)}


def patch_script(text: str) -> str:
    """Insert exactly one call plus its helper; everything else keeps the deployed text."""
    if ANCHOR not in text:
        raise ValueError("배포본 script.py에서 예측 조립 지점을 찾지 못했습니다")
    if IMPORT_ANCHOR not in text or RUN_ANCHOR not in text:
        raise ValueError("배포본 script.py의 import 또는 run() 지점을 찾지 못했습니다")
    out = text.replace(IMPORT_ANCHOR, IMPORT_NEW, 1)
    out = out.replace(RUN_ANCHOR, HELPER + RUN_ANCHOR, 1)
    out = out.replace(ANCHOR, INSERTED, 1)
    return out


def script_diff(before: str, after: str) -> dict:
    lines = list(difflib.unified_diff(before.splitlines(), after.splitlines(),
                                      "deployed/script.py", "probe/script.py", lineterm=""))
    removed = [l for l in lines if l.startswith("-") and not l.startswith("---")]
    added = [l for l in lines if l.startswith("+") and not l.startswith("+++")]
    # The one replaced line is re-added verbatim, so it must appear on both sides.
    unexpected = [l for l in removed if "+" + l[1:] not in added]
    return {"added_lines": len(added), "removed_lines": len(removed),
            "removed_lines_not_re_added": unexpected, "diff": lines}


def build(out_zip: Path) -> dict:
    if not SOURCE_ZIP.is_file():
        raise FileNotFoundError(f"원본 제출 ZIP이 없습니다: {SOURCE_ZIP}")
    actual = sha256_file(SOURCE_ZIP)
    if actual != SOURCE_ZIP_SHA256:
        raise ValueError(f"원본 ZIP SHA-256이 다릅니다: 기대 {SOURCE_ZIP_SHA256}, 실제 {actual}")
    if out_zip.resolve() in {SOURCE_ZIP.resolve(),
                             (ROOT / "submissions" / "team_v11_override_off_P1.zip").resolve()}:
        raise ValueError("기존 제출 ZIP을 덮어쓸 수 없습니다")
    if out_zip.exists():
        raise FileExistsError(f"기존 ZIP은 덮어쓰지 않습니다: {out_zip}")
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    chosen = fixed_vessel_id()
    with ZipFile(SOURCE_ZIP) as archive:
        names = sorted(item.filename for item in archive.infolist())
        if names != sorted(MEMBERS):
            raise ValueError(f"원본 ZIP 구성이 다릅니다: {names}")
        with tempfile.TemporaryDirectory() as directory:
            staged = Path(directory) / "pkg"
            archive.extractall(staged)
            if sha256_file(staged / "model" / "lgb_ranker.txt") != metadata["model_sha256"]:
                raise ValueError("원본 모델 해시가 학습 기록과 다릅니다")
            if sha256_file(staged / "model" / "resources.pkl") != metadata["resources_sha256"]:
                raise ValueError("원본 resources.pkl 해시가 학습 기록과 다릅니다")
            with (staged / "model" / "resources.pkl").open("rb") as handle:
                res = pickle.load(handle)
            deployed_script = (staged / "script.py").read_text(encoding="utf-8")
            patched = patch_script(deployed_script)
            diff = script_diff(deployed_script, patched)
            if diff["removed_lines_not_re_added"]:
                raise ValueError("script.py에서 삭제된 줄이 있습니다: "
                                 + str(diff["removed_lines_not_re_added"]))
            (staged / "script.py").write_text(patched, encoding="utf-8", newline="")
            config = {"mode": "korean_fixed_wrong", "fixed_vessel_id": chosen["fixed_vessel_id"],
                      "rule": FIXED_ID_RULE,
                      "korean_rule": "vessel_name contains at least one character in [가-힣]; "
                                     "a missing or non-string name counts as non-Korean",
                      "purpose": "split the Public score into Korean K and non-Korean N",
                      "formulas": {"N": "S_d / 0.3", "K": "(0.8882950791 - 0.3*N) / 0.7"},
                      "not_a_final_candidate": True, "retrained": False,
                      "selection": {k: chosen[k] for k in
                                    ("catalog_vessel_ids", "train_target_ids", "eligible_ids",
                                     "is_train_target", "in_catalog")}}
            (staged / PROBE_MEMBER).write_text(
                json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
            copied = {name: sha256_file(staged / name) for name in COPIED}
            out_zip.parent.mkdir(parents=True, exist_ok=True)
            with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as new:
                for name in MEMBERS + (PROBE_MEMBER,):
                    new.write(staged / name, name)
            result = {
                "source_zip": SOURCE_ZIP.relative_to(ROOT).as_posix(),
                "source_zip_sha256": actual,
                "fixed_vessel_id": chosen["fixed_vessel_id"], "fixed_id_rule": FIXED_ID_RULE,
                "fixed_id_selection": {k: chosen[k] for k in
                                       ("catalog_vessel_ids", "train_target_ids", "eligible_ids",
                                        "is_train_target", "in_catalog")},
                "copied_unchanged_sha256": copied,
                "resources_flags": {"use_callsign": res["use_callsign"],
                                    "use_name_lookup": res["use_name_lookup"]},
                "script_diff": {k: v for k, v in diff.items() if k != "diff"},
                "script_diff_text": diff["diff"],
                "probe_config": config,
                "out_zip": out_zip.relative_to(ROOT).as_posix() if out_zip.is_relative_to(ROOT)
                           else out_zip.as_posix(),
                "out_zip_sha256": sha256_file(out_zip),
                "out_zip_bytes": out_zip.stat().st_size,
                "retrained": False, "leaderboard_submission": False,
            }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path,
                        default=ROOT / "submissions" / "team_v11_language_probe_korean_fixed.zip")
    parser.add_argument("--show-diff", action="store_true", help="script.py 차이를 그대로 출력")
    args = parser.parse_args()
    report = build(args.out.resolve())
    if args.show_diff:
        print("\n".join(report["script_diff_text"]))
        print()
    print(json.dumps({k: v for k, v in report.items() if k != "script_diff_text"},
                     ensure_ascii=False, indent=2))
