"""Row-independence check for a built submission package (CLAUDE.md section 5 rule 2).

Each evaluation row's prediction must depend only on that row's input plus the official
data. This drives the package's real entry point, ``script.py``, as a subprocess several
times and requires the same ``sample_id`` to get the same ``target_vessel_id`` in every
arrangement:

* the full input,
* a subset holding only the picked rows, in order,
* the same subset reversed,
* each picked row on its own,
* the full input shuffled (optional, with ``--shuffle``; costs a second full pass).

    .\\.venv_competition_py311\\Scripts\\python.exe scripts\\check_row_independence.py ^
        --package <extracted package dir> --test <test-format csv> --rows 8

The package directory must already contain ``data/vessel_catalog.csv``. Nothing here
reads labels; the input must not carry ``target_vessel_id``.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
LABEL_COLUMN = "target_vessel_id"
DEFAULT_PYTHON = ROOT / ".venv_competition_py311" / "Scripts" / "python.exe"


def is_korean(value: object) -> bool:
    return isinstance(value, str) and any("가" <= c <= "힣" for c in value)


def run_package(python: Path, package: Path, frame: pd.DataFrame, tag: str) -> pd.DataFrame:
    """Run script.py once on ``frame`` in an isolated copy of the package's data/output."""
    with tempfile.TemporaryDirectory() as directory:
        work = Path(directory) / "pkg"
        shutil.copytree(package, work, ignore=shutil.ignore_patterns("output", "data"))
        (work / "data").mkdir(parents=True)
        shutil.copy(package / "data" / "vessel_catalog.csv", work / "data" / "vessel_catalog.csv")
        frame.to_csv(work / "data" / "test.csv", index=False, encoding="utf-8")
        env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1",
                   HTTP_PROXY="http://127.0.0.1:9", HTTPS_PROXY="http://127.0.0.1:9",
                   ALL_PROXY="http://127.0.0.1:9")
        env.pop("OPENAI_API_KEY", None)
        started = time.monotonic()
        result = subprocess.run([str(python), "-I", "script.py"], cwd=work, env=env,
                                capture_output=True, text=True, encoding="utf-8", check=False)
        if result.returncode:
            raise RuntimeError(f"{tag}: script.py 실패(exit {result.returncode})\n"
                               f"{result.stdout}\n{result.stderr}")
        out = pd.read_csv(work / "output" / "submission.csv", encoding="utf-8-sig")
        print(f"  {tag}: {len(out)}행 {time.monotonic() - started:.1f}s", flush=True)
        return out


def pick_rows(frame: pd.DataFrame, count: int) -> list[int]:
    """Deterministic positions covering both languages and both ends of the frame."""
    korean = [i for i, value in enumerate(frame["vessel_name"]) if is_korean(value)]
    other = [i for i, value in enumerate(frame["vessel_name"]) if not is_korean(value)]
    half = max(1, count // 2)
    picked = korean[:half] + other[:half]
    for candidate in (0, len(frame) // 2, len(frame) - 1):
        if len(set(picked)) >= count:
            break
        picked.append(candidate)
    return sorted(set(picked))[:count]


def check(package: Path, test_csv: Path, rows: int, python: Path, shuffle: bool) -> dict:
    frame = pd.read_csv(test_csv, encoding="utf-8-sig")
    if LABEL_COLUMN in frame.columns:
        raise ValueError(f"입력에 정답 열이 있습니다: {LABEL_COLUMN}")
    if not (package / "script.py").is_file():
        raise FileNotFoundError(f"script.py가 없습니다: {package}")
    if not (package / "data" / "vessel_catalog.csv").is_file():
        raise FileNotFoundError("패키지 data/vessel_catalog.csv가 필요합니다")
    picked = pick_rows(frame, rows)
    languages = sorted({is_korean(frame.iloc[i]["vessel_name"]) for i in picked})
    print(f"검사 행 {len(picked)}개 (한글·비한글 포함: {languages == [False, True]})", flush=True)

    print("기준: 전체 입력", flush=True)
    full = run_package(python, package, frame, "full")
    reference = dict(zip(full["sample_id"].astype(str), full["target_vessel_id"].astype(str)))

    arrangements: dict[str, dict[str, str]] = {}
    subset = frame.iloc[picked]
    print("부분집합·역순·단독 실행", flush=True)
    for tag, piece in (("subset", subset), ("subset_reversed", subset.iloc[::-1])):
        out = run_package(python, package, piece, tag)
        arrangements[tag] = dict(zip(out["sample_id"].astype(str), out["target_vessel_id"].astype(str)))
    singles: dict[str, str] = {}
    for position in picked:
        one = frame.iloc[[position]]
        out = run_package(python, package, one, f"single_{one.iloc[0]['sample_id']}")
        singles.update(zip(out["sample_id"].astype(str), out["target_vessel_id"].astype(str)))
    arrangements["single_row"] = singles
    if shuffle:
        print("전체 입력 셔플", flush=True)
        shuffled = frame.sample(frac=1.0, random_state=20261019).reset_index(drop=True)
        out = run_package(python, package, shuffled, "shuffled_full")
        arrangements["shuffled_full"] = dict(zip(out["sample_id"].astype(str),
                                                 out["target_vessel_id"].astype(str)))

    mismatches = {}
    for tag, predictions in arrangements.items():
        bad = sorted(key for key, value in predictions.items() if reference.get(key) != value)
        if bad:
            mismatches[tag] = {"count": len(bad), "checked": len(predictions)}
    sample_ids = [str(frame.iloc[i]["sample_id"]) for i in picked]
    return {
        "package": package.as_posix(), "test_csv": test_csv.as_posix(),
        "input_rows": len(frame), "checked_rows": len(picked),
        "both_languages": languages == [False, True],
        "arrangements": sorted(arrangements),
        "compared_predictions": {tag: len(v) for tag, v in arrangements.items()},
        "mismatches": mismatches, "passed": not mismatches,
        "checked_sample_ids_are_local_only": True, "checked_count": len(sample_ids),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--package", type=Path, required=True, help="해제된 제출 패키지 폴더")
    parser.add_argument("--test", type=Path, required=True, help="test 형식 입력 CSV (정답 열 없음)")
    parser.add_argument("--rows", type=int, default=8, help="부분집합·단독으로 확인할 행 수")
    parser.add_argument("--python", type=Path, default=DEFAULT_PYTHON)
    parser.add_argument("--shuffle", action="store_true", help="전체 입력 셔플 비교도 수행(1회 추가 전체 실행)")
    parser.add_argument("--output", type=Path, help="결과 JSON 경로")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    report = check(args.package.resolve(), args.test.resolve(), args.rows,
                   args.python.resolve(), args.shuffle)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["passed"] else 1)
