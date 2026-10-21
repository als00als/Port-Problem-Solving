"""Build an override-policy variant of the shipped v11 submission ZIP.

The diagnostic needs the override policy to be the *only* difference from the submission
that scored on the leaderboard, so this script never retrains. It copies the shipped
``model/lgb_ranker.txt``, ``model/match_lib.py``, ``script.py`` and ``requirements.txt``
byte for byte and rewrites only ``use_callsign`` / ``use_name_lookup`` inside
``model/resources.pkl``. ``script.py`` already reads both flags from that pickle, so no
code changes.

A local retrain is deliberately not used: rebuilding v11 here reproduces
``resources.pkl`` bit for bit but produces a different ``lgb_ranker.txt`` than the
shipped model (see experiments/LOG.md), which would change two things at once.

    .\\.venv_competition_py311\\Scripts\\python.exe scripts\\make_submit_v11_override.py \\
        --policy off --out submissions\\team_v11_override_off_P1.zip

Pickle output is not byte-stable: re-dumping the shipped dict unchanged, at its own
protocol 4, differs by a few bytes. So the proof here is semantic, not byte-level. The
new ``resources.pkl`` is read back and must equal the shipped object on every key except
the two flags, with the same key set in the same order.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import shutil
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

ROOT = Path(__file__).resolve().parents[1]
SOURCE_ZIP = ROOT / "submissions" / "team_v11_merged.zip"
SOURCE_ZIP_SHA256 = "33dc878b5f4987011eb217f85527d960630225bd80c5d7c2ab60e64448420ab2"
METADATA = ROOT / "team_v11_artifacts" / "v11" / "training_metadata.json"
MEMBERS = ("model/lgb_ranker.txt", "model/match_lib.py", "model/resources.pkl",
           "script.py", "requirements.txt")
COPIED = ("model/lgb_ranker.txt", "model/match_lib.py", "script.py", "requirements.txt")
# use_callsign, use_name_lookup
POLICIES = {"off": (False, False), "callsign_only": (True, False),
            "name_only": (False, True), "both": (True, True)}
FLAG_KEYS = ("use_callsign", "use_name_lookup")
SHIPPED_PICKLE_PROTOCOL = 4


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def verify_source() -> dict:
    if not SOURCE_ZIP.is_file():
        raise FileNotFoundError(f"원본 제출 ZIP이 없습니다: {SOURCE_ZIP}")
    actual = sha256_file(SOURCE_ZIP)
    if actual != SOURCE_ZIP_SHA256:
        raise ValueError(f"원본 ZIP SHA-256이 다릅니다: 기대 {SOURCE_ZIP_SHA256}, 실제 {actual}")
    with ZipFile(SOURCE_ZIP) as archive:
        names = [item.filename for item in archive.infolist()]
    if sorted(names) != sorted(MEMBERS):
        raise ValueError(f"원본 ZIP 구성이 다릅니다: {names}")
    return {"source_zip": SOURCE_ZIP.relative_to(ROOT).as_posix(), "source_zip_sha256": actual,
            "members": names}


def dump_resources(res: dict) -> bytes:
    """Serialize at the shipped pickle protocol so the file stays loadable the same way."""
    return pickle.dumps(res, protocol=SHIPPED_PICKLE_PROTOCOL)


def deep_equal(left, right) -> bool:
    """Exact comparison that also handles the numpy arrays nested inside resources."""
    import numpy as np
    if type(left) is not type(right):
        return False
    if isinstance(left, np.ndarray):
        return (left.dtype == right.dtype and left.shape == right.shape
                and np.array_equal(left, right))
    if isinstance(left, dict):
        return (list(left) == list(right)
                and all(deep_equal(left[key], right[key]) for key in left))
    if isinstance(left, (list, tuple)):
        return len(left) == len(right) and all(deep_equal(a, b) for a, b in zip(left, right))
    if isinstance(left, (set, frozenset)):
        return left == right
    return bool(left == right)


def compare_resources(shipped: dict, rewritten: dict, flags: tuple[bool, bool]) -> dict:
    """Everything except the two flags must be unchanged, in the same order."""
    if list(shipped) != list(rewritten):
        raise ValueError("resources.pkl 키 집합·순서가 바뀌었습니다")
    changed = [key for key in shipped if not deep_equal(shipped[key], rewritten[key])]
    unexpected = sorted(set(changed) - set(FLAG_KEYS))
    if unexpected:
        raise ValueError(f"플래그 외의 값이 바뀌었습니다: {unexpected}")
    if (rewritten["use_callsign"], rewritten["use_name_lookup"]) != flags:
        raise ValueError("기록된 플래그가 요청한 정책과 다릅니다")
    return {"keys_and_order_unchanged": True, "only_changed_keys": sorted(changed),
            "compared_keys": len(shipped)}


def build(policy: str, out_zip: Path, *, label: str | None = None) -> dict:
    if policy not in POLICIES:
        raise ValueError(f"알 수 없는 정책: {policy} (가능: {sorted(POLICIES)})")
    if out_zip.resolve() == SOURCE_ZIP.resolve():
        raise ValueError("원본 제출 ZIP을 덮어쓸 수 없습니다")
    if out_zip.exists():
        raise FileExistsError(f"기존 ZIP은 덮어쓰지 않습니다: {out_zip}")
    info = verify_source()
    metadata = json.loads(METADATA.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as directory:
        staged = Path(directory) / "pkg"
        with ZipFile(SOURCE_ZIP) as archive:
            archive.extractall(staged)
        resources_path = staged / "model" / "resources.pkl"
        if sha256_file(resources_path) != metadata["resources_sha256"]:
            raise ValueError("원본 resources.pkl 해시가 학습 기록과 다릅니다")
        if sha256_file(staged / "model" / "lgb_ranker.txt") != metadata["model_sha256"]:
            raise ValueError("원본 lgb_ranker.txt 해시가 학습 기록과 다릅니다")
        with resources_path.open("rb") as stream:
            res = pickle.load(stream)
        before = (bool(res["use_callsign"]), bool(res["use_name_lookup"]))
        shipped_resources = pickle.loads(resources_path.read_bytes())
        res["use_callsign"], res["use_name_lookup"] = POLICIES[policy]
        payload = dump_resources(res)
        resources_path.write_bytes(payload)
        # Semantic proof: read the written file back and diff it against the shipped object.
        comparison = compare_resources(shipped_resources,
                                       pickle.loads(resources_path.read_bytes()),
                                       POLICIES[policy])
        copied = {name: sha256_file(staged / name) for name in COPIED}
        for name in COPIED:
            if copied[name] != sha256_file(staged / name):
                raise AssertionError("복사 대상 파일이 바뀌었습니다")
        out_zip.parent.mkdir(parents=True, exist_ok=True)
        with ZipFile(out_zip, "w", compression=ZIP_DEFLATED) as archive:
            for name in MEMBERS:
                archive.write(staged / name, name)
        result = {
            **info, "policy": policy, "label": label,
            "flags_before": {"use_callsign": before[0], "use_name_lookup": before[1]},
            "flags_after": {"use_callsign": POLICIES[policy][0],
                            "use_name_lookup": POLICIES[policy][1]},
            "resources_comparison": comparison,
            "shipped_resources_sha256": metadata["resources_sha256"],
            "pickle_protocol": SHIPPED_PICKLE_PROTOCOL,
            "copied_unchanged_sha256": copied,
            "new_resources_sha256": sha256_bytes(payload),
            "out_zip": out_zip.relative_to(ROOT).as_posix() if out_zip.is_relative_to(ROOT)
                       else out_zip.as_posix(),
            "out_zip_sha256": sha256_file(out_zip),
            "out_zip_bytes": out_zip.stat().st_size,
            "retrained": False,
        }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", choices=sorted(POLICIES), required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--label", help="LOG에 남길 짧은 설명")
    args = parser.parse_args()
    print(json.dumps(build(args.policy, args.out.resolve(), label=args.label),
                     ensure_ascii=False, indent=2))
