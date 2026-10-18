"""Build a DACON submission package using MatchingService top-ranked results."""

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMIT_ROOT = PROJECT_ROOT / "submit"
OUTPUT_ZIP = PROJECT_ROOT / "submissions" / "final_matching_agent_candidate.zip"


def main() -> None:
    required = [
        SUBMIT_ROOT / "agent_script.py",
        SUBMIT_ROOT / "model" / "__init__.py",
        SUBMIT_ROOT / "model" / "adaptive_lib.py",
        SUBMIT_ROOT / "model" / "record_validation.py",
        SUBMIT_ROOT / "model" / "matching_service.py",
        SUBMIT_ROOT / "requirements.txt",
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing submission files: {missing}")

    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUTPUT_ZIP, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(SUBMIT_ROOT / "agent_script.py", "script.py")
        archive.write(SUBMIT_ROOT / "model" / "__init__.py", "model/__init__.py")
        archive.write(SUBMIT_ROOT / "model" / "adaptive_lib.py", "model/adaptive_lib.py")
        archive.write(SUBMIT_ROOT / "model" / "record_validation.py", "model/record_validation.py")
        archive.write(SUBMIT_ROOT / "model" / "matching_service.py", "model/matching_service.py")
        archive.write(SUBMIT_ROOT / "requirements.txt", "requirements.txt")

    print(f"Created {OUTPUT_ZIP}")


if __name__ == "__main__":
    main()
