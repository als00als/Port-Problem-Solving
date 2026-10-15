from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUBMIT_ROOT = PROJECT_ROOT / "submit"
OUTPUT_ZIP = PROJECT_ROOT / "submissions" / "final_exp014_fixed_40_50_60.zip"


def main() -> None:
    required = [
        SUBMIT_ROOT / "model" / "adaptive_lib.py",
        SUBMIT_ROOT / "script.py",
        SUBMIT_ROOT / "requirements.txt",
    ]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError(f"Missing submit files: {missing}")

    OUTPUT_ZIP.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(OUTPUT_ZIP, "w", compression=ZIP_DEFLATED) as archive:
        archive.write(SUBMIT_ROOT / "model" / "adaptive_lib.py", "model/adaptive_lib.py")
        archive.write(SUBMIT_ROOT / "script.py", "script.py")
        archive.write(SUBMIT_ROOT / "requirements.txt", "requirements.txt")

    print(f"Created {OUTPUT_ZIP}")


if __name__ == "__main__":
    main()
