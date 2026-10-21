"""One-command offline v11 search experiment graph."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.experiment_graph import run_experiment  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=("deterministic", "llm"), default="deterministic",
                        help="llm은 외부 API에 정적 검색 설계 설명만 전송합니다")
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    print(run_experiment(args.planner))
