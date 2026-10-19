"""Diagnosis-driven, budgeted v11 one-change retraining loop (offline only)."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agent.improvement_graph import DEFAULT_SPLITS, Budget, run_improvement  # noqa: E402
from agent.retrain_config import KNOBS, config_for, validate_config  # noqa: E402


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--planner", choices=("deterministic", "llm"), default="deterministic",
                        help="llm은 식별 정보를 뺀 집계 진단만 외부 API에 전송합니다")
    parser.add_argument("--max-llm-calls", type=int, default=3, help="계획 회차(LLM 호출) 최대 수")
    parser.add_argument("--max-training-experiments", type=int, default=3)
    parser.add_argument("--max-proposals", type=int, default=3)
    parser.add_argument("--patience", type=int, default=2, help="개선 없는 회차가 이 수에 도달하면 중단")
    parser.add_argument("--knob", choices=KNOBS, help="허용 설정 하나를 직접 지정; deterministic 모드 전용")
    parser.add_argument("--value", help="--knob의 허용 값; 지정 시 단일 실험 예산으로 실행")
    parser.add_argument("--splits", nargs="+", choices=DEFAULT_SPLITS, default=list(DEFAULT_SPLITS))
    args = parser.parse_args()
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    fixed_change = None
    if args.knob is not None or args.value is not None:
        if args.knob is None or args.value is None or args.planner != "deterministic":
            parser.error("--knob와 --value를 함께 지정하고 --planner deterministic을 사용하세요")
        fixed_change = (args.knob, args.value)
        if len(validate_config(config_for(*fixed_change))) != 1:
            parser.error("기준값과 다른 설정 하나를 지정하세요")
    budget = (Budget(1, 1, 1, 1) if fixed_change else
              Budget(args.max_llm_calls, args.max_training_experiments, args.max_proposals, args.patience))
    print(run_improvement(args.planner, budget, tuple(args.splits), fixed_change))
