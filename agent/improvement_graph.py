"""Closed, budgeted loop: audit -> diagnose -> plan -> guard -> retrain -> decide -> record.

Runs in the agent environment and delegates all data work to the competition
Python 3.11 worker (``scripts/v11_improvement_worker.py``). The planner is
either deterministic (no API) or an LLM that only sees identifier-free
aggregates. Nothing here touches submission folders, ZIPs, or the leaderboard.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph

from .error_diagnostics import DIAGNOSTIC_PROTOCOL, assert_llm_safe, cross_split
from .improvement_planner import build_payload, guard, propose_deterministic, propose_llm, propose_fixed_change
from .retrain_config import DECISION_RULE, KNOBS, classify_outcome, config_for, validate_config


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "v11_improvement_worker.py"
COMPETITION_PYTHON = ROOT / ".venv_competition_py311" / "Scripts" / "python.exe"
LOG = ROOT / "experiments" / "LOG.md"
LEDGER = ROOT / "experiments" / "results" / "agent_improvement_ledger.json"
DEFAULT_SPLITS = ("group_seed42", "group_seed123", "group_seed2026", "row_seed42")


@dataclass(frozen=True)
class Budget:
    max_llm_calls: int = 3            # planning rounds that may call the API
    max_training_experiments: int = 3  # one experiment = one setting on every split
    max_proposals_per_round: int = 3
    patience_rounds_without_improvement: int = 2

    def __post_init__(self) -> None:
        if not (1 <= self.max_llm_calls <= 5 and 1 <= self.max_training_experiments <= 8
                and 1 <= self.max_proposals_per_round <= 4
                and 1 <= self.patience_rounds_without_improvement <= 5):
            raise ValueError("예산은 허용 범위(LLM 1~5, 학습 1~8, 제안 1~4, patience 1~5) 안이어야 합니다")


class LoopState(TypedDict, total=False):
    run_dir: str
    splits: list[str]
    planner: str
    budget: dict[str, int]
    audit: dict[str, Any]
    diagnostics: dict[str, Any]
    cross: dict[str, Any]
    round: int
    planning_calls: int
    llm_calls: int
    experiments_run: int
    rounds_without_improvement: int
    last_round: dict[str, Any]
    rounds: list[dict[str, Any]]
    experiments: list[dict[str, Any]]
    unreviewed: list[dict[str, Any]]
    stop_reason: str
    report_path: str


def _save(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _worker(args: list[str], output: Path) -> dict[str, Any]:
    if not COMPETITION_PYTHON.is_file():
        raise FileNotFoundError(f"대회용 Python 3.11 환경 없음: {COMPETITION_PYTHON}")
    command = [str(COMPETITION_PYTHON), str(WORKER), *args, "--output", str(output)]
    env = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    env.pop("OPENAI_API_KEY", None)  # the data worker never needs API credentials
    process = subprocess.Popen(command, cwd=ROOT, env=env, text=True, encoding="utf-8",
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert process.stdout is not None
    for line in process.stdout:
        print("  " + line.rstrip(), flush=True)
    if process.wait() != 0:
        raise RuntimeError(f"작업자 실패: {' '.join(args[:3])} (exit {process.returncode})")
    return json.loads(output.read_text(encoding="utf-8"))


def load_ledger() -> dict[str, Any]:
    if LEDGER.is_file():
        return json.loads(LEDGER.read_text(encoding="utf-8"))
    return {"description": "aggregated one-change experiment outcomes; no rows or identifiers",
            "entries": []}


def _previous_for_llm(entries: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = ("knob", "value", "status", "outcome", "mean_delta_weighted_raw", "newly_correct_total",
            "newly_wrong_total", "training_seconds_total", "prediction_seconds_total",
            "behavior_check_passed", "rejection_reasons", "splits", "protocol_version")
    return [{k: e[k] for k in keys if k in e} for e in entries]


def build_improvement_graph(planner: str = "llm", fixed_change: tuple[str, str] | None = None):
    if planner not in {"deterministic", "llm"}:
        raise ValueError("planner는 deterministic 또는 llm이어야 합니다")
    if fixed_change is not None:
        if planner != "deterministic" or len(validate_config(config_for(*fixed_change))) != 1:
            raise ValueError("고정 설정은 deterministic 모드의 단일 변경이어야 합니다")
    builder = StateGraph(LoopState)

    def audit(state: LoopState) -> dict[str, Any]:
        run_dir = Path(state["run_dir"])
        print("1/6 공식 데이터·고정 분할 감사", flush=True)
        audits = {s: _worker(["audit", "--split", s], run_dir / f"audit_{s}.json") for s in state["splits"]}
        return {"audit": audits}

    def diagnose(state: LoopState) -> dict[str, Any]:
        run_dir = Path(state["run_dir"])
        print("2/6 분할별 기준 v11 재학습·오류 진단", flush=True)
        out = {s: _worker(["diagnose", "--split", s, "--run-dir", str(run_dir)],
                          run_dir / f"diagnose_{s}.json") for s in state["splits"]}
        summaries = [out[s]["diagnostic_summary"] for s in state["splits"]]
        cross = cross_split(summaries)
        assert_llm_safe({"per_split": summaries, "cross": cross})
        _save(run_dir / "diagnostics_report.json", {
            "protocol": DIAGNOSTIC_PROTOCOL, "per_split": summaries, "cross_split": cross,
            "baseline_metrics": {s: out[s]["metrics"] for s in state["splits"]},
            "exposure": "reused, previously exposed splits; exploratory, not an independent validation",
            "local_case_files": {s: out[s]["local_case_file"] for s in state["splits"]}})
        return {"diagnostics": out, "cross": cross}

    def plan(state: LoopState) -> dict[str, Any]:
        run_dir = Path(state["run_dir"])
        round_no = state.get("round", 0) + 1
        budget = state["budget"]
        print(f"3/6 계획 {round_no}회차 ({planner})", flush=True)
        ledger = load_ledger()
        tried = {e["fingerprint"] for e in ledger["entries"] if e.get("status") == "executed"}
        remaining = {"llm_calls_remaining": budget["max_llm_calls"] - state.get("llm_calls", 0),
                     "training_experiments_remaining":
                         budget["max_training_experiments"] - state.get("experiments_run", 0),
                     "max_proposals_this_round": budget["max_proposals_per_round"]}
        max_props = min(budget["max_proposals_per_round"], remaining["training_experiments_remaining"])
        summaries = [state["diagnostics"][s]["diagnostic_summary"] for s in state["splits"]]
        if fixed_change is not None:
            response = propose_fixed_change(state["cross"], *fixed_change)
            llm_calls = state.get("llm_calls", 0)
        elif planner == "llm":
            payload = build_payload(summaries, state["cross"], _previous_for_llm(ledger["entries"]), remaining)
            response = propose_llm(payload, max_props, run_dir / "llm", round_no)
            llm_calls = state.get("llm_calls", 0) + 1
        else:
            response = propose_deterministic(state["cross"], tried, max_props)
            llm_calls = state.get("llm_calls", 0)
        accepted, rejected = guard(response["proposals"], state["cross"], tried, max_props)
        round_record = {"round": round_no, "planner": response["meta"], "proposals_received":
                        len(response["proposals"]), "accepted": [
                            {k: a[k] for k in ("name", "knob", "value", "fingerprint", "hypothesis",
                                               "evidence_error_types", "pre_run_consistency")}
                            | {"proposal": a["proposal"]} for a in accepted],
                        "rejected": rejected, "unreviewed_ideas": response.get("unreviewed_ideas", []),
                        "error": response.get("error")}
        _save(run_dir / f"round_{round_no}_plan.json", round_record)
        for item in rejected:
            ledger["entries"].append({"run_id": run_dir.name, "round": round_no, "status": "rejected",
                                      "knob": item.get("knob"), "value": item.get("value"),
                                      "fingerprint": item.get("fingerprint"),
                                      "rejection_reasons": item["reasons"],
                                      "protocol_version": DECISION_RULE["protocol_version"]})
        _save(LEDGER, ledger)
        return {"round": round_no, "planning_calls": state.get("planning_calls", 0) + 1,
                "llm_calls": llm_calls, "last_round": {**round_record, "_accepted": accepted}}

    def execute(state: LoopState) -> dict[str, Any]:
        run_dir = Path(state["run_dir"])
        budget = state["budget"]
        accepted = state["last_round"]["_accepted"]
        used = state.get("experiments_run", 0)
        experiments = list(state.get("experiments", []))
        unreviewed = list(state.get("unreviewed", []))
        improved_this_round = False
        ledger = load_ledger()
        print(f"4/6 승인된 설정 {len(accepted)}개 재학습·비교", flush=True)
        for item in accepted:
            if used >= budget["max_training_experiments"]:
                unreviewed.append({"round": state["round"], "name": item["name"], "knob": item["knob"],
                                   "value": item["value"], "reason": "training_budget_exhausted",
                                   "hypothesis": item["hypothesis"]})
                continue
            per_split = [_worker(["retrain", "--split", s, "--run-dir", str(run_dir), "--knob", item["knob"],
                                  "--value", item["value"]], run_dir / f"retrain_{s}_{item['name']}.json")
                         for s in state["splits"]]
            used += 1
            # Post-run: the executed configuration must be exactly what the guard approved.
            consistency = {
                "config_matches_proposal": all(r["config"] == item["config"] for r in per_split),
                "fingerprint_matches": all(r["config_fingerprint"] == item["fingerprint"] for r in per_split),
                "setting_name_matches": all(r["setting"] == item["name"] for r in per_split),
                "behavior_checks_passed": all(r["behavior_check"]["passed"] for r in per_split),
                "behavior_checks": {r["split_name"]: r["behavior_check"]["checks"] for r in per_split},
                "retrained_as_declared": all(r["retrained"] == KNOBS[item["knob"]]["retrain"] for r in per_split),
            }
            ok = all(v for k, v in consistency.items() if k != "behavior_checks")
            outcome = classify_outcome([{"delta_weighted_raw": r["delta"]["weighted_raw"],
                                         "delta_korean_accuracy": r["delta"]["korean_accuracy"],
                                         "delta_non_korean_accuracy": r["delta"]["non_korean_accuracy"],
                                         "newly_correct": r["row_changes"]["newly_correct"],
                                         "newly_wrong": r["row_changes"]["newly_wrong"]} for r in per_split], ok)
            improved_this_round = improved_this_round or outcome["label"] == "improved"
            record = {"round": state["round"], "name": item["name"], "knob": item["knob"], "value": item["value"],
                      "fingerprint": item["fingerprint"], "hypothesis": item["hypothesis"],
                      "evidence_error_types": item["evidence_error_types"], "outcome": outcome,
                      "post_run_consistency": consistency,
                      "per_split": [{k: r[k] for k in ("split_name", "retrained", "delta", "row_changes",
                                                       "feature_importance_shift_top8", "timing_seconds",
                                                       "model_sha256", "config_fingerprint", "training_stats",
                                                       "behavior_check", "checks")}
                                    | {k: r[k] for k in ("complementarity", "prediction_input_hashes",
                                                        "baseline_prediction_input_hashes", "local_model_file",
                                                        "feature_importance_method") if k in r}
                                    | {"metrics": {g: r["metrics"][g] for g in
                                                   ("weighted_raw", "official", "all", "korean", "non_korean",
                                                    "seen_target", "unseen_target")}}
                                    for r in per_split]}
            experiments.append(record)
            ledger["entries"].append({
                "run_id": run_dir.name, "round": state["round"], "status": "executed",
                "knob": item["knob"], "value": item["value"], "name": item["name"],
                "fingerprint": item["fingerprint"], "outcome": outcome["label"],
                "mean_delta_weighted_raw": round(outcome["mean_delta_weighted_raw"], 6),
                "per_split_delta_weighted_raw": [round(r["delta"]["weighted_raw"], 6) for r in per_split],
                "newly_correct_total": sum(r["row_changes"]["newly_correct"] for r in per_split),
                "newly_wrong_total": sum(r["row_changes"]["newly_wrong"] for r in per_split),
                "training_seconds_total": round(sum(r["timing_seconds"]["training"] for r in per_split), 1),
                "prediction_seconds_total": round(sum(r["timing_seconds"]["validation_prediction"]
                                                      for r in per_split), 1),
                "behavior_check_passed": consistency["behavior_checks_passed"],
                "splits": [r["split_name"] for r in per_split],
                "protocol_version": DECISION_RULE["protocol_version"]})
            _save(LEDGER, ledger)
            print(f"  {item['name']}: {outcome['label']} (평균 가중 변화 {outcome['mean_delta_weighted_raw']:+.6f})",
                  flush=True)
        for idea in state["last_round"].get("unreviewed_ideas", []):
            unreviewed.append({"round": state["round"], "reason": "outside_allowed_changes_not_executed",
                               "idea": idea})
        streak = 0 if improved_this_round else state.get("rounds_without_improvement", 0) + 1
        rounds = list(state.get("rounds", [])) + [{k: v for k, v in state["last_round"].items() if k != "_accepted"}
                                                   | {"improved": improved_this_round}]
        return {"experiments_run": used, "experiments": experiments, "unreviewed": unreviewed,
                "rounds_without_improvement": streak, "rounds": rounds}

    def decide(state: LoopState) -> dict[str, Any]:
        budget = state["budget"]
        last = state["last_round"]
        if not last["_accepted"]:
            reason = "all_proposals_rejected_or_none_proposed"
        elif state["experiments_run"] >= budget["max_training_experiments"]:
            reason = "training_experiment_budget_exhausted"
        elif state["planning_calls"] >= budget["max_llm_calls"]:
            reason = "planning_call_budget_exhausted"
        elif state["rounds_without_improvement"] >= budget["patience_rounds_without_improvement"]:
            reason = "no_improvement_within_patience"
        else:
            reason = ""
        print(f"5/6 중단 판단: {reason or '다음 회차 진행'}", flush=True)
        return {"stop_reason": reason}

    def record(state: LoopState) -> dict[str, Any]:
        run_dir = Path(state["run_dir"])
        print("6/6 결과·다음 가설 기록", flush=True)
        budget = state["budget"]
        remaining = {"llm_calls": budget["max_llm_calls"] - state.get("llm_calls", 0),
                     "planning_rounds": budget["max_llm_calls"] - state.get("planning_calls", 0),
                     "training_experiments": budget["max_training_experiments"] - state.get("experiments_run", 0)}
        report = {
            "run_id": run_dir.name, "created_at_local": datetime.now().astimezone().isoformat(),
            "planner": planner, "budget": budget, "remaining_budget": remaining,
            "stop_reason": state.get("stop_reason"), "splits": state["splits"],
            "decision_rule": DECISION_RULE, "diagnostic_protocol": DIAGNOSTIC_PROTOCOL,
            "baseline": {s: {"metrics": d["metrics"], "model_sha256": d["model_sha256"],
                             "config_fingerprint": d["config_fingerprint"],
                             "timing_seconds": d["timing_seconds"], "checks": d["checks"]}
                         for s, d in state["diagnostics"].items()},
            "cross_split_diagnostics": state["cross"], "rounds": state.get("rounds", []),
            "experiments": state.get("experiments", []), "unreviewed": state.get("unreviewed", []),
            "llm_calls_made": state.get("llm_calls", 0),
            "scope": "offline exploratory experiments on reused, previously exposed splits; no "
                     "submission ZIP, submission code, or leaderboard action",
        }
        path = run_dir / "summary.json"
        _save(path, report)
        markdown = render_markdown(report, path)
        (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
        with LOG.open("a", encoding="utf-8") as stream:
            stream.write("\n" + markdown)
        print(f"완료: {path}", flush=True)
        return {"report_path": str(path)}

    for name, fn in (("audit", audit), ("diagnose", diagnose), ("plan", plan), ("execute", execute),
                     ("decide", decide), ("record", record)):
        builder.add_node(name, fn)
    builder.add_edge(START, "audit")
    builder.add_edge("audit", "diagnose")
    builder.add_edge("diagnose", "plan")
    builder.add_edge("plan", "execute")
    builder.add_edge("execute", "decide")
    builder.add_conditional_edges("decide", lambda s: "record" if s["stop_reason"] else "plan",
                                  {"record": "record", "plan": "plan"})
    builder.add_edge("record", END)
    return builder.compile()


def render_markdown(report: dict[str, Any], path: Path) -> str:
    lines = [f"### v11 진단 기반 단일 변경 재학습 루프 ({report['planner']}, "
             f"{report['created_at_local'][:10]})", ""]
    lines.append(f"- 실행 ID: `{report['run_id']}`. 분할: {', '.join(report['splits'])} (모두 과거 검증·오류 "
                 "분석에 노출된 재사용 분할, 행 중복 있음 — 독립 검증 아님). 공식 train/catalog만 사용.")
    lines.append(f"- LLM 호출 {report['llm_calls_made']}회, 학습 실험 "
                 f"{report['budget']['max_training_experiments'] - report['remaining_budget']['training_experiments']}"
                 f"/{report['budget']['max_training_experiments']}회, 중단 사유 `{report['stop_reason']}`.")
    lines.append("- 기준 v11(분할별 재학습, seed 42):")
    lines.append("")
    lines.append("| 분할 | 전체 | 한글 | 비한글 | 가중 원점수 | 후보 포함 | 학습 s | 예측 s |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for split, item in report["baseline"].items():
        m = item["metrics"]
        lines.append(f"| {split} | {m['all']['correct']}/{m['all']['rows']} | {m['korean']['correct']}/"
                     f"{m['korean']['rows']} | {m['non_korean']['correct']}/{m['non_korean']['rows']} | "
                     f"{m['weighted_raw']:.6f} | {m['all']['candidate_hits']}/{m['all']['rows']} | "
                     f"{item['timing_seconds']['training']:.0f} | {item['timing_seconds']['validation_prediction']:.0f} |")
    cross = report["cross_split_diagnostics"]
    recurring = [k for k, v in {**cross["error_flags"], **cross["near_miss_flags"]}.items() if v["pattern"] == "recurring"]
    specific = [k for k, v in {**cross["error_flags"], **cross["near_miss_flags"]}.items()
                if v["pattern"] == "split_specific"]
    lines += ["", f"- 반복 진단 유형: {', '.join(recurring) or '없음'}; 분할 특이 유형: {', '.join(specific) or '없음'}."]
    if report["experiments"]:
        lines += ["", "| 회차 | 설정 | 결과 | 평균 Δ가중 | 분할별 Δ가중 | 새로 맞힘/틀림 | 동작 검사 | 재학습 |",
                  "|---:|---|---|---:|---|---:|---|---|"]
        for e in report["experiments"]:
            deltas = ", ".join(f"{p['delta']['weighted_raw']:+.6f}" for p in e["per_split"])
            nc = sum(p["row_changes"]["newly_correct"] for p in e["per_split"])
            nw = sum(p["row_changes"]["newly_wrong"] for p in e["per_split"])
            lines.append(f"| {e['round']} | `{e['name']}` | {e['outcome']['label']} | "
                         f"{e['outcome']['mean_delta_weighted_raw']:+.6f} | {deltas} | {nc}/{nw} | "
                         f"{'통과' if e['post_run_consistency']['behavior_checks_passed'] else '실패'} | "
                         f"{'예' if e['per_split'][0]['retrained'] else '아니오(후처리)'} |")
        for experiment in report["experiments"]:
            if experiment["knob"] != "ranker_family":
                continue
            lines += ["", "- 랭커 교체의 오류 상보성(조회표 후처리 적용 후):",
                      "", "| 분할 | 둘 다 정답 | 기준만 정답 | 대안만 정답 | 둘 다 오답 | 예측 불일치 |",
                      "|---|---:|---:|---:|---:|---:|"]
            for item in experiment["per_split"]:
                counts = item["complementarity"]["final"]["all"]
                lines.append(f"| {item['split_name']} | {counts['both_correct']} | "
                             f"{counts['baseline_only_correct']} | {counts['alternative_only_correct']} | "
                             f"{counts['both_wrong']} | {counts['prediction_disagreements']} |")
            lines += ["", "- 학습 행렬·검증 후보·검증 특징 해시를 기준과 대조했다. "
                      "CatBoost 중요도(PredictionValuesChange)는 LightGBM gain과 직접 비교하지 않는다. "
                      "JSON의 oracle 정확도는 정답을 사용하는 진단 상한이며 실제 예측이나 앙상블 성능이 아니다."]
    rejected = [r for rnd in report["rounds"] for r in rnd["rejected"]]
    if rejected:
        lines += ["", "- 실행하지 않은 제안: " + "; ".join(
            f"{r.get('knob')}={r.get('value')} ({', '.join(r['reasons'])})" for r in rejected)]
    if report["unreviewed"]:
        lines += ["- 검토·실행하지 않은 가설 {}건(예산 소진 또는 허용 범위 밖), 상세는 summary.json.".format(
            len(report["unreviewed"]))]
    lines += [f"- 남은 예산: 계획 회차 {report['remaining_budget']['planning_rounds']}회"
              f"{' (LLM 호출 포함)' if report['planner'] == 'llm' else ' (결정적 계획, LLM 미사용)'}, 학습 실험 "
              f"{report['remaining_budget']['training_experiments']}회.",
              f"- 개선 판정 규칙(사전 고정, {DECISION_RULE['protocol_version']}): 모든 분할에서 한글·비한글 "
              f"정확도와 가중 원점수가 하락하지 않고, 평균 가중 원점수 변화 >= "
              f"{DECISION_RULE['min_mean_weighted_raw_gain']}, 개선 분할 >= "
              f"{DECISION_RULE['min_splits_with_positive_gain']}개. 결과는 탐색용이며 "
              "독립 일반화 성능이나 제출 승격 근거가 아니다. 제출 ZIP·제출 코드·리더보드는 변경하지 않았다.",
              f"- 재현 기록: `{path.relative_to(ROOT).as_posix()}` (행 ID 포함 사례 파일은 `local_only/`, Git 제외).", ""]
    return "\n".join(lines)


def run_improvement(planner: str = "llm", budget: Budget | None = None,
                    splits: tuple[str, ...] = DEFAULT_SPLITS,
                    fixed_change: tuple[str, str] | None = None) -> Path:
    budget = budget or (Budget(1, 1, 1, 1) if fixed_change else Budget())
    if fixed_change is not None and budget != Budget(1, 1, 1, 1):
        raise ValueError("고정 설정 실행은 계획·실험·제안·patience를 각 1로 제한합니다")
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "experiments" / "results" / f"agent_v11_improve_{timestamp}_{uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    graph = build_improvement_graph(planner, fixed_change)
    limit = 6 + 3 * budget.max_llm_calls + 5
    result = graph.invoke({"run_dir": str(run_dir), "splits": list(splits), "planner": planner,
                           "budget": asdict(budget)}, {"recursion_limit": limit})
    return Path(result["report_path"])
