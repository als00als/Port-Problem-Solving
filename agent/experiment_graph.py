"""Deterministic LangGraph for bounded, offline v11 search experiments.

The graph plans safe settings, delegates model work to the competition Python
environment, compares aggregate results, and records an exploratory report.
"""

from __future__ import annotations

from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
from typing import Any, TypedDict
from uuid import uuid4

from langgraph.graph import END, START, StateGraph


ROOT = Path(__file__).resolve().parents[1]
WORKER = ROOT / "scripts" / "v11_experiment_worker.py"
COMPETITION_PYTHON = ROOT / ".venv_competition_py311" / "Scripts" / "python.exe"
LOG = ROOT / "experiments" / "LOG.md"


class ExperimentState(TypedDict, total=False):
    run_dir: str
    audit: dict[str, Any]
    plan: dict[str, Any]
    results: list[dict[str, Any]]
    comparison: dict[str, Any]
    report_path: str


def _save_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def _worker(mode: str, output: Path, split: str, setting: str | None = None) -> dict[str, Any]:
    if not COMPETITION_PYTHON.is_file():
        raise FileNotFoundError(f"대회용 Python 3.11 환경 없음: {COMPETITION_PYTHON}")
    command = [str(COMPETITION_PYTHON), str(WORKER), mode, "--split", split, "--output", str(output)]
    if setting is not None:
        command += ["--setting", setting]
    environment = dict(os.environ, PYTHONIOENCODING="utf-8", PYTHONUTF8="1")
    process = subprocess.Popen(command, cwd=ROOT, env=environment, text=True, encoding="utf-8",
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert process.stdout is not None
    for line in process.stdout:
        print(line.rstrip(), flush=True)
    if process.wait() != 0:
        raise RuntimeError(f"실험 작업자 실패: {mode} {split} {setting or ''} (exit {process.returncode})")
    return json.loads(output.read_text(encoding="utf-8"))


def build_experiment_graph():
    builder = StateGraph(ExperimentState)

    def audit(state: ExperimentState) -> dict[str, Any]:
        run_dir = Path(state["run_dir"])
        print("1/5 공식 데이터·고정 분할·그룹 누수 점검", flush=True)
        audits = {name: _worker("audit", run_dir / f"{name}_input_audit.json", name)
                  for name in ("group", "row")}
        return {"audit": audits}

    def plan(state: ExperimentState) -> dict[str, Any]:
        print("2/5 한글 후보 검색 가설·설정 생성", flush=True)
        available = state["audit"]["group"]["available_settings"]
        if set(available) != {"v11_baseline", "vocab_top1", "vocab_top3"}:
            raise ValueError("허용 목록 밖의 설정은 자동 실행하지 않습니다")
        plan_value = {
            "planner": "deterministic_allowlist_no_llm",
            "hypothesis": (
                "공식 카탈로그 단어장 표기 확장은 한글 후보의 구성과 랭커 입력 특징을 바꾼다. "
                "확장 폭 1·3을 기준 v11과 비교해 후보 포함률, 최종 정확도, 비용을 확인한다."
            ),
            "settings": [
                {"name": name, "configuration": available[name]}
                for name in ("v11_baseline", "vocab_top1", "vocab_top3")
            ],
            "comparison_rule": "고정 seed42 그룹·행 분할 각각에서 같은 검증 행, 각 설정별 독립 재학습",
            "decision_rule": "탐색 점수는 제출 모델에 자동 반영하지 않음",
            "prior_exposure": "기존 target 그룹이 과거 실험·오류 분석에 노출됨",
        }
        _save_json(Path(state["run_dir"]) / "experiment_plan.json", plan_value)
        return {"plan": plan_value}

    def run_batch(state: ExperimentState) -> dict[str, Any]:
        print("3/5 모든 설정 재학습·동일 검증 구간 평가", flush=True)
        run_dir = Path(state["run_dir"])
        results = []
        for split in ("group", "row"):
            for item in state["plan"]["settings"]:
                name = item["name"]
                results.append(_worker("evaluate", run_dir / f"{split}_{name}.json", split, name))
        return {"results": results}

    def compare(state: ExperimentState) -> dict[str, Any]:
        print("4/5 후보 포함률·최종 점수 비교", flush=True)
        comparisons = {}
        for split in ("group", "row"):
            results = {item["setting"]: item for item in state["results"] if item["split_name"] == split}
            base = results["v11_baseline"]["metrics"]
            deltas = []
            for name in ("vocab_top1", "vocab_top3"):
                current = results[name]["metrics"]
                deltas.append({
                    "setting": name,
                    "delta_weighted_raw": current["weighted_raw"] - base["weighted_raw"],
                    "delta_korean_accuracy": current["korean"]["accuracy"] - base["korean"]["accuracy"],
                    "delta_non_korean_accuracy": current["non_korean"]["accuracy"] - base["non_korean"]["accuracy"],
                    "delta_korean_candidate_recall": current["korean"]["candidate_recall"] - base["korean"]["candidate_recall"],
                    "delta_non_korean_candidate_recall": current["non_korean"]["candidate_recall"] - base["non_korean"]["candidate_recall"],
                    "delta_mean_candidates": current["all"]["mean_candidates"] - base["all"]["mean_candidates"],
                })
            best = max(results.values(),
                       key=lambda item: (item["metrics"]["weighted_raw"],
                                         -item["metrics"]["all"]["mean_candidates"]))
            comparisons[split] = {"deltas": deltas, "best_on_reused_local_split": best["setting"]}
        return {"comparison": {
            "baseline": "v11_baseline",
            "by_split": comparisons,
            "automatic_submission_or_model_promotion": False,
            "interpretation": "반복 노출된 검증 구간의 탐색적 비교; 독립 성능 추정이나 리더보드 예측 아님",
        }}

    def record(state: ExperimentState) -> dict[str, Any]:
        print("5/5 결과·제한 사항 기록", flush=True)
        run_dir = Path(state["run_dir"])
        report = {
            "run_id": run_dir.name,
            "status": "completed",
            "created_at_local": datetime.now().astimezone().isoformat(),
            "audit": state["audit"],
            "plan": state["plan"],
            "results": state["results"],
            "comparison": state["comparison"],
            "scope": "offline experiment only; no submission ZIP or leaderboard action",
        }
        report_path = run_dir / "summary.json"
        _save_json(report_path, report)
        rows = []
        for item in state["results"]:
            m = item["metrics"]
            rows.append(
                f"| {item['split_name']} | {item['setting']} | {m['all']['correct']}/{m['all']['rows']} "
                f"({m['official']['overall']:.6f}) | {m['korean']['correct']}/{m['korean']['rows']} "
                f"({m['official']['korean']:.6f}) | {m['non_korean']['correct']}/{m['non_korean']['rows']} "
                f"({m['official']['non_korean']:.6f}) | {m['weighted_raw']:.6f} | "
                f"{m['korean']['candidate_hits']}/{m['korean']['rows']} | "
                f"{m['non_korean']['candidate_hits']}/{m['non_korean']['rows']} | "
                f"{m['seen_target']['correct']}/{m['seen_target']['rows']} | "
                f"{m['unseen_target']['correct']}/{m['unseen_target']['rows']} | "
                f"{m['all']['mean_candidates']:.2f} |"
            )
        markdown = (
            f"### v11 자동 후보 검색 실험 ({datetime.now():%Y-%m-%d})\n\n"
            f"- 실행 ID: `{run_dir.name}`. 고정 `group_split_seed42.json`과 `row_split_seed42.json`; 공식 train/catalog만 사용. "
            "설정은 결정적 허용 목록에서 자동 생성했고 각 설정의 랭커를 별도 재학습했다.\n"
            f"- 가설: {state['plan']['hypothesis']}\n"
            "- 검증 예측에는 정답 열을 전달하지 않았다. 그룹 분할에서 target 그룹 겹침은 0이며, "
            "행 분할은 학습에서 본 선박과 처음 보는 선박을 별도로 집계한다. "
            "조회표는 각 학습 분할에서만 만들었고 설정마다 4개 행의 부분집합·단독 예측 일치를 확인했다.\n\n"
            "| 분할 | 설정 | 전체 | 한글 | 비한글 | 가중 원점수 | 한글 후보 포함 | 비한글 후보 포함 | 본 선박 정답 | 처음 본 선박 정답 | 평균 후보 수 |\n"
            "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n"
            + "\n".join(rows) + "\n\n"
            f"- 재사용 그룹 분할 최고 설정: `{state['comparison']['by_split']['group']['best_on_reused_local_split']}`; "
            f"행 분할 최고 설정: `{state['comparison']['by_split']['row']['best_on_reused_local_split']}`. "
            "기존 target 그룹은 과거 분석에 노출됐으므로 독립 일반화 추정이나 제출 승격 근거가 아니다. "
            "제출 ZIP과 리더보드는 변경하지 않았다.\n"
            f"- 재현 기록: `{report_path.relative_to(ROOT).as_posix()}` "
            "(입력·코드·분할 해시, 설정별 상세 점수·시간 포함).\n"
        )
        (run_dir / "summary.md").write_text(markdown, encoding="utf-8")
        with LOG.open("a", encoding="utf-8") as stream:
            stream.write("\n" + markdown)
        print(f"완료: {report_path}", flush=True)
        return {"report_path": str(report_path)}

    builder.add_node("audit", audit)
    builder.add_node("plan", plan)
    builder.add_node("run_batch", run_batch)
    builder.add_node("compare", compare)
    builder.add_node("record", record)
    builder.add_edge(START, "audit")
    builder.add_edge("audit", "plan")
    builder.add_edge("plan", "run_batch")
    builder.add_edge("run_batch", "compare")
    builder.add_edge("compare", "record")
    builder.add_edge("record", END)
    return builder.compile()


def run_experiment() -> Path:
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = ROOT / "experiments" / "results" / f"agent_v11_search_{timestamp}_{uuid4().hex[:6]}"
    run_dir.mkdir(parents=True, exist_ok=False)
    result = build_experiment_graph().invoke({"run_dir": str(run_dir)})
    return Path(result["report_path"])
