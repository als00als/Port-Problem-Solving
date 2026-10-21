"""Diagnosis-driven experiment planning with a local guard.

The LLM sees only aggregated, identifier-free diagnostics, the implemented
model description, prior experiment outcomes, the allow-listed knobs, and the
remaining budget. It may only name one knob and one allowed value per proposal;
it cannot supply code, commands, paths, or imports. Every proposal is checked
locally before anything runs, and rejected ones are recorded with reasons.
"""

from __future__ import annotations

import json
from pathlib import Path
import re
import urllib.error
import urllib.request
from typing import Any

from .error_diagnostics import DIAGNOSTIC_PROTOCOL, assert_llm_safe
from .llm_search_planner import ENDPOINT, _setting
from .retrain_config import (DECISION_RULE, EVIDENCE_TYPES, KNOBS, METRICS, OOF_ONLY_RANKERS,
                             STOP_CONDITIONS,
                             allowed_changes_for_llm, canonical_name, config_for, fingerprint,
                             validate_config)


ROOT = Path(__file__).resolve().parents[1]
MAX_PROPOSALS_HARD_LIMIT = 4

IMPLEMENTED = {
    "candidate_generation": [
        "RapidFuzz ratio on compact names, top 30", "ratio on consonant skeletons, top 30",
        "same two searches plus a tonnage bonus (<=10% gap), top 30 each",
        "Korean rows: skeleton of translated name, 3 jamo romanization variants, 3 learned "
        "syllable transliterations, each with and without tonnage bonus",
        "10 nearest catalog gross tonnages for gross and international tonnage",
        "catalog vocabulary expansion exists but is off in v11 (tested earlier, no gain)",
    ],
    "ranker": "LightGBM LGBMRanker lambdarank, 400 trees, lr 0.05, 31 leaves, seed 42, monotone "
              "constraints on similarity/gap/number features; resources for training queries are "
              "built out-of-fold with an inner 5-fold GroupKFold by target vessel; truth forced into "
              "training candidate pools when missing",
    "postprocess": [
        "name lookup override: same normalized name in the outer training rows -> its vessel "
        "(most frequent, then closest tonnage, then id)",
        "call-sign override after the name lookup (call sign has priority)",
    ],
    "tonnage_missing_representation": "relative gap features use 2.0 when either tonnage is missing; "
                                      "q_gt_missing and c_gt_missing flags exist",
    "number_handling": "num_equal, num_conflict, q_has_num, c_has_num, num_q_only, num_c_only; "
                       "Korean je-N-ho numbering normalized in tokenization",
}

PROMPT = """You plan ONE-CHANGE retraining experiments for an offline vessel entity-matching
model (port-arrival record -> one catalog vessel). You receive aggregated error
diagnostics from several previously exposed validation splits, the implemented
components, previous experiment outcomes, the only changes the runner can apply,
and the remaining budget. Propose at most {max_proposals} experiments.
Rules:
- Each proposal changes exactly one knob to one of its allowed_values.
- Cite evidence_error_types that have non-zero counts in the diagnostics AND that the
  knob lists in addresses_evidence_types; explain the link in evidence_summary.
- Do not repeat any knob/value listed in previous_experiments.
- Set duplicates_existing_feature=true if the change only restates current behavior.
- Set requires_validation_labels_or_cross_row=true if the idea would need labels
  at inference, statistics over several evaluation rows, or external data; such
  ideas are not executed. Put out-of-scope ideas in unreviewed_ideas instead.
- Text fields must be plain ASCII prose without identifiers, file paths, URLs,
  code, commands, or mentions of test/external data; such text is rejected.
- The decision rule is fixed: {decision}. Scores on these splits are exploratory.
Return only the JSON required by the schema."""

PROPOSAL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "hypothesis": {"type": "string"},
        "evidence_error_types": {"type": "array", "items": {"type": "string", "enum": list(EVIDENCE_TYPES)}},
        "evidence_summary": {"type": "string"},
        "knob": {"type": "string", "enum": list(KNOBS)},
        "value": {"type": "string", "enum": sorted({v for s in KNOBS.values() for v in s["values"]})},
        "change_kind": {"type": "string", "enum": ["training", "feature", "postprocess"]},
        "expected_effect": {"type": "string"},
        "failure_risk": {"type": "string"},
        "duplicates_existing_feature": {"type": "boolean"},
        "requires_validation_labels_or_cross_row": {"type": "boolean"},
        "metrics_to_check": {"type": "array", "items": {"type": "string", "enum": list(METRICS)}},
        "stop_condition": {"type": "string", "enum": list(STOP_CONDITIONS)},
    },
    "required": ["hypothesis", "evidence_error_types", "evidence_summary", "knob", "value",
                 "change_kind", "expected_effect", "failure_risk", "duplicates_existing_feature",
                 "requires_validation_labels_or_cross_row", "metrics_to_check", "stop_condition"],
    "additionalProperties": False,
}
OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"proposals": {"type": "array", "items": PROPOSAL_SCHEMA},
                   "unreviewed_ideas": {"type": "array", "items": {"type": "string"}}},
    "required": ["proposals", "unreviewed_ideas"],
    "additionalProperties": False,
}

FORBIDDEN_TEXT = re.compile(
    r"(test\s*(set|rows?|data|csv|file)|external\s+(data|database|db|mapping|source)|hungarian|"
    r"cross[- ]row|(several|multiple|other|all)\s+(evaluation|validation|test)\s+rows|leaderboard|subprocess|exec\(|eval\(|import\s+[a-z_]|"
    r"https?://|\.py\b|\.csv\b|[a-z]:\\)", re.IGNORECASE)
ID_TEXT = re.compile(r"\b(TRAIN|TEST|VESSEL)_\d|\bCS_[0-9A-Za-z]")
TEXT_LIMITS = {"hypothesis": (12, 600), "evidence_summary": (12, 600), "expected_effect": (5, 400),
               "failure_risk": (5, 400)}


def evidence_counts(cross: dict[str, Any]) -> dict[str, int]:
    out = {key: value["total"] for key, value in cross["error_flags"].items()}
    out.update({key: value["total"] for key, value in cross["near_miss_flags"].items()})
    return out


def search_screen_history() -> list[dict[str, Any]]:
    """Aggregates of earlier fixed-ranker search screens (no rows, no hypotheses text)."""
    items = []
    for path in sorted((ROOT / "experiments" / "results").glob("agent_v11_search_*/summary.json")):
        report = json.loads(path.read_text(encoding="utf-8"))
        for split, comp in report.get("comparison", {}).get("by_split", {}).items():
            for delta in comp.get("deltas", []):
                items.append({
                    "planner": report["plan"].get("planner", "unknown"),
                    "split": f"{split}_seed42",
                    "setting": delta["setting"],
                    "kind": "fixed_ranker_search_screen" if report["plan"].get("planner", "").startswith("openai")
                            else "retrained_vocabulary_expansion",
                    "delta_weighted_raw": round(delta["delta_weighted_raw"], 6),
                    "delta_candidate_recall_korean": round(delta["delta_korean_candidate_recall"], 6),
                    "delta_mean_candidates": round(delta["delta_mean_candidates"], 3),
                })
    return items + [{"note": "the english_variant search stage of variant_skeleton_fallback fell back to "
                             "the plain compact name for non-Korean rows and token_set_ratio ran on "
                             "space-free strings; that screen did not test its stated hypothesis and "
                             "the primitive has since been corrected"}]


def build_payload(per_split: list[dict[str, Any]], cross: dict[str, Any],
                  previous: list[dict[str, Any]], budget: dict[str, int]) -> dict[str, Any]:
    payload = {
        "diagnostic_protocol": {k: v for k, v in DIAGNOSTIC_PROTOCOL.items()},
        "split_exposure": "all splits are reused and were exposed to earlier validation and error "
                          "analysis; rows overlap between splits; results are exploratory",
        "diagnostics_per_split": per_split,
        "diagnostics_cross_split": cross,
        "implemented_components": IMPLEMENTED,
        "allowed_changes": allowed_changes_for_llm(),
        "previous_experiments": previous,
        "previous_search_screens": search_screen_history(),
        "decision_rule": DECISION_RULE,
        "budget": budget,
    }
    assert_llm_safe(payload)
    return payload


def _request_body(payload: dict[str, Any], model: str, max_proposals: int) -> dict[str, Any]:
    decision = (f"improved only if mean weighted_raw gain >= {DECISION_RULE['min_mean_weighted_raw_gain']}, "
                f"gains on >= {DECISION_RULE['min_splits_with_positive_gain']} splits, and no language "
                "or weighted drop on any split")
    return {
        "model": model,
        "store": False,
        "max_output_tokens": 6000,
        "text": {"format": {"type": "json_schema", "name": "one_change_experiments",
                            "strict": True, "schema": OUTPUT_SCHEMA}},
        "input": [
            {"role": "system", "content": PROMPT.format(max_proposals=max_proposals, decision=decision)},
            {"role": "user", "content": json.dumps(payload, ensure_ascii=True, sort_keys=True)},
        ],
    }


def propose_llm(payload: dict[str, Any], max_proposals: int, log_dir: Path, round_no: int) -> dict[str, Any]:
    api_key = _setting("OPENAI_API_KEY")
    model = _setting("OPENAI_EXPERIMENT_MODEL") or _setting("OPENAI_CHAT_MODEL")
    if not api_key or not model:
        raise RuntimeError("LLM 모드에는 OPENAI_API_KEY와 OPENAI_EXPERIMENT_MODEL/OPENAI_CHAT_MODEL이 필요합니다")
    body = _request_body(payload, model, max_proposals)
    assert_llm_safe(body["input"][1]["content"])
    log_dir.mkdir(parents=True, exist_ok=True)
    # The exact body sent (no Authorization header, no key, no rows) is kept locally.
    (log_dir / f"round_{round_no}_request.json").write_text(
        json.dumps({"endpoint": ENDPOINT, "body": body, "payload_fields": sorted(payload)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(body, ensure_ascii=True).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=300) as response:
            raw = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenAI API HTTP {exc.code}; 응답 본문은 기록하지 않았습니다") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI API 연결 실패: {exc.reason}") from None
    texts = [c.get("text", "") for item in raw.get("output", []) if item.get("type") == "message"
             for c in item.get("content", []) if c.get("type") == "output_text"]
    meta = {"planner": "openai_responses_json_schema", "model": model, "response_id": raw.get("id"),
            "status": raw.get("status"), "usage": raw.get("usage", {})}
    try:
        parsed = json.loads("\n".join(texts)) if raw.get("status") == "completed" and texts else None
    except json.JSONDecodeError:
        parsed = None
    (log_dir / f"round_{round_no}_response.json").write_text(
        json.dumps({"meta": meta, "parsed": parsed}, ensure_ascii=False, indent=2), encoding="utf-8")
    if parsed is None:
        return {"meta": meta, "proposals": [], "unreviewed_ideas": [], "error": "no parseable JSON"}
    return {"meta": meta, "proposals": parsed.get("proposals", []),
            "unreviewed_ideas": parsed.get("unreviewed_ideas", [])}


def propose_deterministic(cross: dict[str, Any], tried: set[str], max_proposals: int) -> dict[str, Any]:
    """No-API planner: rank knobs by observed evidence they address, registry order on ties."""
    counts = evidence_counts(cross)
    recurring = {k for k, v in {**cross["error_flags"], **cross["near_miss_flags"]}.items()
                 if v["pattern"] == "recurring"}
    options = []
    for order, (knob, spec) in enumerate(KNOBS.items()):
        cited = [t for t in spec["addresses"] if counts.get(t, 0) > 0]
        if not cited:
            continue
        weight = sum(counts[t] * (2 if t in recurring else 1) for t in cited)
        for value in spec["values"]:
            if value in OOF_ONLY_RANKERS:
                continue
            if value == spec["baseline"] or fingerprint(config_for(knob, value)) in tried:
                continue
            options.append((-weight, order, spec["values"].index(value), knob, value, cited))
            break  # one value per knob per round keeps rounds single-cause and diverse
    options.sort()
    proposals = [{
        "hypothesis": f"Observed {', '.join(cited)} cases may be reduced by changing {knob} to {value}.",
        "evidence_error_types": cited,
        "evidence_summary": "Counts across reused splits: " + ", ".join(f"{t}={counts[t]}" for t in cited),
        "knob": knob, "value": value, "change_kind": KNOBS[knob]["kind"],
        "expected_effect": "fewer errors or near misses of the cited types",
        "failure_risk": "the change can move other rows; small error counts make the result noisy",
        "duplicates_existing_feature": False, "requires_validation_labels_or_cross_row": False,
        "metrics_to_check": ["weighted_raw", "korean_accuracy", "non_korean_accuracy",
                             "newly_correct_rows", "newly_wrong_rows"],
        "stop_condition": "any_language_accuracy_drop",
    } for _, _, _, knob, value, cited in options[:max_proposals]]
    return {"meta": {"planner": "deterministic_evidence_ranking_no_llm"}, "proposals": proposals,
            "unreviewed_ideas": []}


def propose_fixed_change(cross: dict[str, Any], knob: str, value: str) -> dict[str, Any]:
    """A user-selected allowed setting still passes the same evidence/duplicate guard."""
    validate_config(config_for(knob, value))
    if value in OOF_ONLY_RANKERS:
        raise ValueError("MLP 랭커는 고정 OOF 실험에서만 비교합니다")
    counts = evidence_counts(cross)
    cited = [key for key in KNOBS[knob]["addresses"] if counts.get(key, 0)]
    proposal = {
        "hypothesis": f"Changing {knob} to {value} may address the observed ranking failure patterns.",
        "evidence_error_types": cited,
        "evidence_summary": "Counts across reused splits: " + ", ".join(f"{key}={counts[key]}" for key in cited),
        "knob": knob, "value": value, "change_kind": KNOBS[knob]["kind"],
        "expected_effect": "a different ranking decision for ambiguous candidates",
        "failure_risk": "new errors may replace corrected errors; exposed splits remain exploratory",
        "duplicates_existing_feature": False, "requires_validation_labels_or_cross_row": False,
        "metrics_to_check": list(METRICS), "stop_condition": "any_language_accuracy_drop",
    }
    return {"meta": {"planner": "user_selected_allowlist_no_llm"}, "proposals": [proposal],
            "unreviewed_ideas": []}


def guard(proposals: list[Any], cross: dict[str, Any], tried: set[str], max_proposals: int
          ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Accept only well-formed, single-change, evidence-backed, non-duplicate, in-scope proposals."""
    counts = evidence_counts(cross)
    accepted, rejected, seen = [], [], set()
    for position, item in enumerate(proposals):
        reasons = []
        if not isinstance(item, dict) or set(item) != set(PROPOSAL_SCHEMA["required"]):
            rejected.append({"position": position, "reasons": ["schema_invalid"]})
            continue
        knob, value = item["knob"], item["value"]
        if position >= min(max_proposals, MAX_PROPOSALS_HARD_LIMIT):
            reasons.append("over_proposal_limit")
        if knob not in KNOBS:
            reasons.append("unknown_knob")
        elif not isinstance(value, str) or value not in KNOBS[knob]["values"]:
            reasons.append("value_not_allowed_for_knob")
        elif value == KNOBS[knob]["baseline"]:
            reasons.append("no_change_duplicates_current_implementation")
        if value in OOF_ONLY_RANKERS:
            reasons.append("mlp_ranker_reserved_for_oof")
        if item["duplicates_existing_feature"] is True:
            reasons.append("declared_duplicate_of_existing_feature")
        if item["requires_validation_labels_or_cross_row"] is True:
            reasons.append("requires_labels_cross_row_or_external_data")
        if knob in KNOBS and item["change_kind"] != KNOBS[knob]["kind"]:
            reasons.append("change_kind_does_not_match_knob")
        cited = item["evidence_error_types"]
        if not isinstance(cited, list) or not cited or not all(t in EVIDENCE_TYPES for t in cited):
            reasons.append("evidence_types_invalid")
        else:
            if not any(counts.get(t, 0) > 0 for t in cited):
                reasons.append("evidence_not_observed_in_diagnostics")
            if knob in KNOBS and not set(cited) & set(KNOBS[knob]["addresses"]):
                reasons.append("knob_does_not_address_cited_evidence")
        if (not isinstance(item["metrics_to_check"], list) or not item["metrics_to_check"]
                or not all(m in METRICS for m in item["metrics_to_check"])
                or item["stop_condition"] not in STOP_CONDITIONS):
            reasons.append("metrics_or_stop_condition_invalid")
        for field, (low, high) in TEXT_LIMITS.items():
            text = item[field]
            if not isinstance(text, str) or not low <= len(text) <= high:
                reasons.append(f"{field}_length")
            elif FORBIDDEN_TEXT.search(text) or ID_TEXT.search(text) or any(ord(c) > 126 for c in text):
                reasons.append(f"{field}_forbidden_content")
        name = config = fp = None
        if not any(r in reasons for r in ("unknown_knob", "value_not_allowed_for_knob",
                                          "no_change_duplicates_current_implementation")):
            config = config_for(knob, value)
            if validate_config(config) != [knob]:
                reasons.append("not_a_single_change")
            name, fp = canonical_name(knob, value), fingerprint(config)
            if fp in tried:
                reasons.append("duplicate_of_previous_experiment")
            if fp in seen:
                reasons.append("duplicate_within_round")
        record = {"position": position, "knob": knob, "value": value, "name": name, "fingerprint": fp,
                  "hypothesis": item["hypothesis"] if f"hypothesis_forbidden_content" not in reasons else None,
                  "evidence_error_types": cited if isinstance(cited, list) else None}
        if reasons:
            rejected.append({**record, "reasons": reasons})
            continue
        seen.add(fp)
        accepted.append({**record, "config": config, "proposal": item,
                         "pre_run_consistency": {"single_knob_changed": knob,
                                                 "name_is_canonical_for_change": name == canonical_name(knob, value),
                                                 "kind": KNOBS[knob]["kind"],
                                                 "retrains_ranker": KNOBS[knob]["retrain"],
                                                 "knob_description": KNOBS[knob]["description"]}})
    return accepted, rejected
