"""Request bounded search-policy ideas without sending competition records.

Only a static task description and the policy schema leave the machine. The
response is untrusted until search_policy.validate_proposals accepts it.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
import urllib.error
import urllib.request
from typing import Any

from .search_policy import validate_proposals


ROOT = Path(__file__).resolve().parents[1]
ENDPOINT = "https://api.openai.com/v1/responses"


def _setting(name: str) -> str | None:
    value = os.environ.get(name)
    if value:
        return value
    env_path = ROOT / ".env"
    if env_path.is_file():
        for line in env_path.read_text(encoding="utf-8-sig").splitlines():
            if line.strip().startswith(name + "="):
                return line.split("=", 1)[1].strip().strip("\"'") or None
    return None


STAGE_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "when": {"type": "string", "enum": ["all", "korean", "non_korean"]},
        "query_form": {"type": "string", "enum": ["translated", "romanized", "english_variant", "learned",
                                                  "skeleton", "tokens"]},
        "catalog_field": {"type": "string", "enum": ["compact", "alpha", "skeleton", "tokens"]},
        "scorer": {"type": "string", "enum": ["ratio", "weighted_ratio", "token_set_ratio"]},
        "top_k": {"type": "integer"},
        "min_similarity": {"type": "number"},
        "max_gt_gap": {"type": ["number", "null"]},
    },
    "required": ["when", "query_form", "catalog_field", "scorer", "top_k", "min_similarity", "max_gt_gap"],
    "additionalProperties": False,
}

OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "proposals": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string"},
                    "hypothesis": {"type": "string"},
                    "stages": {"type": "array", "items": STAGE_SCHEMA},
                },
                "required": ["name", "hypothesis", "stages"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["proposals"],
    "additionalProperties": False,
}


STATIC_PROMPT = """Design one or two NEW deterministic candidate-retrieval policies for matching one
port-arrival record to one vessel in an official catalog. Some record names are
Korean transliterations of English catalog names. A separate fixed LightGBM
ranker will evaluate the union of your candidates and its existing candidates.
Use 1-2 stages per policy. Each stage selects a query form and catalog field,
scores strings, keeps up to top_k matches above min_similarity, and may filter
by maximum relative gross-tonnage gap. Different stages are unioned. Prefer a
meaningful hybrid or conditional retrieval policy over merely changing top_k.
Allowed query forms: translated, romanized, english_variant (Korean rows only),
learned (Korean rows only), skeleton, tokens (space-separated words).
Allowed catalog fields: compact, alpha (digits removed), skeleton (all without
spaces), tokens (space-separated words).
Allowed scorers: ratio, weighted_ratio, token_set_ratio. token_set_ratio is
only accepted with query_form tokens and catalog_field tokens; english_variant
and learned are rejected with when=non_korean because they produce no variant.
Allowed when: all, korean, non_korean. top_k must be 5..60, min_similarity
0..100, max_gt_gap null or 0..1. Combined top_k may not exceed 120.
Name: lower-case ASCII identifier 3..40 characters, starting with a letter.
Hypothesis: 12..350 characters. Explain why the combination could find a
correct candidate absent from the existing pool. Do not request external vessel
data, test rows, labels, code execution, or changes to the competition metric.
Return only the required JSON. No actual competition records or metrics have
been provided to you; do not claim measured improvement.
"""


def propose_search_policies() -> dict[str, Any]:
    api_key = _setting("OPENAI_API_KEY")
    model = _setting("OPENAI_EXPERIMENT_MODEL") or _setting("OPENAI_CHAT_MODEL")
    if not api_key:
        raise RuntimeError("LLM 모드에는 OPENAI_API_KEY가 필요합니다")
    if not model:
        raise RuntimeError("LLM 모드에는 OPENAI_EXPERIMENT_MODEL 또는 OPENAI_CHAT_MODEL이 필요합니다")
    body = {
        "model": model,
        "store": False,
        "max_output_tokens": 2200,
        "text": {"format": {"type": "json_schema", "name": "vessel_search_policies",
                            "strict": True, "schema": OUTPUT_SCHEMA}},
        "input": [
            {"role": "system", "content": "You design bounded, locally testable retrieval policies. Return JSON only."},
            {"role": "user", "content": STATIC_PROMPT},
        ],
    }
    request = urllib.request.Request(
        ENDPOINT, data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=180) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"OpenAI API HTTP {exc.code}; 응답 본문은 기록하지 않았습니다") from None
    except urllib.error.URLError as exc:
        raise RuntimeError(f"OpenAI API 연결 실패: {exc.reason}") from None
    if payload.get("status") != "completed":
        raise RuntimeError(f"LLM 응답이 완료되지 않았습니다: {payload.get('status')}")
    text_parts = [content.get("text", "") for item in payload.get("output", [])
                  if item.get("type") == "message" for content in item.get("content", [])
                  if content.get("type") == "output_text"]
    if not text_parts:
        raise RuntimeError("LLM 응답에 정책 JSON이 없습니다")
    try:
        parsed = json.loads("\n".join(text_parts))
    except json.JSONDecodeError as exc:
        raise RuntimeError("LLM 응답 JSON을 읽을 수 없습니다") from exc
    if not isinstance(parsed, dict) or set(parsed) != {"proposals"}:
        raise ValueError("LLM 응답 최상위 필드가 올바르지 않습니다")
    proposals = validate_proposals(parsed["proposals"])
    return {
        "planner": "openai_responses_json_schema",
        "model": model,
        "response_id": payload.get("id"),
        "usage": payload.get("usage", {}),
        "transmitted_data": "static task description and policy schema only; no competition rows, labels, or metrics",
        "proposals": proposals,
    }
