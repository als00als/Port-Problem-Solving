"""Validated, declarative candidate-search policies proposed by an LLM.

The policy language combines existing deterministic search primitives. It
cannot name files, import modules, run Python, or inspect validation labels.
"""

from __future__ import annotations

import re
from typing import Any

import numpy as np
from rapidfuzz import fuzz, process


QUERY_FORMS = ("translated", "romanized", "english_variant", "learned", "skeleton", "tokens")
CATALOG_FIELDS = ("compact", "alpha", "skeleton", "tokens")
SCORERS = {"ratio": fuzz.ratio, "weighted_ratio": fuzz.WRatio,
           "token_set_ratio": fuzz.token_set_ratio}
CONDITIONS = ("all", "korean", "non_korean")
MAX_PROPOSALS = 2
MAX_STAGES = 2
MAX_EXTRA_CANDIDATES = 120

# What each primitive actually does. A stage whose name promises something the
# primitive cannot deliver is rejected instead of silently falling back.
QUERY_FORM_SEMANTICS = {
    "translated": "dictionary/rule translation, tokens joined without spaces",
    "romanized": "Unidecode romanization, tokens joined without spaces",
    "english_variant": "Korean rows only: English-style jamo romanization variant; "
                       "non-Korean rows have no variant and the stage is skipped",
    "learned": "Korean rows only: top learned syllable transliteration; otherwise skipped",
    "skeleton": "consonant skeleton of the romanized name",
    "tokens": "space-separated normalized tokens (keeps word boundaries and order)",
}
CATALOG_FIELD_SEMANTICS = {
    "compact": "normalized tokens joined without spaces",
    "alpha": "compact name with digit tokens removed (no spaces)",
    "skeleton": "consonant skeleton (no spaces)",
    "tokens": "space-separated normalized tokens",
}
KOREAN_ONLY_FORMS = {"english_variant", "learned"}
SPACED = {"tokens"}
TOKEN_SCORERS = {"token_set_ratio"}


def stage_consistency_errors(stage: dict) -> list[str]:
    """Reasons a stage would not do what its primitives claim (empty when consistent)."""
    errors = []
    if stage["query_form"] in KOREAN_ONLY_FORMS and stage["when"] == "non_korean":
        errors.append(f"{stage['query_form']}는 비한글 행에서 별도 변형을 만들지 않아 단계가 무효입니다")
    if stage["scorer"] in TOKEN_SCORERS and (stage["query_form"] not in SPACED
                                             or stage["catalog_field"] not in SPACED):
        errors.append("token_set_ratio는 공백으로 구분된 tokens 질의·카탈로그 필드에서만 토큰 순서 가설과 맞습니다")
    if (stage["query_form"] in SPACED) != (stage["catalog_field"] in SPACED):
        errors.append("공백 토큰 질의와 공백 없는 카탈로그 필드를 섞으면 비교 단위가 달라집니다")
    return errors


def validate_proposals(raw: Any) -> list[dict[str, Any]]:
    """Reject anything outside the small, executable search-policy language."""
    if not isinstance(raw, list) or not 1 <= len(raw) <= MAX_PROPOSALS:
        raise ValueError("LLM 제안은 1~2개여야 합니다")
    proposals: list[dict[str, Any]] = []
    names: set[str] = set()
    for proposal in raw:
        if not isinstance(proposal, dict) or set(proposal) != {"name", "hypothesis", "stages"}:
            raise ValueError("검색 정책 필드가 허용된 형식과 다릅니다")
        name = proposal["name"]
        hypothesis = proposal["hypothesis"]
        stages = proposal["stages"]
        if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9_]{2,39}", name):
            raise ValueError("검색 정책 이름은 영문 소문자 식별자여야 합니다")
        if name in names or name in {"v11_baseline", "vocab_top1", "vocab_top3"}:
            raise ValueError("검색 정책 이름이 중복되거나 기존 설정과 충돌합니다")
        names.add(name)
        if (not isinstance(hypothesis, str) or not 12 <= len(hypothesis) <= 350
                or any(ord(char) < 32 for char in hypothesis) or "|" in hypothesis):
            raise ValueError("가설 길이는 12~350자여야 합니다")
        if not isinstance(stages, list) or not 1 <= len(stages) <= MAX_STAGES:
            raise ValueError("검색 정책 단계는 1~2개여야 합니다")
        total_candidates = 0
        cleaned_stages = []
        for stage in stages:
            expected = {"when", "query_form", "catalog_field", "scorer", "top_k",
                        "min_similarity", "max_gt_gap"}
            if not isinstance(stage, dict) or set(stage) != expected:
                raise ValueError("검색 단계 필드가 허용된 형식과 다릅니다")
            if (not all(isinstance(stage[key], str) for key in
                        ("when", "query_form", "catalog_field", "scorer"))
                    or stage["when"] not in CONDITIONS or stage["query_form"] not in QUERY_FORMS
                    or stage["catalog_field"] not in CATALOG_FIELDS
                    or stage["scorer"] not in SCORERS):
                raise ValueError("허용되지 않은 검색 연산입니다")
            top_k = stage["top_k"]
            cutoff = stage["min_similarity"]
            gap = stage["max_gt_gap"]
            if not isinstance(top_k, int) or isinstance(top_k, bool) or not 5 <= top_k <= 60:
                raise ValueError("top_k는 5~60의 정수여야 합니다")
            if not isinstance(cutoff, (int, float)) or isinstance(cutoff, bool) or not 0 <= cutoff <= 100:
                raise ValueError("유사도 임계값은 0~100이어야 합니다")
            if gap is not None and (not isinstance(gap, (int, float)) or isinstance(gap, bool)
                                    or not 0 <= gap <= 1):
                raise ValueError("톤수 상대 차이 한도는 null 또는 0~1이어야 합니다")
            mismatch = stage_consistency_errors(stage)
            if mismatch:
                raise ValueError("; ".join(mismatch))
            total_candidates += top_k
            cleaned_stages.append({**stage, "min_similarity": float(cutoff),
                                   "max_gt_gap": None if gap is None else float(gap)})
        if total_candidates > MAX_EXTRA_CANDIDATES:
            raise ValueError("추가 후보 예산 120개를 넘었습니다")
        proposals.append({"name": name, "hypothesis": hypothesis, "stages": cleaned_stages})
    return proposals


def _query_value(query: dict, form: str) -> str | None:
    """Query string for one primitive, or None when the primitive does not apply.

    The 2026-10-19 run ``agent_v11_search_20261019_011012_4bed4e`` used an older
    version in which ``english_variant`` silently fell back to ``compact`` for
    non-Korean rows; that fallback was removed so a stage cannot claim a variant
    it never produced.
    """
    if form == "translated":
        return query["compact"]
    if form == "romanized":
        return query["rom_compact"]
    if form == "english_variant":
        return query["variants"][1] if query["hangul"] and len(query["variants"]) > 1 else None
    if form == "learned":
        return query["learned"][0] if query["hangul"] and query["learned"] else None
    if form == "tokens":
        return query["clean"]
    return query["skel_rom"]


def stage_applies(query: dict, stage: dict) -> bool:
    if stage["when"] == "korean" and not query["hangul"]:
        return False
    if stage["when"] == "non_korean" and query["hangul"]:
        return False
    return bool(_query_value(query, stage["query_form"]))


def behavior_probe(queries: list[dict], policy: dict) -> list[dict]:
    """Per stage: how many rows it runs on and how many query strings differ from compact."""
    out = []
    for stage in policy["stages"]:
        applied = [q for q in queries if stage_applies(q, stage)]
        distinct = sum(_query_value(q, stage["query_form"]) != q["compact"] for q in applied)
        out.append({"query_form": stage["query_form"], "when": stage["when"],
                    "applied_rows": len(applied), "query_differs_from_compact_rows": int(distinct)})
    return out


def _extra_indices(query: dict, index: Any, policy: dict, ml: Any) -> set[int]:
    choices = {"compact": index.compact, "alpha": index.alpha, "skeleton": index.skel,
               "tokens": index.clean}
    extra: set[int] = set()
    for stage in policy["stages"]:
        if not stage_applies(query, stage):
            continue
        value = _query_value(query, stage["query_form"])
        if not value:
            continue
        matches = process.extract(
            value, choices[stage["catalog_field"]], scorer=SCORERS[stage["scorer"]],
            limit=min(len(index.ids), stage["top_k"] * 5),
            score_cutoff=stage["min_similarity"],
        )
        added = 0
        query_gt = query["gross"] or query["intl"]
        for _, _, candidate in matches:
            if stage["max_gt_gap"] is not None and query_gt is not None:
                candidate_gt = ml.positive_float(index.gt[candidate])
                if candidate_gt is None or ml.relative_gap(query_gt, candidate_gt) > stage["max_gt_gap"]:
                    continue
            extra.add(int(candidate))
            added += 1
            if added >= stage["top_k"]:
                break
    return extra


def augment_candidates_and_features(queries: list[dict], index: Any, resources: dict,
                                    policy: dict, ml: Any,
                                    base_candidates: list[np.ndarray], base_features: np.ndarray
                                    ) -> tuple[list[np.ndarray], np.ndarray]:
    """Union policy retrieval with v11 candidates and recompute group features."""
    candidate_lists: list[np.ndarray] = []
    blocks: list[np.ndarray] = []
    offset = 0
    for query, base in zip(queries, base_candidates):
        base_block = base_features[offset:offset + len(base)]
        offset += len(base)
        extra = _extra_indices(query, index, policy, ml)
        union = np.asarray(sorted(set(int(x) for x in base) | extra), dtype=np.int64)
        candidate_lists.append(union)
        if len(union) == len(base):
            blocks.append(base_block)
            continue
        compact = [index.compact[i] for i in union]
        skeletons = [index.skel[i] for i in union]
        # The v11 baseline passes uint8 RapidFuzz scores to _pair_features.
        # Preserve that quantization so the comparison changes retrieval only.
        ratio = lambda value, choices: process.cdist([value], choices, scorer=fuzz.ratio,
                                                       dtype=np.uint8, workers=1)[0]
        blocks.append(ml._pair_features(query, union, index, resources,
                                        ratio(query["compact"], compact),
                                        ratio(query["skel_rom"], skeletons),
                                        ratio(query["skel_trans"], skeletons)
                                        if query["hangul"] else None))
    return candidate_lists, np.vstack(blocks).astype(np.float32)
