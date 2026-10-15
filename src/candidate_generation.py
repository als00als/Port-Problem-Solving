# import sys
# from pathlib import Path

# project_root = Path.cwd().parent
# sys.path.insert(0, str(project_root))

from src.normalize import normalize_name
from src.metric import is_korean

from rapidfuzz import fuzz, process
from unidecode import unidecode
import pandas as pd

def build_catalog_index(catalog):
    name_to_indices = {}

    for row_index, vessel_name in enumerate(catalog['vessel_name']):
        normalize = normalize_name(vessel_name)

        if not normalize:
            continue
        if normalize not in name_to_indices:
            name_to_indices[normalize] = []

        name_to_indices[normalize].append(row_index)

    return name_to_indices

def generate_candidates(row, catalog, index, max_candidates=30) -> list:
    # 입력 선박명을 normalize_name()으로 정규화
    input_name = normalize_name(row.get('vessel_name'))
    value= row.get('gross_tonnage')

    candidates = []
    used = set()

    def add_candidates(indices):
        for candidate_index in indices:
            if candidate_index not in used:
                candidates.append(candidate_index)
                used.add(candidate_index)

            if len(candidates) >= max_candidates:
                break
            
    name_candidates = index.get(input_name, [])
    add_candidates(name_candidates)

    try:
        tonnage = float(value)
    except (TypeError, ValueError):
        tonnage = None

    if tonnage is not None and tonnage > 0:
        catalog_tonnage = pd.to_numeric(catalog['gross_tonnage'], errors='coerce')
        gaps = (catalog_tonnage - tonnage).abs()
        nearest_indices = (gaps.dropna().nsmallest(10).index.tolist())
        add_candidates(nearest_indices)

    return candidates[:max_candidates]

def romanized_name(value) -> str:
    normalized = normalize_name(value)

    if not normalized:
        return ""

    return normalize_name(unidecode(normalized))


def generate_fuzzy_candidates_with_scores(
    row,
    catalog,
    catalog_index,
    romanized_catalog_names,
    fuzzy_limit=10,
    fuzzy_cutoff=60,
    max_candidates=30,
):
    base_indices = generate_candidates(
        row=row,
        catalog=catalog,
        index=catalog_index,
        max_candidates=max_candidates,
    )

    fuzzy_indices = []
    fuzzy_scores = {}

    if is_korean(row.get("vessel_name")):
        query_key = romanized_name(row.get("vessel_name"))

        if query_key:
            matches = process.extract(
                query_key,
                romanized_catalog_names,
                scorer=fuzz.ratio,
                limit=fuzzy_limit,
                score_cutoff=fuzzy_cutoff,
            )

            for _, score, candidate_index in matches:
                candidate_index = int(candidate_index)
                fuzzy_indices.append(candidate_index)
                fuzzy_scores[candidate_index] = float(score) / 100.0

    merged_indices = []
    seen = set()

    for candidate_index in fuzzy_indices + base_indices:
        if candidate_index not in seen:
            merged_indices.append(candidate_index)
            seen.add(candidate_index)

        if len(merged_indices) >= max_candidates:
            break

    return merged_indices, fuzzy_scores