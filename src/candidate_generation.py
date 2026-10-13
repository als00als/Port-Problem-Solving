# import sys
# from pathlib import Path

# project_root = Path.cwd().parent
# sys.path.insert(0, str(project_root))

from src.normalize import normalize_name
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
