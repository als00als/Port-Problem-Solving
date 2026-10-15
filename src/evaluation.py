from src.candidate_generation import generate_fuzzy_candidates_with_scores
from src.ranking import rank_candidates_with_fuzzy_score

import time


def evaluate_model(valid_df, candidate_generator, ranker):
    predictions = []
    candidate_lists = []

    true_ids = valid_df["target_vessel_id"].astype(str).tolist()
    vessel_names = valid_df["vessel_name"].tolist()

    started_at = time.perf_counter()

    for _, row in valid_df.iterrows():
        candidate_indices, candidate_metadata = candidate_generator(row)

        ranked = ranker(
            row=row,
            candidate_indices=candidate_indices,
            candidate_metadata=candidate_metadata,
        )

        candidate_lists.append(
            [item["vessel_id"] for item in ranked]
        )

        predictions.append(
            ranked[0]["vessel_id"] if ranked else None
        )

    return {
        "true_ids": true_ids,
        "predictions": predictions,
        "vessel_names": vessel_names,
        "candidate_lists": candidate_lists,
        "elapsed_seconds": time.perf_counter() - started_at,
    }