"""Ten independent runs with fixed partitions; never ensemble run predictions."""
import numpy as np
from .settings import RANDOM_REPETITIONS
from .evaluation_metrics import aggregate_subject_scores, row_normalized_confusion


def evaluate_ten_runs(evaluate_run, random_seeds):
    """evaluate_run(seed) returns one result per subject-session for that seed.

    Use the same ten distinct seeds for all units and controlled variants. Supply
    actual experiment seeds explicitly; the manuscript does not list their values.
    Every result must already pool SEED-V blocks within that unit/run.
    """
    seeds = tuple(random_seeds)
    if len(seeds) != RANDOM_REPETITIONS or len(set(seeds)) != RANDOM_REPETITIONS:
        raise ValueError("Supply exactly ten distinct random seeds")
    records = []
    for seed in seeds:
        results = list(evaluate_run(seed))
        if any(result["run"] != seed for result in results):
            raise ValueError("Run IDs must equal the supplied random seed")
        records.extend(results)
    counts = np.sum([result["confusion_matrix"] for result in records], axis=0)
    return {"records": records, "scores": aggregate_subject_scores(records),
            "pooled_confusion_counts": counts, "row_normalized_confusion_percent": row_normalized_confusion(counts)}
