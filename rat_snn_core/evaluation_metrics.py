"""Trial-level scores; average sessions/runs per subject before across-subject SD."""
import numpy as np


def confusion_counts(labels, predictions, n_classes):
    labels, predictions = np.asarray(labels, dtype=int), np.asarray(predictions, dtype=int)
    if labels.shape != predictions.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("Expected nonempty matching trial-level label vectors")
    if min(labels.min(), predictions.min()) < 0 or max(labels.max(), predictions.max()) >= n_classes:
        raise ValueError("Class index outside 0,...,K-1")
    return np.bincount(n_classes * labels + predictions, minlength=n_classes ** 2).reshape(n_classes, n_classes)


def trial_metrics(labels, predictions, n_classes):
    cm = confusion_counts(labels, predictions, n_classes)
    tp = np.diag(cm)
    denominator = cm.sum(0) + cm.sum(1)
    f1 = np.divide(2 * tp, denominator, out=np.zeros(n_classes, dtype=float), where=denominator != 0)
    return {"accuracy": float(tp.sum() / cm.sum()), "macro_f1": float(f1.mean()), "confusion_matrix": cm}


def negative_log_likelihood(logits, labels):
    z = np.asarray(logits, dtype=np.float64)
    y = np.asarray(labels, dtype=int)
    shifted = z - z.max(1, keepdims=True)
    return float((np.log(np.exp(shifted).sum(1)) - shifted[np.arange(len(y)), y]).mean())


def aggregate_subject_scores(records):
    """records: iterable of dict(subject, session, run, accuracy, macro_f1).

    Supply each subject-session-run once, with 3 sessions and 10 distinct runs.
    SEED-V's three test blocks must already be pooled at the prediction level.
    Metrics returned in [0,1]; multiply by 100 for paper tables.
    """
    records = list(records)
    keys = [(r["subject"], r["session"], r["run"]) for r in records]
    if len(set(keys)) != len(keys):
        raise ValueError("Duplicate subject-session-run metrics")
    subjects = sorted({r["subject"] for r in records})
    if len(subjects) < 2:
        raise ValueError("Across-subject SD requires at least two subjects")
    output = {}
    for metric in ("accuracy", "macro_f1"):
        scores = []
        for subject in subjects:
            subset = [r for r in records if r["subject"] == subject]
            sessions = {r["session"] for r in subset}
            runs = {r["run"] for r in subset}
            if len(sessions) != 3 or len(runs) != 10 or len(subset) != 30:
                raise ValueError("Each subject must have all 3 sessions x 10 runs")
            scores.append(np.mean([r[metric] for r in subset]))
        output[metric] = {"mean": float(np.mean(scores)), "std": float(np.std(scores, ddof=1)),
                          "per_subject": dict(zip(subjects, map(float, scores)))}
    return output


def row_normalized_confusion(pooled_counts):
    counts = np.asarray(pooled_counts, dtype=float)
    denominator = counts.sum(1, keepdims=True)
    return np.divide(100 * counts, denominator, out=np.zeros_like(counts), where=denominator != 0)
