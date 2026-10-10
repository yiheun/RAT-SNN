"""Fixed trial partitions, personalized source exclusions and session-wise LOSO."""
from dataclasses import dataclass
from .settings import DATASETS


@dataclass(frozen=True)
class TrialPartition:
    training: tuple
    testing: tuple
    validation_folds: tuple  # ((adaptation_trials, validation_trials), ...)
    test_block: int | None = None


def personalized_partitions(dataset, target_trials):
    spec = DATASETS[dataset]
    trials = sorted(target_trials, key=lambda t: t.trial)
    if len({(t.subject, t.session) for t in trials}) != 1:
        raise ValueError("Supply one target subject-session unit")
    if [t.trial for t in trials] != list(range(1, spec.trials_per_session + 1)):
        raise ValueError("Complete chronological trial IDs are required")
    if any(t.label not in range(spec.classes) for t in trials):
        raise ValueError("Invalid dataset class labels")
    if dataset == "SEED-V":
        blocks = [tuple(trials[start:start + 5]) for start in (0, 5, 10)]
        if any({t.label for t in block} != set(range(5)) for block in blocks):
            raise ValueError("Each SEED-V block must contain one trial per class")
        result = []
        for held_out in range(3):
            a, b = [block for i, block in enumerate(blocks) if i != held_out]
            result.append(TrialPartition(a + b, blocks[held_out], ((a, b), (b, a)), held_out + 1))
        return result
    if dataset == "SEED":
        training, testing, folds = trials[:9], trials[9:], 3
    else:
        testing = [t for k in range(4) for t in [q for q in trials if q.label == k][-2:]]
        test_ids = {t.identity for t in testing}
        training, folds = [t for t in trials if t.identity not in test_ids], 4
    by_class = [[t for t in training if t.label == k] for k in range(spec.classes)]
    if any(len(pool) != folds for pool in by_class) or len(testing) != 2 * spec.classes:
        raise ValueError("Trial class counts do not match the prescribed evaluation split")
    validation = []
    for i in range(folds):
        val = tuple(pool[i] for pool in by_class)
        ids = {t.identity for t in val}
        adaptation = tuple(t for t in training if t.identity not in ids)
        validation.append((adaptation, val))
    return [TrialPartition(tuple(training), tuple(sorted(testing, key=lambda t: t.trial)), tuple(validation))]


def personalized_source_pool(all_trials, target_subject, target_session):
    # Sharing either index excludes a unit: both indices must differ.
    return [t for t in all_trials if t.subject != target_subject and t.session != target_session]


def session_loso_folds(dataset, all_trials, session):
    spec = DATASETS[dataset]
    trials = [t for t in all_trials if t.session == session]
    subjects = sorted({t.subject for t in trials})
    if len(subjects) != spec.subjects or any(sum(t.subject == s for t in trials) != spec.trials_per_session for s in subjects):
        raise ValueError("LOSO requires complete subjects/trials in the current session")
    for subject in subjects:
        yield subject, tuple(t for t in trials if t.subject != subject), tuple(t for t in trials if t.subject == subject)


def source_inner_loso_folds(source_trials):
    if len({t.session for t in source_trials}) != 1:
        raise ValueError("Inner LOSO is performed within the current session")
    subjects = sorted({t.subject for t in source_trials})
    for subject in subjects:
        yield (tuple(t for t in source_trials if t.subject != subject),
               tuple(t for t in source_trials if t.subject == subject))
