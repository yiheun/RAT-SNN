"""Input construction and trial-balanced electrode-band standardization.

Estimate first/second moments from original frames, never overlapping windows.
Labels use zero-based indices 0,...,K-1, corresponding to 1,...,K in the paper.
"""
from dataclasses import dataclass
import numpy as np
from .settings import STD_EPSILON, CLIPPING_BOUND, WINDOW_LENGTH, WINDOW_STRIDE


@dataclass(frozen=True)
class DETrial:
    subject: int
    session: int
    trial: int
    label: int
    frames: np.ndarray  # [L,62,5], ordered according to electrode_metadata.py

    @property
    def identity(self):
        return self.subject, self.session, self.trial


@dataclass(frozen=True)
class WindowTrial:
    subject: int
    session: int
    trial: int
    label: int
    windows: np.ndarray  # [N,4,62,5]; each window belongs to this trial only

    @property
    def identity(self):
        return self.subject, self.session, self.trial


def validate_trials(trials):
    if not trials or len({t.identity for t in trials}) != len(trials):
        raise ValueError("Trials must be nonempty and have unique subject-session-trial identities")
    for t in trials:
        x = np.asarray(t.frames)
        if x.ndim != 3 or x.shape[1:] != (62, 5) or len(x) < WINDOW_LENGTH:
            raise ValueError("Each trial must contain at least four frames shaped [L,62,5]")
        if not np.isfinite(x).all() or t.label < 0:
            raise ValueError("DE values must be finite and class labels nonnegative")


@dataclass(frozen=True)
class TrialBalancedStandardizer:
    mean: np.ndarray
    std: np.ndarray
    epsilon: float = STD_EPSILON
    clip: float = CLIPPING_BOUND

    @classmethod
    def fit(cls, training_trials, epsilon=STD_EPSILON, clip=CLIPPING_BOUND):
        validate_trials(training_trials)
        if epsilon <= 0 or clip <= 0:
            raise ValueError("epsilon and clip must be positive")
        frames = [np.asarray(t.frames, dtype=np.float64) for t in training_trials]
        mean = np.stack([x.mean(axis=0) for x in frames]).mean(axis=0)
        second = np.stack([(x * x).mean(axis=0) for x in frames]).mean(axis=0)
        std = np.sqrt(np.maximum(second - mean * mean, epsilon ** 2))
        return cls(mean, std, float(epsilon), float(clip))

    def transform(self, frames):
        x = np.asarray(frames, dtype=np.float64)
        return np.clip((x - self.mean) / self.std, -self.clip, self.clip).astype(np.float32)


def construct_trial_windows(standardized_frames):
    """T=4, stride=1; no padding, no windows crossing trial boundaries."""
    x = np.asarray(standardized_frames, dtype=np.float32)
    if x.ndim != 3 or x.shape[1:] != (62, 5) or len(x) < WINDOW_LENGTH:
        raise ValueError("Expected at least four frames with shape [L,62,5]")
    view = np.lib.stride_tricks.sliding_window_view(x, WINDOW_LENGTH, axis=0)
    return np.moveaxis(view, -1, 1)[::WINDOW_STRIDE].copy()


def prepare_window_trials(trials, standardizer):
    validate_trials(trials)
    return [WindowTrial(t.subject, t.session, t.trial, t.label,
                        construct_trial_windows(standardizer.transform(t.frames))) for t in trials]


def prepare_source_units(source_trials):
    """Personalized source pool: fit separately for each source subject-session."""
    validate_trials(source_trials)
    units = sorted({(t.subject, t.session) for t in source_trials})
    statistics, windows = {}, []
    for unit in units:
        subset = [t for t in source_trials if (t.subject, t.session) == unit]
        statistics[unit] = TrialBalancedStandardizer.fit(subset)
        windows.extend(prepare_window_trials(subset, statistics[unit]))
    return windows, statistics


def prepare_training_and_evaluation(training_trials, evaluation_trials):
    """Fit only supplied training trials, then apply unchanged to evaluation.

Use for each target adaptation/validation split; for LOSO, supply all
inner-training (or final source-training) subjects as the training trials.
"""
    if {t.identity for t in training_trials} & {t.identity for t in evaluation_trials}:
        raise ValueError("Training and evaluation trials must be disjoint")
    statistics = TrialBalancedStandardizer.fit(training_trials)
    return (prepare_window_trials(training_trials, statistics),
            prepare_window_trials(evaluation_trials, statistics), statistics)
