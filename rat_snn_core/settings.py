"""Numerical settings from RAT-SNN Methods and Experimental setup.

Global/Regional models are trained independently. Batch size depends on the
number of selected trials; it is not a fixed 64 windows.
"""
from dataclasses import dataclass

# Final manuscript used for equation/setting alignment; no experimental scores
# or trained weights are embedded in this core implementation.
MANUSCRIPT_SHA256 = "047a90ce22adb77f036bce2ad06783ba19ad420d5b865a54fa8cb0aa91dabeb4"

WINDOW_LENGTH = 4
WINDOW_STRIDE = 1
WINDOWS_PER_TRIAL = 4
N_CHANNELS = 62
N_BANDS = 5
STD_EPSILON = 1e-6
CLIPPING_BOUND = 8.0
RMS_EPSILON = 1e-8
BAND_RECALIBRATION_HIDDEN = 8
GLOBAL_DIMENSIONS = (64, 32, 96, 96, 32)
GLOBAL_THRESHOLDS = (1.0, 0.35, 1.0, 1.0, 1.0)
GLOBAL_SURROGATE_SCALE = 2.0
FRONT_END_DROPOUT = 0.10
TEMPORAL_DROPOUT = 0.15
REGIONAL_BAND_DIMENSION = 8
REGIONAL_HIDDEN_DIMENSION = 64
REGIONAL_BETA_MIN = 0.10
REGIONAL_BETA_MAX = 0.99
REGIONAL_INITIAL_BETA = 0.70
REGIONAL_INITIAL_SOFTPLUS_THRESHOLD = 0.75
REGIONAL_THRESHOLD_EPSILON = 1e-4
REGIONAL_SURROGATE_SCALE = 4.0
LABEL_SMOOTHING = 0.05
TRIAL_LOSS_WEIGHT = 0.5
FIRING_RATE_WEIGHT = 0.005
TARGET_FIRING_RATE = 0.15
SOURCE_REHEARSAL_WEIGHT = 0.5
WEIGHT_DECAY = 1e-3
GRADIENT_CLIPPING = 5.0
RANDOM_REPETITIONS = 10
CROSS_SUBJECT_MAX_EPOCHS = 100
EARLY_STOPPING_PATIENCE = 30
REGIONAL_FUSION_WEIGHTS = tuple(0.05 + 0.025 * j for j in range(37))
FUSION_TEMPERATURES = (0.8, 1.0, 1.2, 1.4)


@dataclass(frozen=True)
class TrainingBudget:
    pretraining_epochs: int
    adaptation_epochs: int
    source_trials_per_class: int
    pretraining_learning_rate: float
    adaptation_learning_rate: float


TRAINING_BUDGETS = {
    "global": TrainingBudget(8, 12, 4, 3e-4, 3e-4),
    "regional": TrainingBudget(4, 6, 3, 1e-3, 3e-4),
}


@dataclass(frozen=True)
class DatasetSpecification:
    classes: int
    subjects: int
    sessions: int
    trials_per_session: int


DATASETS = {
    "SEED": DatasetSpecification(3, 15, 3, 15),
    "SEED-IV": DatasetSpecification(4, 15, 3, 24),
    "SEED-V": DatasetSpecification(5, 16, 3, 15),
}
