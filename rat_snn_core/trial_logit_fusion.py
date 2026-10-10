"""Within-trial mean logits; temperature-scaled LOGIT fusion and OOF selection."""
from dataclasses import dataclass
import numpy as np
import torch
from .settings import REGIONAL_FUSION_WEIGHTS, FUSION_TEMPERATURES
from .evaluation_metrics import trial_metrics, negative_log_likelihood


@dataclass(frozen=True)
class FusionParameters:
    regional_weight: float
    global_temperature: float
    regional_temperature: float

    def __post_init__(self):
        if not 0 <= self.regional_weight <= 1 or min(self.global_temperature, self.regional_temperature) <= 0:
            raise ValueError("Invalid fusion weight or temperature")


def fuse_trial_logits(global_logits, regional_logits, parameters):
    if global_logits.shape != regional_logits.shape:
        raise ValueError("Pathways must predict the same ordered trials/classes")
    p = parameters
    return ((1.0 - p.regional_weight) * global_logits / p.global_temperature
            + p.regional_weight * regional_logits / p.regional_temperature)


@torch.no_grad()
def predict_trial_logits(model, window_trials, device="cpu", inference_batch_size=256):
    """Every eligible window contributes once, regardless of training sampling.

    Inference chunk size only controls memory; trials are aggregated by sums and
    counts, so a short final chunk receives its correct window weight.
    """
    if inference_batch_size < 1 or not window_trials:
        raise ValueError("Nonempty trials and positive inference chunk size required")
    was_training = model.training
    model.eval()
    logits = []
    try:
        for trial in window_trials:
            total, count = None, 0
            for start in range(0, len(trial.windows), inference_batch_size):
                x = torch.as_tensor(trial.windows[start:start + inference_batch_size], device=device)
                prediction = model(x)
                chunk_sum = prediction.double().sum(0)
                total = chunk_sum if total is None else total + chunk_sum
                count += len(prediction)
            if not count:
                raise ValueError("Trial has no eligible windows")
            logits.append((total / count).cpu().numpy())
    finally:
        model.train(was_training)
    return np.stack(logits)


def select_fusion(global_oof_logits, regional_oof_logits, labels, trial_identities,
                  expected_trial_identities, variant="temperature_scaled"):
    """Each allowed validation trial appears exactly once in concatenated OOF data.

    Invoke independently for each target unit/run (SEED-V: also each test block),
    or for each outer LOSO fold/run. Candidate order is lambda, Tg, Tr ascending.
    """
    identities, expected = list(trial_identities), set(expected_trial_identities)
    if len(identities) != len(set(identities)) or set(identities) != expected:
        raise ValueError("OOF predictions must cover each allowed training trial exactly once")
    g, r, labels = np.asarray(global_oof_logits), np.asarray(regional_oof_logits), np.asarray(labels, dtype=int)
    if g.ndim != 2 or g.shape != r.shape or len(g) != len(labels) or len(g) != len(identities):
        raise ValueError("Invalid OOF shapes")
    if not np.isfinite(g).all() or not np.isfinite(r).all():
        raise ValueError("Nonfinite logits")
    if variant == "temperature_scaled":
        weights, temperatures = REGIONAL_FUSION_WEIGHTS, FUSION_TEMPERATURES
    elif variant == "weighted_without_temperature":
        weights, temperatures = REGIONAL_FUSION_WEIGHTS, (1.0,)
    elif variant == "equal_logits":
        weights, temperatures = (0.5,), (1.0,)
    else:
        raise ValueError(variant)
    best = None
    for weight in weights:
        for tg in temperatures:
            for tr in temperatures:
                parameters = FusionParameters(weight, tg, tr)
                fused = fuse_trial_logits(g, r, parameters)
                metrics = trial_metrics(labels, fused.argmax(1), fused.shape[1])
                nll = negative_log_likelihood(fused, labels)
                rank = (metrics["macro_f1"], metrics["accuracy"], -nll,
                        -(abs(tg - 1) + abs(tr - 1)), -abs(weight - 0.5))
                if best is None or rank > best[0]:
                    best = rank, parameters, {**metrics, "nll": nll}
    return best[1], best[2]
