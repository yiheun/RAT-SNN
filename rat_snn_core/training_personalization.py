"""Independent AdamW updates, source initialization, final-epoch early stopping."""
import copy
import math
import random
import numpy as np
import torch
from .settings import (TRAINING_BUDGETS, WEIGHT_DECAY, GRADIENT_CLIPPING,
                       CROSS_SUBJECT_MAX_EPOCHS, EARLY_STOPPING_PATIENCE)
from .pathway_objectives import pathway_objective, rehearsal_objective


def seed_random_generators(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_optimizer(model, pathway, adaptation=False):
    budget = TRAINING_BUDGETS[pathway]
    lr = budget.adaptation_learning_rate if adaptation else budget.pretraining_learning_rate
    return torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
                             lr=lr, weight_decay=WEIGHT_DECAY)


def copy_pretrained_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def initialize_from_source(model, source_state):
    """Call before EVERY validation fold and again before final adaptation."""
    model.load_state_dict(copy.deepcopy(source_state), strict=True)
    return model


def optimization_step(model, optimizer, target_batch, pathway, source_batch=None):
    model.train()
    optimizer.zero_grad(set_to_none=True)
    target_loss = pathway_objective(model(target_batch.windows, return_details=True),
                                    target_batch.labels, target_batch.parent_trial_ids, pathway)
    if source_batch is not None:
        source_loss = pathway_objective(model(source_batch.windows, return_details=True),
                                        source_batch.labels, source_batch.parent_trial_ids, pathway)
        loss = rehearsal_objective(target_loss, source_loss)
    else:
        loss = target_loss
    if not torch.isfinite(loss):
        raise FloatingPointError("Nonfinite training loss")
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), GRADIENT_CLIPPING)
    optimizer.step()
    return float(loss.detach())


@torch.no_grad()
def validation_objective(model, window_trials, pathway, device="cpu"):
    """Evaluate the same pathway objective on ALL validation windows.

    Window CE is weighted by window count; trial CE equally weights trials.
    Global firing-rate means are pooled over all windows before penalization.
    Validation subject data never enter optimizer updates or fitted statistics.
    """
    from torch.nn import functional as F
    from .settings import LABEL_SMOOTHING, TRIAL_LOSS_WEIGHT, FIRING_RATE_WEIGHT, TARGET_FIRING_RATE
    was_training = model.training
    model.eval()
    window_ce, count, trial_logits, labels, rate_sum = 0.0, 0, [], [], None
    try:
        for trial in window_trials:
            # Process one complete trial so grouping follows the manuscript loss.
            x = torch.as_tensor(trial.windows, device=device)
            details = model(x, return_details=True)
            logits = details["main_logits"]
            y = torch.full((len(x),), trial.label, device=device, dtype=torch.long)
            window_ce += F.cross_entropy(logits, y, label_smoothing=LABEL_SMOOTHING, reduction="sum")
            count += len(x)
            trial_logits.append(logits.mean(0))
            labels.append(trial.label)
            if pathway == "global":
                rates = details["firing_rates"] * len(x)
                rate_sum = rates if rate_sum is None else rate_sum + rates
        if not count:
            raise ValueError("Validation trials cannot be empty")
        loss = window_ce / count + TRIAL_LOSS_WEIGHT * F.cross_entropy(
            torch.stack(trial_logits), torch.tensor(labels, device=device), label_smoothing=LABEL_SMOOTHING)
        if pathway == "global":
            loss = loss + FIRING_RATE_WEIGHT * (rate_sum / count - TARGET_FIRING_RATE).square().mean()
        return float(loss)
    finally:
        model.train(was_training)


def train_fixed_epochs(train_epoch, epochs):
    """train_epoch(epoch_number) updates the current model; no checkpoint restore."""
    if epochs < 1:
        raise ValueError("Epoch count must be positive")
    return [train_epoch(epoch) for epoch in range(1, epochs + 1)]


def train_cross_subject_inner(train_epoch, evaluate_validation,
                              max_epochs=CROSS_SUBJECT_MAX_EPOCHS,
                              patience=EARLY_STOPPING_PATIENCE):
    """Stop after 30 consecutive epochs without a STRICT validation-loss decrease.

    The callback owns the model. Its final completed epoch weights are retained;
    this function deliberately never restores the minimum-loss checkpoint.
    """
    if max_epochs < 1 or patience < 1:
        raise ValueError("Maximum epochs and patience must be positive")
    best_loss, stale, history = math.inf, 0, []
    for epoch in range(1, max_epochs + 1):
        train_epoch(epoch)
        loss = float(evaluate_validation())
        if not math.isfinite(loss):
            raise FloatingPointError("Nonfinite validation loss")
        history.append(loss)
        if loss < best_loss:
            best_loss, stale = loss, 0
        else:
            stale += 1
        if stale >= patience:
            break
    return {"completed_epochs": len(history), "validation_losses": history,
            "final_validation_loss": history[-1]}


def final_source_retraining_epochs(inner_completed_epochs):
    values = np.asarray(inner_completed_epochs, dtype=int)
    if values.ndim != 1 or not len(values) or np.any((values < 1) | (values > 100)):
        raise ValueError("Supply completed epoch counts from all corresponding inner folds")
    return int(math.ceil(float(np.median(values))))
