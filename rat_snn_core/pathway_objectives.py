"""Window/trial classification, Global firing rate, normalized rehearsal loss."""
import torch
from torch.nn import functional as F
from .settings import (LABEL_SMOOTHING, TRIAL_LOSS_WEIGHT, TARGET_FIRING_RATE,
                       FIRING_RATE_WEIGHT, SOURCE_REHEARSAL_WEIGHT)


def classification_objective(logits, labels, parent_trial_ids):
    if len(logits) != len(labels) or len(labels) != len(parent_trial_ids):
        raise ValueError("Every sampled window needs a label and parent trial ID")
    window_loss = F.cross_entropy(logits, labels, label_smoothing=LABEL_SMOOTHING)
    trial_logits, trial_labels = [], []
    for identity in torch.unique(parent_trial_ids, sorted=True):
        mask = parent_trial_ids == identity
        if not torch.all(labels[mask] == labels[mask][0]):
            raise ValueError("A parent trial must have a single label")
        # Repeated sampled windows are separate occurrences in this batch mean.
        trial_logits.append(logits[mask].mean(0))
        trial_labels.append(labels[mask][0])
    trial_loss = F.cross_entropy(torch.stack(trial_logits), torch.stack(trial_labels),
                                 label_smoothing=LABEL_SMOOTHING)
    return window_loss + TRIAL_LOSS_WEIGHT * trial_loss


def pathway_objective(details, labels, parent_trial_ids, pathway):
    loss = classification_objective(details["main_logits"], labels, parent_trial_ids)
    if pathway == "global":
        loss = loss + FIRING_RATE_WEIGHT * (details["firing_rates"] - TARGET_FIRING_RATE).square().mean()
    elif pathway != "regional":
        raise ValueError(pathway)
    return loss


def rehearsal_objective(target_loss, source_loss, weight=SOURCE_REHEARSAL_WEIGHT):
    if weight < 0:
        raise ValueError("Rehearsal weight cannot be negative")
    return (target_loss + weight * source_loss) / (1.0 + weight)
