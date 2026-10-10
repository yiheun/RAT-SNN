"""Matched Global source-retention comparison on previously available source data."""
import copy
import numpy as np
from .personalized_evaluation import adapt_pathway
from .trial_logit_fusion import predict_trial_logits
from .evaluation_metrics import trial_metrics
from .training_personalization import seed_random_generators


def compare_source_retention(pretrained_global_model, source_state, target_training_windows,
                              target_test_windows, source_windows, audit_windows, seed,
                              device="cpu", steps_per_epoch=None):
    """SEED-IV Global-only controlled comparison; caller supplies a FIXED audit set.

    Audit trials belong to the source pool and are deliberately not removed from
    rehearsal eligibility. This measures retention, not unseen-source generalization.
    Both conditions use the same initialization, target split and sampling seed.
    """
    if not audit_windows or not {t.identity for t in audit_windows} <= {t.identity for t in source_windows}:
        raise ValueError("The fixed audit subset must belong to the available source pool")
    if len({t.identity for t in audit_windows}) != len(audit_windows):
        raise ValueError("Audit trials must be unique")
    counts = [sum(t.label == k for t in audit_windows) for k in range(4)]
    if min(counts) < 1 or len(set(counts)) != 1:
        raise ValueError("SEED-IV audit subset must contain equal trial counts for all four classes")
    audit_labels = [t.label for t in audit_windows]
    target_labels = [t.label for t in target_test_windows]
    output = {}
    for rehearsal in (False, True):
        seed_random_generators(seed)
        model = copy.deepcopy(pretrained_global_model).to(device)
        model.load_state_dict(source_state)
        before = trial_metrics(audit_labels, predict_trial_logits(model, audit_windows, device).argmax(1), 4)
        adapt_pathway(model, source_state, target_training_windows, source_windows,
                      "global", 4, np.random.default_rng(seed), device, rehearsal, steps_per_epoch)
        after = trial_metrics(audit_labels, predict_trial_logits(model, audit_windows, device).argmax(1), 4)
        target = trial_metrics(target_labels, predict_trial_logits(model, target_test_windows, device).argmax(1), 4)
        output["with_rehearsal" if rehearsal else "without_rehearsal"] = {
            "source_before": before, "source_after": after, "target": target,
            "source_accuracy_change": after["accuracy"] - before["accuracy"]}
    return output
