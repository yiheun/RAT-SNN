"""Source-only inner LOSO, independent duration selection, and outer evaluation."""
import numpy as np
from .component_ablations import build_pathways
from .data_standardization import prepare_training_and_evaluation
from .evaluation_protocols import source_inner_loso_folds
from .settings import DATASETS
from .trial_batch_sampling import SourceTrialSampler
from .training_personalization import (seed_random_generators, make_optimizer, optimization_step,
                                      validation_objective, train_cross_subject_inner,
                                      final_source_retraining_epochs)
from .trial_logit_fusion import predict_trial_logits, select_fusion, fuse_trial_logits
from .evaluation_metrics import trial_metrics


def evaluate_cross_subject_fold(dataset, source_trials, test_trials, seed, device="cpu",
                                variant="full", source_steps_per_epoch=None,
                                fusion_variant="temperature_scaled"):
    """One outer fold/run. Both pathways share partitions and fitted statistics.

    Final retraining starts with a fresh initialization and includes every source
    subject, for ceil(median(completed inner epochs)) separately per pathway.
    No target-subject fitting, tuning, adaptation or early stopping is performed.
    """
    spec = DATASETS[dataset]
    source_subjects = {t.subject for t in source_trials}
    test_subjects = {t.subject for t in test_trials}
    if (len(source_subjects) != spec.subjects - 1 or len(test_subjects) != 1
            or source_subjects & test_subjects
            or len({t.session for t in (*source_trials, *test_trials)}) != 1
            or len(test_trials) != spec.trials_per_session
            or any(sum(t.subject == s for t in source_trials) != spec.trials_per_session for s in source_subjects)):
        raise ValueError("Supply complete disjoint source/test subjects in one outer session fold")
    seed_random_generators(seed)
    rng = np.random.default_rng(seed)
    classes = spec.classes
    oof_logits, completed = {p: [] for p in ("global", "regional")}, {p: [] for p in ("global", "regional")}
    oof_ids, oof_labels = [], []
    for training, validation in source_inner_loso_folds(source_trials):
        training_windows, validation_windows, _ = prepare_training_and_evaluation(training, validation)
        for pathway, model in build_pathways(classes, variant).items():
            model.to(device)
            optimizer = make_optimizer(model, pathway)
            sampler = SourceTrialSampler(training_windows, classes, rng)

            def train_epoch(epoch):
                for batch in sampler.pretraining_epoch(pathway, source_steps_per_epoch):
                    optimization_step(model, optimizer, batch.to(device), pathway)

            report = train_cross_subject_inner(train_epoch,
                lambda: validation_objective(model, validation_windows, pathway, device))
            completed[pathway].append(report["completed_epochs"])
            oof_logits[pathway].append(predict_trial_logits(model, validation_windows, device))
        oof_ids.extend(t.identity for t in validation)
        oof_labels.extend(t.label for t in validation)
    fusion, validation_metrics = select_fusion(
        np.concatenate(oof_logits["global"]), np.concatenate(oof_logits["regional"]),
        oof_labels, oof_ids, [t.identity for t in source_trials], fusion_variant)
    epochs = {p: final_source_retraining_epochs(values) for p, values in completed.items()}
    training_windows, test_windows, statistics = prepare_training_and_evaluation(source_trials, test_trials)
    test_logits = {}
    for pathway, model in build_pathways(classes, variant).items():
        model.to(device)
        optimizer = make_optimizer(model, pathway)
        sampler = SourceTrialSampler(training_windows, classes, rng)
        for _ in range(epochs[pathway]):
            for batch in sampler.pretraining_epoch(pathway, source_steps_per_epoch):
                optimization_step(model, optimizer, batch.to(device), pathway)
        test_logits[pathway] = predict_trial_logits(model, test_windows, device)
    fused = fuse_trial_logits(test_logits["global"], test_logits["regional"], fusion)
    labels = np.asarray([t.label for t in test_trials])
    predictions = fused.argmax(1)
    return {"subject": next(iter(test_subjects)), "session": test_trials[0].session, "run": seed,
            "fusion": fusion, "validation_metrics": validation_metrics, "retraining_epochs": epochs,
            "inner_completed_epochs": completed, "source_standardizer": statistics,
            "trial_identities": [t.identity for t in test_trials], "labels": labels,
            "predictions": predictions, "global_logits": test_logits["global"],
            "regional_logits": test_logits["regional"], "fused_logits": fused,
            **trial_metrics(labels, predictions, classes)}
