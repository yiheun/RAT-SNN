"""Core source-pretrain -> target OOF validation -> final adaptation procedure.

Accept DETrial objects; dataset file discovery/loading stays outside this module.
Models from the final completed epoch are used throughout. No test-driven tuning.
"""
import copy
import numpy as np
from .component_ablations import build_pathways
from .data_standardization import prepare_source_units, prepare_training_and_evaluation
from .evaluation_protocols import personalized_partitions, personalized_source_pool
from .settings import DATASETS, TRAINING_BUDGETS
from .trial_batch_sampling import SourceTrialSampler, TargetTrialSampler
from .training_personalization import (seed_random_generators, make_optimizer, optimization_step,
                                      copy_pretrained_state, initialize_from_source)
from .trial_logit_fusion import predict_trial_logits, select_fusion, fuse_trial_logits
from .evaluation_metrics import trial_metrics


def adapt_pathway(model, source_state, target_windows, source_windows, pathway, classes,
                  rng, device, rehearsal=True, steps_per_epoch=None):
    initialize_from_source(model, source_state)
    optimizer = make_optimizer(model, pathway, adaptation=True)
    # Independent sampling streams keep target windows matched in the rehearsal
    # comparison: replay draws cannot change subsequent target index cycles.
    target_rng = np.random.default_rng(rng.integers(0, 2 ** 63))
    source_rng = np.random.default_rng(rng.integers(0, 2 ** 63))
    target_sampler = TargetTrialSampler(target_windows, classes, target_rng)
    source_sampler = SourceTrialSampler(source_windows, classes, source_rng) if rehearsal else None
    for _ in range(TRAINING_BUDGETS[pathway].adaptation_epochs):
        for target, source in target_sampler.adaptation_epoch(source_sampler, steps_per_epoch):
            optimization_step(model, optimizer, target.to(device), pathway,
                              source.to(device) if source is not None else None)


def evaluate_personalized_unit(dataset, all_trials, target_subject, target_session, seed,
                               device="cpu", variant="full", rehearsal=True,
                               source_steps_per_epoch=None, target_steps_per_epoch=None,
                               fusion_variant="temperature_scaled"):
    """One subject-session and one random run; SEED-V pools all three test blocks.

    Step overrides expose epoch scheduling for integration with an existing
    experiment driver. Sampling counts, loss and prescribed epochs are unchanged.
    Returns genuine held-out predictions, scores and selected fusion parameters.
    """
    seed_random_generators(seed)
    rng = np.random.default_rng(seed)
    classes = DATASETS[dataset].classes
    target = [t for t in all_trials if (t.subject, t.session) == (target_subject, target_session)]
    partitions = personalized_partitions(dataset, target)
    source = personalized_source_pool(all_trials, target_subject, target_session)
    source_windows, _ = prepare_source_units(source)
    pretrained, states = build_pathways(classes, variant), {}
    for pathway, model in pretrained.items():
        model.to(device)
        optimizer = make_optimizer(model, pathway)
        sampler = SourceTrialSampler(source_windows, classes, rng)
        for _ in range(TRAINING_BUDGETS[pathway].pretraining_epochs):
            for batch in sampler.pretraining_epoch(pathway, source_steps_per_epoch):
                optimization_step(model, optimizer, batch.to(device), pathway)
        states[pathway] = copy_pretrained_state(model)
    blocks, pooled_ids, pooled_labels, pooled_predictions = [], [], [], []
    for partition in partitions:
        validation_logits = {p: [] for p in pretrained}
        oof_labels, oof_ids = [], []
        for adaptation, validation in partition.validation_folds:
            adaptation_windows, validation_windows, _ = prepare_training_and_evaluation(adaptation, validation)
            for pathway, prototype in pretrained.items():
                model = copy.deepcopy(prototype)
                adapt_pathway(model, states[pathway], adaptation_windows, source_windows,
                              pathway, classes, rng, device, rehearsal, target_steps_per_epoch)
                validation_logits[pathway].append(predict_trial_logits(model, validation_windows, device))
            oof_labels.extend(t.label for t in validation)
            oof_ids.extend(t.identity for t in validation)
        fusion, validation_metrics = select_fusion(
            np.concatenate(validation_logits["global"]), np.concatenate(validation_logits["regional"]),
            oof_labels, oof_ids, [t.identity for t in partition.training], fusion_variant)
        adaptation_windows, test_windows, _ = prepare_training_and_evaluation(partition.training, partition.testing)
        test_logits = {}
        for pathway, prototype in pretrained.items():
            model = copy.deepcopy(prototype)
            adapt_pathway(model, states[pathway], adaptation_windows, source_windows,
                          pathway, classes, rng, device, rehearsal, target_steps_per_epoch)
            test_logits[pathway] = predict_trial_logits(model, test_windows, device)
        fused = fuse_trial_logits(test_logits["global"], test_logits["regional"], fusion)
        predictions = fused.argmax(1)
        labels = [t.label for t in partition.testing]
        identities = [t.identity for t in partition.testing]
        blocks.append({"test_block": partition.test_block, "fusion": fusion,
                       "validation_metrics": validation_metrics, "trial_identities": identities,
                       "labels": np.asarray(labels), "predictions": predictions,
                       "global_logits": test_logits["global"], "regional_logits": test_logits["regional"],
                       "fused_logits": fused})
        pooled_ids.extend(identities)
        pooled_labels.extend(labels)
        pooled_predictions.extend(predictions)
    if len(pooled_ids) != len(set(pooled_ids)):
        raise RuntimeError("A trial must contribute exactly one test prediction per run")
    return {"subject": target_subject, "session": target_session, "run": seed, "blocks": blocks,
            "trial_identities": pooled_ids, "labels": np.asarray(pooled_labels),
            "predictions": np.asarray(pooled_predictions),
            **trial_metrics(pooled_labels, pooled_predictions, classes)}
