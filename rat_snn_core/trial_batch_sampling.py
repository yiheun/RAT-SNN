"""Class-balanced source trials, target window cycles, and size-matched replay."""
from dataclasses import dataclass
import math
import numpy as np
import torch
from .settings import WINDOWS_PER_TRIAL, TRAINING_BUDGETS


@dataclass(frozen=True)
class TrialBatch:
    windows: torch.Tensor
    labels: torch.Tensor
    parent_trial_ids: torch.Tensor

    def to(self, device):
        return TrialBatch(self.windows.to(device), self.labels.to(device), self.parent_trial_ids.to(device))


def assemble_batch(trials, indices):
    if not trials or len(trials) != len(indices):
        raise ValueError("Expected selected trials and matching window indices")
    if len({t.identity for t in trials}) != len(trials):
        raise ValueError("Do not select the same trial twice in one batch")
    return TrialBatch(torch.from_numpy(np.concatenate([t.windows[ix] for t, ix in zip(trials, indices)])),
                      torch.tensor([t.label for t, ix in zip(trials, indices) for _ in ix], dtype=torch.long),
                      torch.tensor([i for i, ix in enumerate(indices) for _ in ix], dtype=torch.long))


class SourceTrialSampler:
    def __init__(self, trials, n_classes, rng):
        self.rng = rng
        if any(t.label not in range(n_classes) or not len(t.windows) for t in trials):
            raise ValueError("Invalid source class or empty window list")
        self.pools = {k: [t for t in trials if t.label == k] for k in range(n_classes)}
        if len({t.identity for t in trials}) != len(trials) or any(not pool for pool in self.pools.values()):
            raise ValueError("Source trials must have unique identities and every class")
        self.queues = {k: [] for k in self.pools}

    def select(self, counts):
        selected = []
        for k, number in enumerate(counts):
            pool = self.pools[k]
            if number < 1 or number > len(pool):
                raise ValueError("Not enough unique source trials in class")
            chosen = []
            while len(chosen) < number:
                if not self.queues[k]:
                    self.queues[k] = list(self.rng.permutation(len(pool)))
                index = self.queues[k].pop()
                # Completing a tail never introduces duplicates within this batch.
                if index not in chosen:
                    chosen.append(index)
            selected.extend(pool[i] for i in chosen)
        indices = [self.rng.choice(len(t.windows), WINDOWS_PER_TRIAL,
                                  replace=len(t.windows) < WINDOWS_PER_TRIAL) for t in selected]
        return assemble_batch(selected, indices)

    def pretraining_epoch(self, pathway, steps=None):
        per_class = TRAINING_BUDGETS[pathway].source_trials_per_class
        if steps is None:
            steps = max(math.ceil(len(pool) / per_class) for pool in self.pools.values())
        for _ in range(steps):
            yield self.select([per_class] * len(self.pools))


class TargetTrialSampler:
    def __init__(self, trials, n_classes, rng):
        self.trials, self.rng = list(trials), rng
        if any(t.label not in range(n_classes) or not len(t.windows) for t in trials):
            raise ValueError("Invalid target class or empty window list")
        if not self.trials or len({t.identity for t in trials}) != len(trials):
            raise ValueError("Target trials must be nonempty and unique")
        self.class_counts = [sum(t.label == k for t in trials) for k in range(n_classes)]
        if min(self.class_counts) < 1 or len(set(self.class_counts)) != 1:
            raise ValueError("The prescribed target adaptation sets are class-balanced")
        self.queues = {t.identity: [] for t in trials}

    def batch(self):
        indices = []
        for trial in self.trials:
            sampled = []
            while len(sampled) < WINDOWS_PER_TRIAL:
                if not self.queues[trial.identity]:
                    self.queues[trial.identity] = list(self.rng.permutation(len(trial.windows)))
                sampled.append(self.queues[trial.identity].pop())
            indices.append(np.asarray(sampled))
        return assemble_batch(self.trials, indices)

    def adaptation_epoch(self, source_sampler=None, steps=None):
        # One epoch cycles through the longest target trial's window list.
        # Explicit steps can be supplied when integrating an existing driver.
        if steps is None:
            steps = math.ceil(max(len(t.windows) for t in self.trials) / WINDOWS_PER_TRIAL)
        for _ in range(steps):
            target = self.batch()
            source = source_sampler.select(self.class_counts) if source_sampler is not None else None
            if source is not None and len(source.labels) != len(target.labels):
                raise RuntimeError("Target and rehearsal batches must have equal window counts")
            yield target, source
