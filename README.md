[README (1).md](https://github.com/user-attachments/files/33383493/README.1.md)
# RAT-SNN

Core implementation of **RAT-SNN: A Representation-Aware Temporal Spiking Neural Network for EEG Emotion Recognition**.

RAT-SNN processes multiband differential entropy (DE) sequences through two independently trained spiking pathways:

- **Global pathway:** complete electrode–band states, band recalibration, and a five-layer PLIF temporal encoder.
- **Regional pathway:** electrode-centered features, 27 bilateral difference pairs, band-specific projections, and a PLIF temporal encoder.
- **Trial-level fusion:** within-trial mean logits followed by temperature-scaled weighted logit fusion.
- **Personalization:** source pretraining followed by supervised target adaptation with class-balanced source rehearsal.

The package contains model, preprocessing, sampling, training, evaluation, and ablation modules. Dataset readers and command-line experiment launchers must be supplied separately. Dataset files, pretrained checkpoints, and the original experiment prediction files are not included.

## Installation

Run the following command from the repository root:

```bash
python -m pip install -r requirements.txt
```

The core dependencies are NumPy 2.1.3 and PyTorch 2.5.1. The manuscript records Python 3.12.3, CUDA 12.4, and an NVIDIA GeForce RTX 4090 D. CPU execution is also supported through `device="cpu"`; GPU execution requires a compatible CUDA-enabled PyTorch installation.

## Code structure

All implementation modules are in `rat_snn_core/`.

| File | Purpose |
| --- | --- |
| `__init__.py` | Public model, data, standardization, and fusion interfaces. |
| `settings.py` | Architecture dimensions, numerical constants, training budgets, and fusion grids. |
| `electrode_metadata.py` | Ordered 62-channel montage, five bands, bilateral pairs, and midline sites. |
| `data_standardization.py` | Trial data structures, trial-balanced standardization, and within-trial windows. |
| `spiking_neurons.py` | PLIF dynamics, surrogate gradients, and reset rules. |
| `global_pathway.py` | Global representation, temporal encoder, and window-level logits. |
| `regional_pathway.py` | Regional representation, temporal encoder, and window-level logits. |
| `pathway_objectives.py` | Window loss, grouped trial loss, firing-rate penalty, and rehearsal objective. |
| `trial_batch_sampling.py` | Class-balanced source sampling, target window cycles, and size-matched rehearsal. |
| `training_personalization.py` | Optimization, source-weight initialization, validation loss, and early stopping. |
| `evaluation_protocols.py` | Personalized trial partitions, source exclusions, and session-wise LOSO folds. |
| `personalized_evaluation.py` | Source pretraining, target validation, fusion selection, and final adaptation. |
| `cross_subject_evaluation.py` | Source-only inner LOSO, final source retraining, and held-out subject evaluation. |
| `trial_logit_fusion.py` | All-window inference, trial aggregation, logit fusion, and validation selection. |
| `evaluation_metrics.py` | ACC, Macro-F1, subject-level summaries, and confusion matrices. |
| `repeated_evaluation.py` | Ten independent runs with fixed partitions and subject-level aggregation. |
| `component_ablations.py` | Component switches and fixed-LIF ablation. |
| `source_retention_audit.py` | Matched adaptation comparison with and without source rehearsal. |

## Datasets and input format

Dataset information and access instructions are available from the BCMI Laboratory:

- [SEED](https://bcmi.sjtu.edu.cn/home/seed/seed.html)
- [SEED-IV](https://bcmi.sjtu.edu.cn/home/seed/seed-iv.html)
- [SEED-V](https://bcmi.sjtu.edu.cn/home/seed/seed-v.html)

Prepare the released DE features as a list of `DETrial` objects. Each object represents one complete trial:

| Field | Format |
| --- | --- |
| `subject` | Subject identifier; use consistent identifiers across sessions. |
| `session` | Session identifier; the evaluation protocol uses three sessions. |
| `trial` | One-based chronological trial index within the session. |
| `label` | Zero-based class index in `0, ..., K-1`. |
| `frames` | Finite DE values with shape `[L, 62, 5]`, where `L >= 4`. |

The electrode axis must follow `CHANNEL_NAMES` in `electrode_metadata.py`. The band axis follows `delta, theta, alpha, beta, gamma`. Keep the same class-label mapping across all subjects, sessions, and pathways within a dataset. Convert the dataset's original labels to zero-based indices before constructing trials.

The core standardizer accepts the original DE frames. It estimates electrode–band statistics from permitted training trials, giving each trial equal weight. Windows are then constructed with **four DE observations and stride one**, without crossing trial boundaries. Model inputs have shape `[batch, 4, 62, 5]` and dtype `float32`; model outputs are window-level logits of shape `[batch, K]`.

## Minimal model example

This synthetic example checks the model interface; it does not evaluate EEG recognition performance.

```python
import torch
from rat_snn_core import GlobalPathway, RegionalPathway

x = torch.randn(8, 4, 62, 5)  # Synthetic standardized windows.
global_model = GlobalPathway(n_classes=3).eval()
regional_model = RegionalPathway(n_classes=3).eval()

with torch.no_grad():
    global_logits = global_model(x)
    regional_logits = regional_model(x)

print(global_logits.shape)    # torch.Size([8, 3])
print(regional_logits.shape)  # torch.Size([8, 3])
```

## Evaluation interfaces

The following functions accept `all_trials`, a list of `DETrial` objects produced by your dataset reader. Supply the complete subjects, sessions, trials, and labels required by the selected protocol. Dataset names are exactly `"SEED"`, `"SEED-IV"`, and `"SEED-V"`.

```python
from rat_snn_core.personalized_evaluation import evaluate_personalized_unit
from rat_snn_core.cross_subject_evaluation import evaluate_cross_subject_fold
from rat_snn_core.evaluation_protocols import session_loso_folds


def run_personalized(all_trials, dataset, subject, session, seed, device="cpu"):
    return evaluate_personalized_unit(
        dataset=dataset,
        all_trials=all_trials,
        target_subject=subject,
        target_session=session,
        seed=seed,
        device=device,
    )


def run_cross_subject_session(all_trials, dataset, session, seed, device="cpu"):
    results = []
    for _, source_trials, test_trials in session_loso_folds(
        dataset, all_trials, session
    ):
        results.append(evaluate_cross_subject_fold(
            dataset=dataset,
            source_trials=source_trials,
            test_trials=test_trials,
            seed=seed,
            device=device,
        ))
    return results
```

Use `device="cuda"` for GPU execution. Returned records contain held-out trial labels, predictions, metrics, and selected fusion parameters. They do not automatically save files; your experiment driver should save the returned records and the actual configuration used.

### Personalized evaluation

| Dataset | Target training/test partition | Validation within target training trials |
| --- | --- | --- |
| SEED | First 9 trials for training; last 6 for testing. | 3 class-balanced folds: 6 adaptation and 3 validation trials. |
| SEED-IV | Last 2 trials of each class for testing; 16 training and 8 test trials. | 4 class-balanced folds: 12 adaptation and 4 validation trials. |
| SEED-V | Three blocks: trials 1–5, 6–10, and 11–15; each block is tested once. | The other two blocks alternate between adaptation and validation. |

Each SEED-V block must contain one trial per class. Its three test blocks are pooled within a subject–session/run before calculating metrics.

For a target subject–session unit, source units sharing either the target subject or the target session index are excluded. Source standardization is fitted separately for each source subject–session unit. Target validation uses statistics fitted only on the current adaptation trials; final adaptation fits statistics on all target training trials. Rehearsal retains each source unit's standardization.

### Cross-subject evaluation

LOSO evaluation is performed separately within each session. SEED and SEED-IV use 15 outer folds, and SEED-V uses 16. Source-only inner LOSO provides validation predictions for fusion selection and completed epoch counts for final training.

In each inner fold, one common trial-balanced standardizer is fitted on the inner-training subjects and applied unchanged to the validation subject. Final training refits it on all outer-fold source subjects and applies it unchanged to the test subject. The test subject supplies no training data, standardization statistics, or selection criteria. No target adaptation is performed.

## Training defaults

| Setting | Global | Regional |
| --- | --- | --- |
| Optimizer | AdamW | AdamW |
| Personalized source-pretraining epochs | 8 | 4 |
| Personalized adaptation epochs | 12 | 6 |
| Source-training trials per class per batch | 4 | 3 |
| Windows sampled per selected trial | 4 | 4 |
| Source-training learning rate | `3e-4` | `1e-3` |
| Adaptation learning rate | `3e-4` | `3e-4` |
| Weight decay | `1e-3` | `1e-3` |
| Label smoothing | `0.05` | `0.05` |
| Trial-loss weight | `0.5` | `0.5` |
| Firing-rate penalty weight | `0.005` | — |
| Target firing rate | `0.15` | — |
| Gradient clipping norm | `5` | `5` |
| Rehearsal weight | `0.5` | `0.5` |

Batch size depends on the selected trials and training stage. During adaptation, each batch includes all current target adaptation trials, with four windows per trial, and an equally sized class-balanced source rehearsal batch. The adaptation objective is `(L_target + 0.5 * L_source) / 1.5`, and the two losses are combined in one optimizer update.

For source training, the default updates per epoch are `ceil(N_max / 4)` for Global and `ceil(N_max / 3)` for Regional, where `N_max` is the largest class-specific source-trial count. For target adaptation, the default is `ceil(max(N_q) / 4)`, where `N_q` is an eligible target trial's window count. Both evaluation interfaces accept step-count overrides for integration with an existing driver; record any overrides used.

Cross-subject inner training uses **at most 100 epochs**, with validation once per epoch and **patience 30**. The monitor is the pathway objective on all validation windows and trials: window-level label-smoothed cross-entropy plus `0.5` times trial-level label-smoothed cross-entropy, with the `0.005` firing-rate penalty additionally applied to Global. Window loss averages windows; trial loss equally averages trials. Global firing rates are pooled over all validation windows before penalization. Validation uses evaluation mode and no gradient updates.

An improvement is a strict decrease relative to the lowest previous validation loss (`min_delta=0`). Final completed-epoch weights are retained. Final source retraining starts from fresh initialization and uses the ceiling of the median completed inner-fold epoch count, separately for each pathway.

## Logit fusion

For each trial, average all eligible window logits separately for each pathway, then fuse:

$$
\bar{\mathbf z}^{\mathrm f}_q
=(1-\lambda_{\mathrm r})\frac{\bar{\mathbf z}^{\mathrm g}_q}{T_{\mathrm g}}
+\lambda_{\mathrm r}\frac{\bar{\mathbf z}^{\mathrm r}_q}{T_{\mathrm r}}.
$$

The final prediction is the class with the largest fused logit. Fusion combines temperature-scaled **logits**, with softmax used to evaluate candidate negative log-likelihood.

The search contains 592 candidates: `lambda_r = 0.05 + 0.025*j`, for `j = 0, ..., 36`, and each temperature in `{0.8, 1.0, 1.2, 1.4}`. Candidates maximize Macro-F1 on concatenated out-of-fold trial predictions. Ties are resolved by higher ACC, lower NLL, temperatures closer to one, weight closer to 0.5, and finally ascending candidate order.

Selection is separate for each target unit/run, each SEED-V held-out block/run, or each outer LOSO fold/run. Selected parameters remain fixed during the corresponding test evaluation.

## Controlled variants

Pass one of the following `variant` values to an evaluation interface:

- `"full"`
- `"without_band_recalibration"`
- `"without_centering"`
- `"without_dasm"`
- `"fixed_lif"`

For fixed LIF, Global integration coefficients remain at `0.5`, and Regional retention and threshold remain at `0.70` and `0.7501`. Surrogate gradients and reset-gradient rules are unchanged.

Supported `fusion_variant` values are `"temperature_scaled"`, `"weighted_without_temperature"`, and `"equal_logits"`. Equal-logit fusion fixes the weight at 0.5 and both temperatures at 1. Personalized evaluation also accepts `rehearsal=False` for target-only adaptation.

`source_retention_audit.py` compares Global adaptation with and without rehearsal on a fixed, class-balanced SEED-IV source subset. Audit trials remain eligible for rehearsal, so this analysis measures retention on previously available source data.

## Repetitions and reporting

`evaluate_ten_runs(evaluate_run, random_seeds)` in `repeated_evaluation.py` requires ten distinct seeds supplied explicitly. `evaluate_run(seed)` must return one record for every subject–session unit for that seed. Keep partitions and the seed list fixed across controlled variants.

ACC and Macro-F1 are computed from trial-level predictions. Scores are averaged over three sessions and ten runs within each subject, then the mean and sample standard deviation (`ddof=1`) are calculated across subjects. Metric values returned by the code are fractions; multiply by 100 for percentages.

Confusion matrices pool integer trial-level counts across subjects, sessions, and runs, then normalize each row to percentages. Predictions from different runs are not ensembled. Save raw labels and predictions so tables, confusion matrices, and statistical summaries can be traced to the same experiment records.

The core package does not specify the ten seeds used in the original experiments or contain their raw outputs. Reproducing a reported numerical result requires the matching dataset files, experiment settings, random seeds, and saved predictions.
