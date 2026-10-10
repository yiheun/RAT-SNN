"""RAT-SNN core implementation corresponding to the finalized manuscript.

    from rat_snn_core import GlobalPathway, RegionalPathway
    global_model = GlobalPathway(n_classes=3)
    regional_model = RegionalPathway(n_classes=3)

Input: standardized float32 DE windows [batch,4,62,5]. Output: window logits.
Dataset trial loading and experiment launcher scripts are intentionally external.
"""
from .global_pathway import GlobalPathway
from .regional_pathway import RegionalPathway
from .data_standardization import DETrial, WindowTrial, TrialBalancedStandardizer
from .trial_logit_fusion import FusionParameters, fuse_trial_logits, select_fusion

__all__ = ["GlobalPathway", "RegionalPathway", "DETrial", "WindowTrial",
           "TrialBalancedStandardizer", "FusionParameters", "fuse_trial_logits", "select_fusion"]
