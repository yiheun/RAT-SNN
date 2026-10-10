"""Controlled component switches; reset/surrogate rules remain unchanged."""
import torch
from .global_pathway import GlobalPathway
from .regional_pathway import RegionalPathway
from .spiking_neurons import GlobalPLIF, RegionalPLIF


def freeze_plif_dynamics(model):
    """Fix Global alpha=.5, Regional beta=.70 and theta=.7501 at initial values."""
    import math
    with torch.no_grad():
        for module in model.modules():
            if isinstance(module, GlobalPLIF):
                module.w.zero_()
                module.w.requires_grad_(False)
            elif isinstance(module, RegionalPLIF):
                module.leak.raw.fill_(math.log(0.60 / 0.29))
                module.raw_threshold.fill_(math.log(math.expm1(0.75)))
                module.leak.raw.requires_grad_(False)
                module.raw_threshold.requires_grad_(False)
    return model


def build_pathways(n_classes, variant="full"):
    if variant not in ("full", "without_band_recalibration", "without_centering", "without_dasm", "fixed_lif"):
        raise ValueError(variant)
    global_model = GlobalPathway(n_classes, band_recalibration=variant != "without_band_recalibration")
    if variant == "without_band_recalibration":
        for parameter in global_model.bandse.parameters():
            parameter.requires_grad_(False)
    regional_model = RegionalPathway(n_classes, centering=variant != "without_centering",
                                     bilateral_differences=variant != "without_dasm")
    if variant == "fixed_lif":
        freeze_plif_dynamics(global_model)
        freeze_plif_dynamics(regional_model)
    # Removing DASM reduces the projection input from 89 to 62: no padding.
    return {"global": global_model, "regional": regional_model}


def parameter_counts(model):
    return {"total": sum(p.numel() for p in model.parameters()),
            "trainable": sum(p.numel() for p in model.parameters() if p.requires_grad)}
