"""Electrode centering, bilateral differences, band encoder and Regional PLIF."""
import torch
from torch import nn
from torch.nn import functional as F
from .electrode_metadata import LEFT_INDICES, RIGHT_INDICES
from .spiking_neurons import RegionalPLIF


class CenteredBilateralRepresentation(nn.Module):
    def __init__(self, centering=True, bilateral_differences=True):
        super().__init__()
        self.centering = centering
        self.bilateral_differences = bilateral_differences
        self.register_buffer("bilateral_left", torch.tensor(LEFT_INDICES))
        self.register_buffer("bilateral_right", torch.tensor(RIGHT_INDICES))

    def forward(self, x):
        centered = x - x.mean(2, keepdim=True) if self.centering else x
        if not self.bilateral_differences:
            return centered
        differences = x.index_select(2, self.bilateral_left) - x.index_select(2, self.bilateral_right)
        return torch.cat((centered, differences), 2)


class BandwiseEncoder(nn.Module):
    def __init__(self, spatial_dim=89):
        super().__init__()
        self.band_projection = nn.ModuleList([nn.Linear(spatial_dim, 8) for _ in range(5)])
        self.fusion = nn.Linear(40, 64)
        self.norm = nn.LayerNorm(64)

    def forward(self, x):
        encoded = [F.gelu(layer(x[..., band])) for band, layer in enumerate(self.band_projection)]
        return F.softplus(self.norm(self.fusion(torch.cat(encoded, -1))))


class RegionalPathway(nn.Module):
    def __init__(self, n_classes, centering=True, bilateral_differences=True):
        super().__init__()
        if n_classes not in (3, 4, 5):
            raise ValueError("RAT-SNN datasets contain three, four or five classes")
        self.transform = CenteredBilateralRepresentation(centering, bilateral_differences)
        self.encoder = BandwiseEncoder(89 if bilateral_differences else 62)
        self.temporal = RegionalPLIF(64)
        self.classifier = nn.Linear(64, n_classes)
        nn.init.zeros_(self.classifier.bias)

    def forward(self, x, return_details=False):
        if x.ndim != 4 or tuple(x.shape[1:]) != (4, 62, 5) or not torch.isfinite(x).all():
            raise ValueError("Expected finite standardized DE windows [B,4,62,5]")
        spikes, membrane = self.temporal(self.encoder(self.transform(x)))
        logits = self.classifier(spikes.mean(1))
        if return_details:
            return {"main_logits": logits, "spikes": spikes, "post_reset_membrane": membrane}
        return logits
