"""Complete electrode-band state, band recalibration and Global PLIF pathway."""
import torch
from torch import nn
from torch.nn import functional as F
from .spiking_neurons import GlobalPLIF
from .settings import RMS_EPSILON


def xavier_linear(layer):
    nn.init.xavier_uniform_(layer.weight)
    nn.init.zeros_(layer.bias)
    return layer


class BandRecalibration(nn.Module):
    def __init__(self, enabled=True):
        super().__init__()
        self.enabled = enabled
        if not enabled:
            return
        self.fc1 = xavier_linear(nn.Linear(10, 8))
        self.fc2 = nn.Linear(8, 5)
        nn.init.zeros_(self.fc2.weight)
        nn.init.zeros_(self.fc2.bias)

    def forward(self, x):
        if not self.enabled:
            return x, torch.ones_like(x[:, 0, 0, :])
        descriptor = torch.cat((x.mean((1, 2)),
                                torch.sqrt(x.square().mean((1, 2)) + RMS_EPSILON)), -1)
        scale = 1.0 + 0.5 * torch.tanh(self.fc2(F.silu(self.fc1(descriptor))))
        return x * scale[:, None, None, :], scale


class StateBridge(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw_coefficient = nn.Parameter(torch.zeros(()))

    def forward(self, spikes, membrane):
        gamma = torch.tanh(self.raw_coefficient)
        return (spikes + gamma * torch.tanh(membrane)) / torch.sqrt(1 + gamma.square())


class GlobalFrontEnd(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = xavier_linear(nn.Linear(310, 64))
        self.lif1 = GlobalPLIF(64, 1.0)
        self.fc2 = xavier_linear(nn.Linear(64, 32))
        self.norm2 = nn.Identity()
        self.lif2 = GlobalPLIF(32, 0.35)
        self.readout = xavier_linear(nn.Linear(64, 32))
        self.dropout = nn.Dropout(0.10)
        self.state_bridge = StateBridge()

    def forward(self, x):
        s1, u1 = self.lif1(self.fc1(x.flatten(2)).transpose(0, 1))
        state = self.state_bridge(s1, u1)
        s2, u2 = self.lif2(self.norm2(self.fc2(self.dropout(state))))
        return self.readout(torch.cat((s2, torch.tanh(u2)), -1))


class TemporalPLIFBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = xavier_linear(nn.Linear(input_dim, output_dim))
        self.norm = nn.LayerNorm(output_dim)
        self.plif = GlobalPLIF(output_dim)
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        return self.plif(self.norm(self.linear(self.dropout(x))))


class MembraneClassificationHead(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.head_fc = xavier_linear(nn.Linear(96, 32))
        self.head_norm = nn.LayerNorm(32)
        self.head_lif = GlobalPLIF(32)
        self.classifier = xavier_linear(nn.Linear(32, n_classes))
        self.dropout = nn.Dropout(0.15)

    def forward(self, x):
        spikes, membrane = self.head_lif(self.head_norm(self.head_fc(self.dropout(x))))
        return self.classifier(self.dropout(membrane)).mean(0), spikes.mean()


class GlobalTemporalClassifier(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        self.plif1 = TemporalPLIFBlock(32, 96)
        self.plif2 = TemporalPLIFBlock(96, 96)
        self.head = MembraneClassificationHead(n_classes)

    def forward(self, x):
        s3, _ = self.plif1(x)
        s4, u4 = self.plif2(s3)
        logits, head_rate = self.head(0.5 * s4 + 0.5 * torch.tanh(u4))
        return logits, torch.stack((s3.mean(), s4.mean(), head_rate))


class GlobalPathway(nn.Module):
    def __init__(self, n_classes, band_recalibration=True):
        super().__init__()
        if n_classes not in (3, 4, 5):
            raise ValueError("RAT-SNN datasets contain three, four or five classes")
        self.bandse = BandRecalibration(band_recalibration)
        self.global_branch = GlobalFrontEnd()
        self.temporal_classifier = GlobalTemporalClassifier(n_classes)

    def forward(self, x, return_details=False):
        if x.ndim != 4 or tuple(x.shape[1:]) != (4, 62, 5) or not torch.isfinite(x).all():
            raise ValueError("Expected finite standardized DE windows [B,4,62,5]")
        weighted, scales = self.bandse(x)
        logits, rates = self.temporal_classifier(self.global_branch(weighted))
        if return_details:
            return {"main_logits": logits, "firing_rates": rates, "band_scales": scales}
        return logits
