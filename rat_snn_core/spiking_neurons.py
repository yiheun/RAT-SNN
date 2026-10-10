"""Global/Regional PLIF equations, binary spikes, and surrogate derivatives.

Global: alpha=sigmoid(w), feature-wise; reset spike is detached.
Regional: bounded shared beta and softplus threshold; reset spike is NOT detached.
Each forward call starts every membrane state at zero, independently per window.
"""
import math
import torch
from torch import nn
from torch.nn import functional as F
from .settings import (GLOBAL_SURROGATE_SCALE, REGIONAL_SURROGATE_SCALE,
                       REGIONAL_BETA_MIN, REGIONAL_BETA_MAX,
                       REGIONAL_INITIAL_BETA, REGIONAL_THRESHOLD_EPSILON,
                       REGIONAL_INITIAL_SOFTPLUS_THRESHOLD)


class FastSigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.save_for_backward(value)
        ctx.scale = float(scale)
        return (value >= 0).to(value.dtype)

    @staticmethod
    def backward(ctx, upstream):
        (value,) = ctx.saved_tensors
        derivative = ctx.scale / (1.0 + ctx.scale * value.abs()).square()
        return upstream * derivative, None


class SigmoidSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, scale):
        ctx.save_for_backward(value)
        ctx.scale = float(scale)
        return (value >= 0).to(value.dtype)

    @staticmethod
    def backward(ctx, upstream):
        (value,) = ctx.saved_tensors
        soft = torch.sigmoid(ctx.scale * value)
        return upstream * ctx.scale * soft * (1.0 - soft), None


class GlobalPLIF(nn.Module):
    """eq:global_plif_alpha, eq:global_plif_dynamics, eq:global_surrogate_gradient."""
    def __init__(self, features, threshold=1.0):
        super().__init__()
        self.w = nn.Parameter(torch.zeros(features))
        self.threshold = float(threshold)

    def forward(self, current):
        if current.ndim != 3 or current.shape[-1] != self.w.numel() or not current.shape[0]:
            raise ValueError("Expected nonempty [T,B,D] Global currents")
        alpha = torch.sigmoid(self.w)
        membrane = torch.zeros_like(current[0])
        spikes, membranes = [], []
        for current_t in current.unbind(0):
            integrated = membrane + alpha * (current_t - membrane)
            spike = FastSigmoidSpike.apply(integrated - self.threshold, GLOBAL_SURROGATE_SCALE)
            membrane = integrated - spike.detach() * self.threshold
            spikes.append(spike)
            membranes.append(membrane)
        return torch.stack(spikes), torch.stack(membranes)


class LearnableRetention(nn.Module):
    def __init__(self):
        super().__init__()
        self.raw = nn.Parameter(torch.tensor(math.log(0.60 / 0.29), dtype=torch.float32))

    def forward(self):
        return REGIONAL_BETA_MIN + (REGIONAL_BETA_MAX - REGIONAL_BETA_MIN) * torch.sigmoid(self.raw)


class RegionalPLIF(nn.Module):
    """eq:regional_plif_parameters/dynamics/surrogate_gradient; input [B,T,64]."""
    def __init__(self, hidden_dim=64):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.leak = LearnableRetention()
        self.raw_threshold = nn.Parameter(torch.tensor(
            math.log(math.expm1(REGIONAL_INITIAL_SOFTPLUS_THRESHOLD)), dtype=torch.float32))

    def threshold(self):
        return F.softplus(self.raw_threshold) + REGIONAL_THRESHOLD_EPSILON

    def forward(self, current):
        if current.ndim != 3 or current.shape[-1] != self.hidden_dim or not current.shape[1]:
            raise ValueError("Expected nonempty [B,T,64] Regional currents")
        beta, threshold = self.leak(), self.threshold()
        membrane = torch.zeros_like(current[:, 0])
        spikes, membranes = [], []
        for current_t in current.unbind(1):
            integrated = beta * membrane + (1.0 - beta) * current_t
            spike = SigmoidSpike.apply(integrated - threshold, REGIONAL_SURROGATE_SCALE)
            membrane = integrated - spike * threshold
            spikes.append(spike)
            membranes.append(membrane)
        return torch.stack(spikes, dim=1), torch.stack(membranes, dim=1)
