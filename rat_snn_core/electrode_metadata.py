"""Ordered 62-channel montage, 27 bilateral pairs, and eight midline sites.

Input frames must use CHANNEL_NAMES order. No midline channel is removed from
the centered 62-channel representation.
"""
CHANNEL_NAMES = (
    "FP1", "FPZ", "FP2", "AF3", "AF4",
    "F7", "F5", "F3", "F1", "FZ", "F2", "F4", "F6", "F8",
    "FT7", "FC5", "FC3", "FC1", "FCZ", "FC2", "FC4", "FC6", "FT8",
    "T7", "C5", "C3", "C1", "CZ", "C2", "C4", "C6", "T8",
    "TP7", "CP5", "CP3", "CP1", "CPZ", "CP2", "CP4", "CP6", "TP8",
    "P7", "P5", "P3", "P1", "PZ", "P2", "P4", "P6", "P8",
    "PO7", "PO5", "PO3", "POZ", "PO4", "PO6", "PO8",
    "CB1", "O1", "OZ", "O2", "CB2",
)
BAND_NAMES = ("delta", "theta", "alpha", "beta", "gamma")
BAND_LIMITS_HZ = ((1, 4), (4, 8), (8, 14), (14, 31), (31, 50))
BILATERAL_PAIRS = (
    ("FP1", "FP2"), ("AF3", "AF4"), ("F7", "F8"), ("F5", "F6"),
    ("F3", "F4"), ("F1", "F2"), ("FT7", "FT8"), ("FC5", "FC6"),
    ("FC3", "FC4"), ("FC1", "FC2"), ("T7", "T8"), ("C5", "C6"),
    ("C3", "C4"), ("C1", "C2"), ("TP7", "TP8"), ("CP5", "CP6"),
    ("CP3", "CP4"), ("CP1", "CP2"), ("P7", "P8"), ("P5", "P6"),
    ("P3", "P4"), ("P1", "P2"), ("PO7", "PO8"), ("PO5", "PO6"),
    ("PO3", "PO4"), ("CB1", "CB2"), ("O1", "O2"),
)
MIDLINE_CHANNELS = ("FPZ", "FZ", "FCZ", "CZ", "CPZ", "PZ", "POZ", "OZ")
CHANNEL_INDEX = {name: index for index, name in enumerate(CHANNEL_NAMES)}
LEFT_INDICES = tuple(CHANNEL_INDEX[left] for left, _ in BILATERAL_PAIRS)
RIGHT_INDICES = tuple(CHANNEL_INDEX[right] for _, right in BILATERAL_PAIRS)


def validate_channel_order(channel_names):
    if tuple(channel_names) != CHANNEL_NAMES:
        raise ValueError("Reorder DE electrodes to CHANNEL_NAMES before constructing windows")


assert len(CHANNEL_NAMES) == len(set(CHANNEL_NAMES)) == 62
assert len(BILATERAL_PAIRS) == 27
assert set(CHANNEL_NAMES) - {name for pair in BILATERAL_PAIRS for name in pair} == set(MIDLINE_CHANNELS)
