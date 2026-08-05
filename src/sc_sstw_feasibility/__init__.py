"""CPU-only primitives for SC-SSTW scientific feasibility probes."""

from .calibration import CalibrationResult, calibrate_channel, equalize_observations
from .aisb import BurstCandidate, BurstTemplate, affine_burst_residual, scan_burst_candidates
from .channel import SyntheticChannel, generate_observations
from .scoring import score_key
from .state import PUBLIC_SYNC_DIRECTIONS, PilotPattern, generate_state_sequence
from .sync import SyncResult, dynamic_time_sync, dynamic_time_sync_score

__all__ = [
    "CalibrationResult",
    "BurstCandidate",
    "BurstTemplate",
    "PilotPattern",
    "PUBLIC_SYNC_DIRECTIONS",
    "SyncResult",
    "SyntheticChannel",
    "affine_burst_residual",
    "calibrate_channel",
    "dynamic_time_sync",
    "dynamic_time_sync_score",
    "equalize_observations",
    "generate_observations",
    "generate_state_sequence",
    "score_key",
]
