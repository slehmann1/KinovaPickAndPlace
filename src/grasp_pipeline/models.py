from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class GraspCandidate:
    """One candidate grasp expressed in the MuJoCo object body frame."""

    grasp_id: str
    object_T_grasp: np.ndarray
    grasp_width_m: float | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GraspObjectSample:
    """Object mesh plus a set of executable grasp candidates."""

    object_id: str
    mesh_path: Path
    candidates: tuple[GraspCandidate, ...]
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrialRecord:
    """Serializable record of one executed grasp trial."""

    object_id: str
    grasp_id: str
    success: bool
    failure_reason: str
    initial_object_pos: tuple[float, float, float]
    final_object_pos: tuple[float, float, float]
    lift_delta_m: float
    grasp_width_m: float | None = None
    score: float | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
