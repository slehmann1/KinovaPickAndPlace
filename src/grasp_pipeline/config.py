from dataclasses import dataclass


@dataclass(frozen=True)
class SceneConfig:
    """Configuration for placing one object into the benchmarking environment."""

    object_xy: tuple[float, float] = (-0.2, 0.0)
    object_scale: tuple[float, float, float] | None = None
    collision_mode: str = "box"
    object_drop_height: float = 0.05


@dataclass(frozen=True)
class ExecutionConfig:
    """Controller timings and geometric thresholds for scripted grasp execution."""

    settle_steps: int = 60
    safe_start_pos: tuple[float, float, float] = (0.15, 0.0, 1.25)
    pregrasp_offset_m: float = 0.10
    lift_distance_m: float = 0.10
    move_steps: int = 150
    gripper_close_steps: int = 40
    lift_steps: int = 120
    min_successful_lift_m: float = 0.05
    max_post_lift_xy_slip_m: float = 0.08
