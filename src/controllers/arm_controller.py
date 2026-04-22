import json
from datetime import datetime
from pathlib import Path

import numpy as np
from robosuite.utils.transform_utils import get_orientation_error
from scipy.interpolate import CubicSpline

from src.environments import TargetEnvironment

K_P = 4
K_I = 0
K_D = 0
MAX_POS_ACTION = 0.2
MAX_ROT_ACTION = 0.2


def _make_serializable_value(value):
    """Convert simulator values into JSON-serializable Python types."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {
            key: _make_serializable_value(nested_value)
            for key, nested_value in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_make_serializable_value(item) for item in value]
    return value


def _record_step(data_log, command_name, action, obs, reward, done, env):
    """Append one simulator step to the in-memory run log."""
    if data_log is None:
        return

    object_position, object_orientation = env.get_object_pose()

    data_log.append(
        {
            "step_index": len(data_log),
            "command": command_name,
            "action": _make_serializable_value(action),
            "observation": _make_serializable_value(dict(obs)),
            "reward": _make_serializable_value(reward),
            "done": bool(done),
            "object_position": _make_serializable_value(object_position),
            "object_orientation": _make_serializable_value(object_orientation),
        }
    )


def _step_and_render(env, action, data_log=None, command_name="unnamed_command"):
    """Advance the environment by one action and render the viewer.

    Args:
        env (TargetEnvironment): Active robosuite environment.
        action (np.ndarray): Action vector to apply for one simulation step.
        data_log (list[dict] | None): Optional list used to accumulate step records.
        command_name (str): Label describing the active high-level controller command.

    Returns:
        OrderedDict: Updated observation from the environment.
    """
    obs, reward, done, _ = env.step(action)
    _record_step(data_log, command_name, action, obs, reward, done, env)

    if getattr(env, "has_renderer", False):
        env.render()
    return obs


def store_all_data(data_log, output_path=None, metadata=None):
    """Persist a recorded controller run to disk as JSON.

    Args:
        data_log (list[dict]): Step-by-step run data collected by controller functions.
        output_path (str | Path | None): Destination JSON path. Defaults to a timestamped
            file under `data/run_logs`.
        metadata (dict | None): Optional run-level metadata to include in the file.

    Returns:
        Path: Absolute path to the written JSON file.
    """
    project_root = Path(__file__).resolve().parents[2]
    if output_path is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = project_root / "data" / "run_logs" / f"run_{timestamp}.json"
    else:
        output_path = Path(output_path)
        if not output_path.is_absolute():
            output_path = project_root / output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": _make_serializable_value(metadata or {}),
        "step_count": len(data_log),
        "steps": data_log,
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path.resolve()


def initialize_environment(
    dataset_mesh_path=None,
    object_qpos=None,
    object_scale=(1.0, 1.0, 1.0),
    has_renderer=True,
    obstacle_configs=None,
):
    """Create and reset the demo environment.

    Args:
        dataset_mesh_path (str | None): Optional dataset mesh to load in the scene.
        object_qpos (sequence[float] | None): Optional object pose override.
        object_scale (tuple[float, float, float]): Per-axis mesh scale for dataset objects.
        has_renderer (bool): Whether to launch the interactive MuJoCo viewer.
        obstacle_configs (list[dict] | None): Optional fixed box obstacles to add.

    Returns:
        tuple[TargetEnvironment, OrderedDict]: Initialized environment and first observation.
    """
    env = TargetEnvironment(
        dataset_mesh_path=dataset_mesh_path,
        object_scale=object_scale,
        object_qpos=object_qpos,
        obstacle_configs=obstacle_configs,
        robots="Kinova3",
        has_renderer=has_renderer,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        gripper_types="Robotiq85Gripper",
        horizon=10000,
    )
    obs = env.reset()
    if has_renderer:
        env.viewer.set_camera(camera_id=-1)
    return env, obs


def set_gripper(step_count, gripper_state, env, obs, data_log=None):
    """Open or close the gripper for a fixed number of control steps.

    Args:
        step_count (int): Number of control steps to apply.
        gripper_state (int): Gripper command, typically `-1` for open or `1` for close.
        env (TargetEnvironment): Active robosuite environment.
        obs (OrderedDict): Latest environment observation.

    Returns:
        OrderedDict: Final observation after the command completes.
    """
    for _ in range(step_count):
        action = np.zeros(7)
        action[-1] = gripper_state
        obs = _step_and_render(
            env,
            action,
            data_log=data_log,
            command_name="set_gripper",
        )
    return obs


def move_ee_to_position(step_count, goal_pos, gripper_state, env, obs, data_log=None):
    """Move the end effector toward a Cartesian position with PID control.

    Args:
        step_count (int): Number of controller iterations to run.
        goal_pos (list | np.ndarray): Target end-effector position `[x, y, z]`.
        gripper_state (int): Gripper command applied during the move.
        env (TargetEnvironment): Active robosuite environment.
        obs (OrderedDict): Latest environment observation.

    Returns:
        OrderedDict: Final observation after the move completes.
    """
    integral_error = np.zeros(3)
    prev_error = np.zeros(3)
    for _ in range(step_count):
        action = np.zeros(7)
        position_error = np.array(goal_pos) - obs["robot0_eef_pos"]

        # PID calculations, assuming time step is 1
        integral_error += position_error
        derivative_error = position_error - prev_error
        prev_error = position_error

        action[:3] = np.clip(
            K_P * position_error + K_I * integral_error + K_D * derivative_error,
            -MAX_POS_ACTION,
            MAX_POS_ACTION,
        )
        action[-1] = gripper_state  # Gripper state: -1 for open, 1 for closed

        obs = _step_and_render(
            env,
            action,
            data_log=data_log,
            command_name="move_ee_to_position",
        )

    print("Final positional error:", position_error)
    return obs


def rotate_ee_to_orientation(
    step_count, goal_quat, gripper_state, env, obs, data_log=None
):
    """Rotate the end effector toward a target quaternion with PID control.

    Args:
        step_count (int): Number of controller iterations to run.
        goal_quat (list | np.ndarray): Target quaternion in robosuite order `[x, y, z, w]`.
        gripper_state (int): Gripper command applied during the move.
        env (TargetEnvironment): Active robosuite environment.
        obs (OrderedDict): Latest environment observation.

    Returns:
        OrderedDict: Final observation after the rotation completes.
    """
    integral_error = np.zeros(3)
    prev_error = np.zeros(3)
    for _ in range(step_count):
        angular_error = get_orientation_error(goal_quat, obs["robot0_eef_quat"])

        # PID calculations, assuming time step is 1
        integral_error += angular_error
        derivative_error = angular_error - prev_error
        prev_error = angular_error

        action = np.zeros(7)
        action[3:6] = np.clip(
            K_P * angular_error + K_I * integral_error + K_D * derivative_error,
            -MAX_ROT_ACTION,
            MAX_ROT_ACTION,
        )
        action[-1] = gripper_state  # Open
        obs = _step_and_render(
            env,
            action,
            data_log=data_log,
            command_name="rotate_ee_to_orientation",
        )

    print("Final angular error:", angular_error)
    return obs


def move_ee_to_pose(
    step_count, goal_pos, goal_quat, gripper_state, env, obs, data_log=None
):
    """Move the end effector toward a target position and orientation.

    Args:
        step_count (int): Number of controller iterations to run.
        goal_pos (list | np.ndarray): Target end-effector position `[x, y, z]`.
        goal_quat (list | np.ndarray): Target quaternion in robosuite order `[x, y, z, w]`.
        gripper_state (int): Gripper command applied during the move.
        env (TargetEnvironment): Active robosuite environment.
        obs (OrderedDict): Latest environment observation.

    Returns:
        OrderedDict: Final observation after the move completes.
    """
    integral_pos = np.zeros(3)
    prev_pos = np.zeros(3)
    integral_rot = np.zeros(3)
    prev_rot = np.zeros(3)

    for _ in range(step_count):
        pos_error = np.array(goal_pos) - obs["robot0_eef_pos"]
        rot_error = get_orientation_error(goal_quat, obs["robot0_eef_quat"])

        integral_pos += pos_error
        derivative_pos = pos_error - prev_pos
        prev_pos = pos_error

        integral_rot += rot_error
        derivative_rot = rot_error - prev_rot
        prev_rot = rot_error

        action = np.zeros(7)
        action[:3] = np.clip(
            K_P * pos_error + K_I * integral_pos + K_D * derivative_pos,
            -MAX_POS_ACTION,
            MAX_POS_ACTION,
        )
        action[3:6] = np.clip(
            K_P * rot_error + K_I * integral_rot + K_D * derivative_rot,
            -MAX_ROT_ACTION,
            MAX_ROT_ACTION,
        )
        action[-1] = gripper_state

        obs = _step_and_render(
            env,
            action,
            data_log=data_log,
            command_name="move_ee_to_pose",
        )

    return obs


def move_ee_along_trajectory(
    trajectory_points, step_count, gripper_state, env, obs, data_log=None
):
    """Move the end effector along an interpolated waypoint trajectory.

    Args:
        trajectory_points (list[list[float]] | np.ndarray): Waypoints defining the path.
        step_count (int): Number of controller iterations to run.
        gripper_state (int): Gripper command applied during the move.
        env (TargetEnvironment): Active robosuite environment.
        obs (OrderedDict): Latest environment observation.

    Returns:
        OrderedDict: Final observation after traversing the path.
    """
    integral_error = np.zeros(3)
    prev_error = np.zeros(3)

    trajectory_points = np.asarray(trajectory_points, dtype=float)
    if len(trajectory_points) < 2:
        raise ValueError("must contain at least two waypoints")

    # Create cubic splines for x, y, z
    t_points = np.arange(len(trajectory_points))
    t_interp = np.linspace(0, len(trajectory_points) - 1, step_count)
    cs_x = CubicSpline(t_points, trajectory_points[:, 0])
    cs_y = CubicSpline(t_points, trajectory_points[:, 1])
    cs_z = CubicSpline(t_points, trajectory_points[:, 2])

    for i in range(step_count):
        goal_pos = np.array([cs_x(t_interp[i]), cs_y(t_interp[i]), cs_z(t_interp[i])])
        gripper_pos = obs["robot0_eef_pos"]

        action = np.zeros(7)
        position_error = goal_pos - gripper_pos

        # PID calculations, assuming time step is 1
        integral_error += position_error
        derivative_error = position_error - prev_error
        prev_error = position_error

        action[:3] = (
            K_P * position_error + K_I * integral_error + K_D * derivative_error
        )
        action[-1] = gripper_state  # Closed

        obs = _step_and_render(
            env,
            action,
            data_log=data_log,
            command_name="move_ee_along_trajectory",
        )

    return obs
