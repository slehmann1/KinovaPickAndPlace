import numpy as np
from robosuite.utils.transform_utils import get_orientation_error
from scipy.interpolate import CubicSpline

from src.environments import TargetEnvironment

K_P = 10
K_I = 0
K_D = 0


def _step_and_render(env, action):
    """Advance the environment by one action and render the viewer.

    Args:
        env (TargetEnvironment): Active robosuite environment.
        action (np.ndarray): Action vector to apply for one simulation step.

    Returns:
        OrderedDict: Updated observation from the environment.
    """
    obs, _, _, _ = env.step(action)
    env.render()
    return obs


def initialize_environment(
    dataset_mesh_path=None,
    object_qpos=None,
    object_scale=(1.0, 1.0, 1.0),
):
    """Create and reset the demo environment.

    Args:
        dataset_mesh_path (str | None): Optional dataset mesh to load in the scene.
        object_qpos (sequence[float] | None): Optional object pose override.
        object_scale (tuple[float, float, float]): Per-axis mesh scale for dataset objects.

    Returns:
        tuple[TargetEnvironment, OrderedDict]: Initialized environment and first observation.
    """
    env = TargetEnvironment(
        dataset_mesh_path=dataset_mesh_path,
        object_scale=object_scale,
        object_qpos=object_qpos,
        robots="Kinova3",
        has_renderer=True,
        has_offscreen_renderer=False,
        use_camera_obs=False,
        control_freq=20,
        gripper_types="Robotiq85Gripper",
        horizon=10000,
    )
    obs = env.reset()
    env.viewer.set_camera(camera_id=-1)
    return env, obs


def set_gripper(step_count, gripper_state, env, obs):
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
        obs = _step_and_render(env, action)
    return obs


def move_ee_to_position(step_count, goal_pos, gripper_state, env, obs):
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

        action[:3] = (
            K_P * position_error + K_I * integral_error + K_D * derivative_error
        )
        action[-1] = gripper_state  # Gripper state: -1 for open, 1 for closed

        obs = _step_and_render(env, action)

    print("Final positional error:", position_error)
    return obs


def rotate_ee_to_orientation(step_count, goal_quat, gripper_state, env, obs):
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
        action[3:6] = (
            K_P * angular_error + K_I * integral_error + K_D * derivative_error
        )
        action[-1] = gripper_state  # Open
        obs = _step_and_render(env, action)

    print("Final angular error:", angular_error)
    return obs


def move_ee_to_pose(step_count, goal_pos, goal_quat, gripper_state, env, obs):
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
        action[:3] = K_P * pos_error + K_I * integral_pos + K_D * derivative_pos
        action[3:6] = K_P * rot_error + K_I * integral_rot + K_D * derivative_rot
        action[-1] = gripper_state

        obs = _step_and_render(env, action)

    return obs


def move_ee_along_trajectory(trajectory_points, step_count, gripper_state, env, obs):
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
    if trajectory_points.ndim != 2 or trajectory_points.shape[1] != 3:
        raise ValueError(
            "trajectory_points must have shape (n, 3) for Cartesian waypoints"
        )
    if len(trajectory_points) < 2:
        raise ValueError("trajectory_points must contain at least two waypoints")

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

        obs = _step_and_render(env, action)

    return obs
