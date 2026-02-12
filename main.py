import numpy as np
import robosuite as suite
from robosuite.utils.transform_utils import get_orientation_error
from scipy.interpolate import CubicSpline

from environments import TargetEnvironment

# PID control gains
K_P = 10
K_I = 0
K_D = 0


def initialize_environment():
    """Handles all setup tasks for simulation environment.

    Returns:
        TargetEnvironment: Initialized environment
        OrderedDict: Environment observation space
    """
    # Initialization
    env = TargetEnvironment(
        robots="Panda",
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


def task_command_chain(env, obs):
    """Makes the robot undergo a chain of commands

    Args:
        env (TargetEnvironment): Robot's environment
        obs (OrderedDict): Environment observation space
    """
    # Move end effector high above the table to avoid collisions during initial approach.
    print("Moving end effector to initial position above the table")
    start_position = [0.1, 0.1, 1.5]
    obs = move_ee_to_position(100, start_position, -1, env, obs)

    # Rotate end effector to desired orientation for grasping
    # TODO: This is currently a hardcoded orientation that works for the cylinder object. We need to compute this based on the shape and pose of the object in the future.
    print("Rotating end effector to desired orientation for grasping")
    ee_grab_orientation = [
        0.5,
        0.5,
        -0.5,
        0.5,
    ]  # Quaternion for desired gripper orientation
    obs = rotate_ee_to_orientation(100, ee_grab_orientation, -1, env, obs)

    # Move to initial grab position
    print("Moving end effector to initial grab position")
    ee_grab_pos = [-0.2, 0.0, 0.9]
    obs = move_ee_to_position(100, ee_grab_pos, -1, env, obs)

    # Close gripper to grasp object
    print("Reached initial position. Closing gripper to grasp object")
    for _ in range(50):
        action = np.zeros(7)
        action[-1] = 1  # Closed
        obs, reward, done, info = env.step(action)
        env.render()

    # Move gripper along a trajectory defined by splines through custom points.
    # TODO: This is currently a hardcoded trajectory that works for the cylinder object. We need to compute this based on obstacles in the future.
    print("Gripper closed. Moving along trajectory")
    trajectory_points = np.array(
        [
            [0.0, 0.0, 1.0],
            [0.1, 0.0, 1.0],
            [0.1, 0.1, 1.0],
            [0.0, 0.1, 1.0],
            [-0.1, 0.1, 1.0],
            [-0.1, 0.0, 1.0],
            [0.2, 0.0, 1.0],
        ]
    )
    obs = move_ee_along_trajectory(trajectory_points, 500, 1, env, obs)

    # Open gripper to release object
    print("Trajectory complete. Opening gripper to release object")
    for _ in range(50):
        action = np.zeros(7)
        action[-1] = -1  # Open

        obs, reward, done, info = env.step(action)
        env.render()

    print("Gripper opened, task complete.")


def move_ee_to_position(step_count, goal_pos, gripper_state, env, obs):
    """Moves the end effector to an X,Y,Z position using a PID controller. The gripper state can be set to open or closed during the movement.

    Args:
        step_count (int): Number of steps to take to reach the goal position
        goal_pos (list or np.ndarray): Target position for the end effector (x, y, z)
        gripper_state (int): Gripper state, -1 for open and 1 for closed
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

        obs, reward, done, info = env.step(action)
        env.render()

    print("Final positional error:", position_error)
    return obs


def rotate_ee_to_orientation(step_count, goal_pos, gripper_state, env, obs):
    """Rotates the end effector to an quaternion position using a PID controller. The gripper state can be set to open or closed during the movement.

    Args:
        step_count (int): Number of steps to take to reach the goal position
        goal_pos (list or np.ndarray): Target quaternion for the end effector (x, y, z, w)
        gripper_state (int): Gripper state, -1 for open and 1 for closed
    """
    integral_error = np.zeros(3)
    prev_error = np.zeros(3)
    for _ in range(step_count):
        angular_error = get_orientation_error(goal_pos, obs["robot0_eef_quat"])

        # PID calculations, assuming time step is 1
        integral_error += angular_error
        derivative_error = angular_error - prev_error
        prev_error = angular_error

        action = np.zeros(7)
        action[3:6] = (
            K_P * angular_error + K_I * integral_error + K_D * derivative_error
        )
        action[-1] = gripper_state  # Open
        obs, reward, done, info = env.step(action)
        env.render()

    print("Final angular error:", angular_error)
    return obs


def move_ee_along_trajectory(trajectory_points, step_count, gripper_state, env, obs):
    """Moves the end effector along a trajectory defined by a list of waypoints using a PID controller. The gripper state can be set to open or closed during the movement.

    Args:
        trajectory_points (list of lists or np.ndarray): List of waypoints defining the trajectory, where each waypoint is an (x, y, z) position.
        step_count (int): Number of steps to take to reach the goal position
        gripper_state (int): Gripper state, -1 for open and 1 for closed
    """
    integral_error = np.zeros(3)
    prev_error = np.zeros(3)

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

        obs, reward, done, info = env.step(action)
        env.render()

    return obs


if __name__ == "__main__":
    env, obs = initialize_environment()
    task_command_chain(env, obs)
