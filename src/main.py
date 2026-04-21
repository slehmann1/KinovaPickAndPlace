import numpy as np
from src.controllers import arm_controller
import argparse


def task_command_chain(env, obs):
    """Makes the robot undergo a chain of commands

    Args:
        env (TargetEnvironment): Robot's environment
        obs (OrderedDict): Environment observation space
    """
    # Move end effector high above the table to avoid collisions during initial approach.
    print("Moving end effector to initial position above the table")
    start_position = [0.1, 0.1, 1.5]
    data_log = []
    obs = arm_controller.move_ee_to_position(
        100, start_position, -1, env, obs, data_log=data_log
    )

    # Rotate end effector to desired orientation for grasping
    # TODO: This is currently a hardcoded orientation that works for the cylinder object. We need to compute this based on the shape and pose of the object in the future.
    print("Rotating end effector to desired orientation for grasping")
    ee_grab_orientation = [
        0.5,
        0.5,
        -0.5,
        0.5,
    ]  # Quaternion for desired gripper orientation
    obs = arm_controller.rotate_ee_to_orientation(
        100, ee_grab_orientation, -1, env, obs, data_log=data_log
    )

    # Move to initial grab position
    print("Moving end effector to initial grab position")
    ee_grab_pos = [-0.2, 0.0, 0.9]
    obs = arm_controller.move_ee_to_position(
        100, ee_grab_pos, -1, env, obs, data_log=data_log
    )

    # Close gripper to grasp object
    print("Reached initial position. Closing gripper to grasp object")
    obs = arm_controller.set_gripper(50, 1, env, obs, data_log=data_log)

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
    obs = arm_controller.move_ee_along_trajectory(
        trajectory_points, 500, 1, env, obs, data_log=data_log
    )

    # Open gripper to release object
    print("Trajectory complete. Opening gripper to release object")
    obs = arm_controller.set_gripper(50, -1, env, obs, data_log=data_log)

    log_path = arm_controller.store_all_data(
        data_log,
        metadata={
            "script": "src/main.py",
            "trajectory_points": trajectory_points.tolist(),
            "start_position": start_position,
            "grab_position": ee_grab_pos,
            "grab_orientation": ee_grab_orientation,
        },
    )
    print(f"Stored run data at: {log_path}")
    print("Gripper opened, task complete.")
    return obs


def main(has_renderer=True):
    env, obs = arm_controller.initialize_environment(has_renderer=has_renderer)
    try:
        task_command_chain(env, obs)
    finally:
        env.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the default pick-and-place demo.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the MuJoCo viewer and run without rendering.",
    )
    args = parser.parse_args()
    main(has_renderer=not args.headless)
