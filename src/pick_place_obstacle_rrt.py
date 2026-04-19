from __future__ import annotations

import argparse
import copy

import numpy as np

from src.controllers import arm_controller
from src.cli_utils import should_enable_renderer
from src.obstacle_navigation_rrt import (
    SAFE_Z,
    apply_obstacle_overrides,
    plan_rrt_path,
    xy_path_to_cartesian,
    yaw_degrees_to_quat,
)

GRASP_Z_OFFSET = 0.03
POST_PLACE_LIFT = 0.12
START_POS = np.array([0.10, -0.22, SAFE_Z], dtype=float)
PLANNING_BOUNDS = np.array([[-0.20, 0.30], [-0.30, 0.30]], dtype=float)
PLACE_Z_OFFSET = 0.08
PLACE_CLEARANCE_RADIUS = 0.08
DEFAULT_PICK_PLACE_OBSTACLES = [
    {
        "pos": [0.02, -0.08, 0.92],
        "size": [0.05, 0.05, 0.10],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "rgba": [0.85, 0.35, 0.2, 1.0],
    },
    {
        "pos": [0.10, 0.14, 0.92],
        "size": [0.05, 0.05, 0.10],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "rgba": [0.2, 0.45, 0.85, 1.0],
    },
]


def get_object_position(env) -> np.ndarray:
    object_pos, _ = env.get_object_pose()
    return np.asarray(object_pos, dtype=float)


def get_target_position(env) -> np.ndarray:
    return env.sim.data.body_xpos[env.target_zone_body_id].copy()


def point_clear_of_obstacles(
    point_xy: np.ndarray,
    obstacle_configs: list[dict],
    clearance_radius: float,
) -> bool:
    for obstacle in obstacle_configs:
        obstacle_xy = np.asarray(obstacle["pos"][:2], dtype=float)
        padded_half_extents = np.asarray(obstacle["size"][:2], dtype=float) + clearance_radius
        lower = obstacle_xy - padded_half_extents
        upper = obstacle_xy + padded_half_extents
        if np.all(point_xy >= lower) and np.all(point_xy <= upper):
            return False
    return True


def choose_safe_place_xy(preferred_xy: np.ndarray, obstacle_configs: list[dict]) -> np.ndarray:
    if point_clear_of_obstacles(preferred_xy, obstacle_configs, PLACE_CLEARANCE_RADIUS):
        return preferred_xy

    x_offsets = np.linspace(-0.12, 0.12, 7)
    y_offsets = np.linspace(-0.12, 0.12, 7)
    candidates = []
    for x_offset in x_offsets:
        for y_offset in y_offsets:
            candidate = preferred_xy + np.array([x_offset, y_offset], dtype=float)
            if not (
                PLANNING_BOUNDS[0, 0] <= candidate[0] <= PLANNING_BOUNDS[0, 1]
                and PLANNING_BOUNDS[1, 0] <= candidate[1] <= PLANNING_BOUNDS[1, 1]
            ):
                continue
            if point_clear_of_obstacles(candidate, obstacle_configs, PLACE_CLEARANCE_RADIUS):
                candidates.append(candidate)

    if not candidates:
        raise RuntimeError(
            "Unable to find an obstacle-free placement point near the target zone."
        )

    return min(candidates, key=lambda candidate: np.linalg.norm(candidate - preferred_xy))


def set_target_zone_xy(env, target_xy: np.ndarray):
    env.sim.model.body_pos[env.target_zone_body_id][:2] = np.asarray(target_xy, dtype=float)
    env.sim.forward()


def plan_cartesian_rrt(
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    obstacle_configs: list[dict],
) -> np.ndarray:
    path_xy = plan_rrt_path(
        start_xy=np.asarray(start_pos[:2], dtype=float),
        goal_xy=np.asarray(goal_pos[:2], dtype=float),
        obstacles=obstacle_configs,
        bounds=PLANNING_BOUNDS,
    )
    return xy_path_to_cartesian(path_xy, SAFE_Z)


def move_along_rrt(
    env,
    obs,
    start_pos: np.ndarray,
    goal_pos: np.ndarray,
    gripper_state: int,
    obstacle_configs: list[dict],
):
    trajectory = plan_cartesian_rrt(start_pos, goal_pos, obstacle_configs)
    print("Planned transport waypoints:")
    for waypoint in trajectory:
        print(waypoint.tolist())

    obs = arm_controller.move_ee_along_trajectory(
        trajectory_points=trajectory,
        step_count=max(240, len(trajectory) * 80),
        gripper_state=gripper_state,
        env=env,
        obs=obs,
    )
    return obs


def align_for_place(env, obs, target_pos: np.ndarray):
    """Refine the final placement pose directly above and then into the target zone."""
    preplace_pos = np.array([target_pos[0], target_pos[1], SAFE_Z], dtype=float)
    place_pos = np.array(
        [target_pos[0], target_pos[1], target_pos[2] + PLACE_Z_OFFSET],
        dtype=float,
    )

    print("Refining alignment above target zone...")
    obs = arm_controller.move_ee_to_position(
        step_count=140,
        goal_pos=preplace_pos,
        gripper_state=1,
        env=env,
        obs=obs,
    )

    print("Lowering with final place alignment...")
    obs = arm_controller.move_ee_to_position(
        step_count=180,
        goal_pos=place_pos,
        gripper_state=1,
        env=env,
        obs=obs,
    )
    return obs


def evaluate_scene_result(env) -> dict:
    object_pos = get_object_position(env)
    target_pos = get_target_position(env)
    xy_error = float(np.linalg.norm(object_pos[:2] - target_pos[:2]))
    z_error = float(abs(object_pos[2] - target_pos[2]))
    success = bool(env._check_success())
    return {
        "success": success,
        "object_position": object_pos.tolist(),
        "target_position": target_pos.tolist(),
        "xy_error": xy_error,
        "z_error": z_error,
    }


def run_demo_with_obstacles(obstacle_configs: list[dict], has_renderer: bool = True) -> dict:
    env, obs = arm_controller.initialize_environment(
        has_renderer=has_renderer,
        obstacle_configs=obstacle_configs,
    )

    try:
        object_pos = get_object_position(env)
        target_pos = get_target_position(env)
        preferred_target_xy = target_pos[:2].copy()
        place_target_xy = choose_safe_place_xy(preferred_target_xy, obstacle_configs)
        set_target_zone_xy(env, place_target_xy)
        target_pos = get_target_position(env)
        print("Selected obstacle-free placement target:", target_pos.tolist())

        pregrasp_pos = np.array([object_pos[0], object_pos[1], SAFE_Z], dtype=float)
        grasp_pos = np.array(
            [object_pos[0], object_pos[1], object_pos[2] + GRASP_Z_OFFSET],
            dtype=float,
        )
        retreat_pos = np.array(
            [target_pos[0], target_pos[1], SAFE_Z + POST_PLACE_LIFT],
            dtype=float,
        )

        print("Moving to safe start...")
        obs = arm_controller.move_ee_to_position(
            step_count=120,
            goal_pos=START_POS,
            gripper_state=-1,
            env=env,
            obs=obs,
        )

        print("Navigating to pre-grasp above the object...")
        obs = move_along_rrt(
            env=env,
            obs=obs,
            start_pos=START_POS,
            goal_pos=pregrasp_pos,
            gripper_state=-1,
            obstacle_configs=obstacle_configs,
        )

        print("Descending to grasp...")
        obs = arm_controller.move_ee_to_position(
            step_count=120,
            goal_pos=grasp_pos,
            gripper_state=-1,
            env=env,
            obs=obs,
        )

        print("Closing gripper...")
        obs = arm_controller.set_gripper(
            step_count=50,
            gripper_state=1,
            env=env,
            obs=obs,
        )

        print("Lifting object...")
        obs = arm_controller.move_ee_to_position(
            step_count=120,
            goal_pos=pregrasp_pos,
            gripper_state=1,
            env=env,
            obs=obs,
        )

        print("Transporting object to target while avoiding obstacles...")
        obs = move_along_rrt(
            env=env,
            obs=obs,
            start_pos=pregrasp_pos,
            goal_pos=np.array([target_pos[0], target_pos[1], SAFE_Z], dtype=float),
            gripper_state=1,
            obstacle_configs=obstacle_configs,
        )

        target_pos = get_target_position(env)
        obs = align_for_place(env, obs, target_pos)

        print("Opening gripper...")
        obs = arm_controller.set_gripper(
            step_count=50,
            gripper_state=-1,
            env=env,
            obs=obs,
        )

        print("Retreating upward...")
        obs = arm_controller.move_ee_to_position(
            step_count=120,
            goal_pos=retreat_pos,
            gripper_state=-1,
            env=env,
            obs=obs,
        )
        result = evaluate_scene_result(env)
        result["placement_target"] = target_pos.tolist()
        result["obstacles"] = copy.deepcopy(obstacle_configs)
        return result
    finally:
        env.close()


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run pick-and-place while using RRT to avoid obstacles during transport."
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the MuJoCo viewer and run without rendering.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Force-enable the MuJoCo viewer. Use this with mjpython on macOS.",
    )
    parser.add_argument(
        "--obstacle-pos",
        action="append",
        nargs=3,
        type=float,
        metavar=("X", "Y", "Z"),
        help="Override one obstacle position. Pass once per obstacle, in order.",
    )
    parser.add_argument(
        "--obstacle-yaw",
        action="append",
        type=float,
        metavar="DEGREES",
        help="Override one obstacle yaw angle in degrees. Pass once per obstacle, in order.",
    )
    return parser


def main():
    args = build_arg_parser().parse_args()
    obstacle_configs = apply_obstacle_overrides(
        base_obstacles=copy.deepcopy(DEFAULT_PICK_PLACE_OBSTACLES),
        obstacle_positions=args.obstacle_pos,
        obstacle_yaws=args.obstacle_yaw,
    )
    run_demo_with_obstacles(
        obstacle_configs=obstacle_configs,
        has_renderer=should_enable_renderer(args.headless, args.render),
    )


if __name__ == "__main__":
    main()
