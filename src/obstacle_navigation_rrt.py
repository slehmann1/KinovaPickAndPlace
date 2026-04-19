from __future__ import annotations

import argparse
import copy
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from src.controllers import arm_controller
from src.cli_utils import should_enable_renderer

SAFE_Z = 1.05
START_POS = np.array([0.10, -0.20, SAFE_Z], dtype=float)
GOAL_POS = np.array([0.22, 0.18, SAFE_Z], dtype=float)
DEFAULT_BOUNDS = np.array([[-0.05, 0.30], [-0.30, 0.30]], dtype=float)
DEFAULT_OBSTACLES = [
    {
        "pos": [0.10, -0.02, 0.92],
        "size": [0.05, 0.05, 0.10],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "rgba": [0.85, 0.35, 0.2, 1.0],
    },
    {
        "pos": [0.17, 0.08, 0.92],
        "size": [0.04, 0.06, 0.10],
        "quat": [0.9238795, 0.0, 0.0, 0.3826834],
        "rgba": [0.2, 0.45, 0.85, 1.0],
    },
]


@dataclass
class RRTNode:
    point: np.ndarray
    parent_index: int | None


def _xy(point_3d: Iterable[float]) -> np.ndarray:
    point = np.asarray(point_3d, dtype=float)
    return point[:2]


def _obstacle_padding(obstacle: dict, robot_radius: float) -> tuple[np.ndarray, np.ndarray]:
    center = _xy(obstacle["pos"])
    half_extents = np.asarray(obstacle["size"][:2], dtype=float) + robot_radius
    return center - half_extents, center + half_extents


def point_in_collision(point_xy: np.ndarray, obstacles: list[dict], robot_radius: float) -> bool:
    for obstacle in obstacles:
        lower, upper = _obstacle_padding(obstacle, robot_radius)
        if np.all(point_xy >= lower) and np.all(point_xy <= upper):
            return True
    return False


def segment_in_collision(
    start_xy: np.ndarray,
    end_xy: np.ndarray,
    obstacles: list[dict],
    robot_radius: float,
    samples: int = 30,
) -> bool:
    for alpha in np.linspace(0.0, 1.0, samples):
        point = start_xy + alpha * (end_xy - start_xy)
        if point_in_collision(point, obstacles, robot_radius):
            return True
    return False


def sample_free_point(
    rng: np.random.Generator,
    bounds: np.ndarray,
    obstacles: list[dict],
    robot_radius: float,
) -> np.ndarray:
    while True:
        candidate = np.array(
            [
                rng.uniform(bounds[0, 0], bounds[0, 1]),
                rng.uniform(bounds[1, 0], bounds[1, 1]),
            ],
            dtype=float,
        )
        if not point_in_collision(candidate, obstacles, robot_radius):
            return candidate


def reconstruct_path(nodes: list[RRTNode], goal_index: int) -> np.ndarray:
    points = []
    current_index = goal_index
    while current_index is not None:
        node = nodes[current_index]
        points.append(node.point)
        current_index = node.parent_index
    points.reverse()
    return np.asarray(points, dtype=float)


def simplify_path(path_xy: np.ndarray, obstacles: list[dict], robot_radius: float) -> np.ndarray:
    if len(path_xy) <= 2:
        return path_xy

    simplified = [path_xy[0]]
    anchor_index = 0
    while anchor_index < len(path_xy) - 1:
        next_index = len(path_xy) - 1
        while next_index > anchor_index + 1:
            if not segment_in_collision(
                path_xy[anchor_index],
                path_xy[next_index],
                obstacles,
                robot_radius,
            ):
                break
            next_index -= 1
        simplified.append(path_xy[next_index])
        anchor_index = next_index

    return np.asarray(simplified, dtype=float)


def plan_rrt_path(
    start_xy: np.ndarray,
    goal_xy: np.ndarray,
    obstacles: list[dict],
    bounds: np.ndarray,
    step_size: float = 0.05,
    max_iterations: int = 3000,
    goal_bias: float = 0.15,
    robot_radius: float = 0.035,
    seed: int = 7,
) -> np.ndarray:
    if point_in_collision(start_xy, obstacles, robot_radius):
        raise ValueError("Start point is inside an obstacle.")
    if point_in_collision(goal_xy, obstacles, robot_radius):
        raise ValueError("Goal point is inside an obstacle.")

    rng = np.random.default_rng(seed)
    nodes = [RRTNode(point=np.asarray(start_xy, dtype=float), parent_index=None)]

    for _ in range(max_iterations):
        if rng.random() < goal_bias:
            sample_xy = goal_xy
        else:
            sample_xy = sample_free_point(rng, bounds, obstacles, robot_radius)

        distances = [np.linalg.norm(node.point - sample_xy) for node in nodes]
        nearest_index = int(np.argmin(distances))
        nearest_xy = nodes[nearest_index].point

        direction = sample_xy - nearest_xy
        distance = np.linalg.norm(direction)
        if distance == 0:
            continue

        step = min(step_size, distance)
        candidate_xy = nearest_xy + (direction / distance) * step

        if segment_in_collision(nearest_xy, candidate_xy, obstacles, robot_radius):
            continue

        nodes.append(RRTNode(point=candidate_xy, parent_index=nearest_index))
        new_index = len(nodes) - 1

        if np.linalg.norm(candidate_xy - goal_xy) <= step_size and not segment_in_collision(
            candidate_xy,
            goal_xy,
            obstacles,
            robot_radius,
        ):
            nodes.append(RRTNode(point=np.asarray(goal_xy, dtype=float), parent_index=new_index))
            return simplify_path(reconstruct_path(nodes, len(nodes) - 1), obstacles, robot_radius)

    raise RuntimeError("RRT failed to find a collision-free path within the iteration limit.")


def xy_path_to_cartesian(path_xy: np.ndarray, z_height: float) -> np.ndarray:
    z_column = np.full((len(path_xy), 1), z_height, dtype=float)
    return np.hstack([path_xy, z_column])


def run_demo(has_renderer: bool = True) -> np.ndarray:
    return run_demo_with_obstacles(DEFAULT_OBSTACLES, has_renderer=has_renderer)


def run_demo_with_obstacles(obstacle_configs: list[dict], has_renderer: bool = True) -> np.ndarray:
    path_xy = plan_rrt_path(
        start_xy=_xy(START_POS),
        goal_xy=_xy(GOAL_POS),
        obstacles=obstacle_configs,
        bounds=DEFAULT_BOUNDS,
    )
    cartesian_path = xy_path_to_cartesian(path_xy, SAFE_Z)

    print("Planned RRT waypoints:")
    for waypoint in cartesian_path:
        print(waypoint.tolist())

    env, obs = arm_controller.initialize_environment(
        has_renderer=has_renderer,
        obstacle_configs=obstacle_configs,
    )

    try:
        obs = arm_controller.move_ee_to_position(
            step_count=120,
            goal_pos=START_POS,
            gripper_state=-1,
            env=env,
            obs=obs,
        )
        obs = arm_controller.move_ee_along_trajectory(
            trajectory_points=cartesian_path,
            step_count=max(240, len(cartesian_path) * 80),
            gripper_state=-1,
            env=env,
            obs=obs,
        )
        obs = arm_controller.move_ee_to_position(
            step_count=80,
            goal_pos=GOAL_POS,
            gripper_state=-1,
            env=env,
            obs=obs,
        )
    finally:
        env.close()

    return cartesian_path


def yaw_degrees_to_quat(yaw_degrees: float) -> list[float]:
    yaw_radians = np.deg2rad(yaw_degrees)
    half_angle = yaw_radians / 2.0
    return [float(np.cos(half_angle)), 0.0, 0.0, float(np.sin(half_angle))]


def apply_obstacle_overrides(
    base_obstacles: list[dict],
    obstacle_positions: list[list[float]] | None,
    obstacle_yaws: list[float] | None,
) -> list[dict]:
    obstacles = copy.deepcopy(base_obstacles)

    if obstacle_positions is not None and len(obstacle_positions) != len(obstacles):
        raise ValueError(
            f"Expected {len(obstacles)} obstacle positions, got {len(obstacle_positions)}."
        )
    if obstacle_yaws is not None and len(obstacle_yaws) != len(obstacles):
        raise ValueError(
            f"Expected {len(obstacles)} obstacle yaw values, got {len(obstacle_yaws)}."
        )

    for index, obstacle in enumerate(obstacles):
        if obstacle_positions is not None:
            obstacle["pos"] = obstacle_positions[index]
        if obstacle_yaws is not None:
            obstacle["quat"] = yaw_degrees_to_quat(obstacle_yaws[index])

    return obstacles


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Plan and execute a simple obstacle-avoiding RRT path.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the MuJoCo viewer and run the demo without rendering.",
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


def main() -> None:
    args = build_arg_parser().parse_args()
    obstacle_configs = apply_obstacle_overrides(
        base_obstacles=DEFAULT_OBSTACLES,
        obstacle_positions=args.obstacle_pos,
        obstacle_yaws=args.obstacle_yaw,
    )
    run_demo_with_obstacles(
        obstacle_configs=obstacle_configs,
        has_renderer=should_enable_renderer(args.headless, args.render),
    )


if __name__ == "__main__":
    main()
