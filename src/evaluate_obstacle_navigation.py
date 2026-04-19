from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import numpy as np

from src.pick_place_obstacle_rrt import (
    PLANNING_BOUNDS,
    run_demo_with_obstacles,
    yaw_degrees_to_quat,
)

OBSTACLE_Z = 0.92
OBSTACLE_SIZE_CHOICES = [
    [0.04, 0.04, 0.10],
    [0.05, 0.05, 0.10],
    [0.06, 0.04, 0.10],
]
START_CLEARANCE_POINT = np.array([0.10, -0.22], dtype=float)
OBJECT_CLEARANCE_POINT = np.array([-0.20, 0.00], dtype=float)
TARGET_CLEARANCE_POINT = np.array([0.20, 0.00], dtype=float)
KEEP_OUT_RADIUS = 0.10


def _within_keep_out(candidate_xy: np.ndarray, protected_xy: np.ndarray, radius: float) -> bool:
    return float(np.linalg.norm(candidate_xy - protected_xy)) < radius


def sample_random_obstacles(
    rng: np.random.Generator,
    obstacle_count: int,
) -> list[dict]:
    obstacles = []
    attempts = 0
    while len(obstacles) < obstacle_count:
        attempts += 1
        if attempts > 3000:
            raise RuntimeError("Failed to sample a valid obstacle layout.")

        candidate_xy = np.array(
            [
                rng.uniform(PLANNING_BOUNDS[0, 0] + 0.04, PLANNING_BOUNDS[0, 1] - 0.04),
                rng.uniform(PLANNING_BOUNDS[1, 0] + 0.04, PLANNING_BOUNDS[1, 1] - 0.04),
            ],
            dtype=float,
        )
        if _within_keep_out(candidate_xy, START_CLEARANCE_POINT, KEEP_OUT_RADIUS):
            continue
        if _within_keep_out(candidate_xy, OBJECT_CLEARANCE_POINT, KEEP_OUT_RADIUS):
            continue
        if _within_keep_out(candidate_xy, TARGET_CLEARANCE_POINT, KEEP_OUT_RADIUS):
            continue

        size = copy.deepcopy(
            OBSTACLE_SIZE_CHOICES[int(rng.integers(0, len(OBSTACLE_SIZE_CHOICES)))]
        )
        obstacle = {
            "pos": [float(candidate_xy[0]), float(candidate_xy[1]), OBSTACLE_Z],
            "size": size,
            "quat": yaw_degrees_to_quat(float(rng.uniform(0.0, 90.0))),
            "rgba": [
                float(rng.uniform(0.1, 0.9)),
                float(rng.uniform(0.1, 0.9)),
                float(rng.uniform(0.1, 0.9)),
                1.0,
            ],
        }

        overlaps_existing = False
        for existing in obstacles:
            existing_xy = np.asarray(existing["pos"][:2], dtype=float)
            min_distance = max(
                existing["size"][0] + obstacle["size"][0],
                existing["size"][1] + obstacle["size"][1],
            ) + 0.02
            if np.linalg.norm(candidate_xy - existing_xy) < min_distance:
                overlaps_existing = True
                break
        if overlaps_existing:
            continue

        obstacles.append(obstacle)

    return obstacles


def run_evaluation(scene_count: int, seed: int, obstacle_count: int, has_renderer: bool) -> dict:
    rng = np.random.default_rng(seed)
    results = []

    for scene_index in range(scene_count):
        obstacles = sample_random_obstacles(rng, obstacle_count=obstacle_count)
        result = run_demo_with_obstacles(
            obstacle_configs=obstacles,
            has_renderer=has_renderer,
        )
        result["scene_index"] = scene_index
        results.append(result)

    success_count = sum(1 for result in results if result["success"])
    return {
        "scene_count": scene_count,
        "success_count": success_count,
        "success_rate": success_count / scene_count if scene_count else 0.0,
        "seed": seed,
        "obstacle_count": obstacle_count,
        "results": results,
    }


def write_report(report: dict, output_path: str | None) -> Path:
    if output_path is None:
        output_path = "data/evaluations/obstacle_navigation_eval.json"
    path = Path(output_path)
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        path = project_root / path
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Evaluate classical obstacle-aware pick-and-place across randomized scenes."
    )
    parser.add_argument(
        "--scenes",
        type=int,
        default=10,
        help="Number of randomized scenes to run.",
    )
    parser.add_argument("--seed", type=int, default=7, help="Random seed for obstacle generation.")
    parser.add_argument(
        "--obstacle-count",
        type=int,
        default=2,
        help="Number of obstacles per scene.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render each evaluation scene. Use mjpython on macOS.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path for the evaluation report.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_evaluation(
        scene_count=args.scenes,
        seed=args.seed,
        obstacle_count=args.obstacle_count,
        has_renderer=args.render,
    )
    report_path = write_report(report, args.output)
    print(f"Scenes run: {report['scene_count']}")
    print(f"Successes: {report['success_count']}")
    print(f"Success rate: {report['success_rate']:.2%}")
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
