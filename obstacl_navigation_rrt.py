from obstacle_navigation_rrt import should_enable_renderer
from src.obstacle_navigation_rrt import (
    DEFAULT_OBSTACLES,
    apply_obstacle_overrides,
    run_demo_with_obstacles,
)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run the obstacle navigation RRT demo.")
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
    args = parser.parse_args()
    obstacle_configs = apply_obstacle_overrides(
        base_obstacles=DEFAULT_OBSTACLES,
        obstacle_positions=args.obstacle_pos,
        obstacle_yaws=args.obstacle_yaw,
    )
    run_demo_with_obstacles(
        obstacle_configs=obstacle_configs,
        has_renderer=should_enable_renderer(args.headless, args.render),
    )
