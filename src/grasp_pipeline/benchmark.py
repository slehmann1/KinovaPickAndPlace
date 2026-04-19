from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow direct execution via `python .\src\grasp_pipeline\benchmark.py` by adding the
# project root to `sys.path` before importing the package modules.
if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from src.grasp_pipeline.config import ExecutionConfig, SceneConfig
from src.grasp_pipeline.runner import run_grasp_trials
from src.grasp_pipeline.sources import AcronymSource, GraspFactorySource


def build_arg_parser():
    """Create the CLI used to benchmark scripted grasp execution."""
    parser = argparse.ArgumentParser(
        description="Run a scripted grasp benchmark for a dataset object."
    )
    parser.add_argument(
        "--source",
        choices=("graspfactory", "acronym"),
        default="graspfactory",
        help="Dataset adapter to use for loading known grasps.",
    )
    parser.add_argument(
        "--root-dir",
        default=None,
        help="Optional dataset root directory override for the selected source.",
    )
    parser.add_argument(
        "--mesh-root",
        default=None,
        help="Optional separate mesh root directory for datasets such as ACRONYM.",
    )
    parser.add_argument(
        "--object-index",
        type=int,
        default=0,
        help="Zero-based object index inside the selected dataset source.",
    )
    parser.add_argument(
        "--max-grasps",
        type=int,
        default=5,
        help="Maximum number of candidate grasps to execute for the selected object.",
    )
    parser.add_argument(
        "--collision-mode",
        choices=("box", "convex_hull", "mesh"),
        default="box",
        help="Collision approximation used during physics execution.",
    )
    parser.add_argument(
        "--object-drop-height",
        type=float,
        default=0.05,
        help="Spawn height above the table before settling.",
    )
    parser.add_argument(
        "--object-scale",
        type=float,
        nargs=3,
        default=None,
        metavar=("SX", "SY", "SZ"),
        help="Optional explicit object scale override. When omitted, the dataset "
        "adapter can provide a per-object scale.",
    )
    parser.add_argument(
        "--output-path",
        default=str(Path("data") / "benchmark_runs" / "grasp_trials.jsonl"),
        help="JSONL file used to append benchmark trial records.",
    )
    return parser


def _build_source(args):
    """Instantiate the requested dataset adapter."""
    if args.source == "graspfactory":
        return GraspFactorySource(root_dir=args.root_dir)
    if args.root_dir is None:
        raise ValueError("--root-dir is required when --source acronym is selected")
    return AcronymSource(root_dir=args.root_dir, mesh_root=args.mesh_root)


def main():
    """CLI entry point for the scripted grasp benchmarking pipeline."""
    args = build_arg_parser().parse_args()
    source = _build_source(args)

    scene_config = SceneConfig(
        object_scale=None if args.object_scale is None else tuple(args.object_scale),
        collision_mode=args.collision_mode,
        object_drop_height=args.object_drop_height,
    )
    execution_config = ExecutionConfig()

    results = run_grasp_trials(
        source=source,
        object_index=args.object_index,
        max_grasps=args.max_grasps,
        output_path=args.output_path,
        scene_config=scene_config,
        execution_config=execution_config,
    )

    success_count = sum(record.success for record in results)
    print(f"Executed {len(results)} grasp trials")
    print(f"Successful trials: {success_count}")
    print(f"Output log: {Path(args.output_path).resolve()}")


if __name__ == "__main__":
    main()
