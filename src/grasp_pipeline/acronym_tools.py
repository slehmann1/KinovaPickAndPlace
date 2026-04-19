from __future__ import annotations

import argparse
from pathlib import Path
import sys

# Allow direct execution via `python .\src\grasp_pipeline\acronym_tools.py`.
if __package__ in {None, ""}:
    PROJECT_ROOT = Path(__file__).resolve().parents[2]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

from src.grasp_pipeline.acronym import AcronymParser, dump_acronym_inspection


def build_arg_parser():
    """Create the CLI used to inspect an ACRONYM dataset installation."""
    parser = argparse.ArgumentParser(
        description="Inspect ACRONYM grasp files and resolve their meshes."
    )
    parser.add_argument(
        "--root-dir",
        required=True,
        help="Root directory containing ACRONYM grasp files.",
    )
    parser.add_argument(
        "--mesh-root",
        default=None,
        help="Optional separate mesh root directory, such as a ShapeNetSem export.",
    )
    parser.add_argument(
        "--object-index",
        type=int,
        default=0,
        help="Zero-based grasp-file index to inspect.",
    )
    return parser


def main():
    """CLI entry point for inspecting ACRONYM data / mesh layout."""
    args = build_arg_parser().parse_args()
    parser = AcronymParser(root_dir=args.root_dir, mesh_root=args.mesh_root)
    print(f"Indexed ACRONYM grasp files: {len(parser)}")
    print(dump_acronym_inspection(parser, object_index=args.object_index))


if __name__ == "__main__":
    main()
