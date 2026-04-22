import argparse

from src.main import main as src_main
from src.utils.cli_utils import should_enable_renderer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the default pick-and-place demo.")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the MuJoCo viewer and run without rendering.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Force-enable rendering when supported.",
    )
    args = parser.parse_args()
    src_main(has_renderer=should_enable_renderer(args.headless, args.render))
