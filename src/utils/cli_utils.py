import sys


def should_enable_renderer(requested_headless: bool, requested_render: bool) -> bool:
    if requested_render:
        return True
    if requested_headless:
        return False
    if sys.platform == "darwin" and "mjpython" not in sys.executable:
        print(
            "Renderer disabled: on macOS, run this script with mjpython to open the MuJoCo viewer."
        )
        print("Using headless mode instead.")
        return False
    return True
