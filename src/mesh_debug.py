import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path

from src.controllers import arm_controller
from src.utils.graspfactory_parser import GraspFactoryParser
from src.utils.dataset_xml_builder import get_mesh_bounds

TEST_QUATS = [
    ("identity", [1, 0, 0, 0]),
    ("rot_x_90", [0.7071, 0.7071, 0, 0]),
    ("rot_y_90", [0.7071, 0, 0.7071, 0]),
    ("rot_z_90", [0.7071, 0, 0, 0.7071]),
    ("rot_x_180", [0, 1, 0, 0]),
    ("rot_y_180", [0, 0, 1, 0]),
    ("rot_z_180", [0, 0, 0, 1]),
]
DEFAULT_COLLISION_MODE = "mesh"
DEFAULT_OBJECT_SCALE = (0.01, 0.01, 0.01)


def _format_scale_tag(scale):
    """Match the generated asset directory naming used by DatasetMeshObject."""
    scale = tuple(float(v) for v in scale)
    return "scale_" + "_".join(f"{v:.6f}".replace(".", "p").replace("-", "m") for v in scale)


def load_generated_geom_pos(
    mesh_path,
    collision_mode=DEFAULT_COLLISION_MODE,
    object_scale=DEFAULT_OBJECT_SCALE,
):
    """Read the local visual-mesh offset written to the generated MJCF file.

    Args:
        mesh_path (Path): Source mesh path.
        collision_mode (str): Collision mode used to build the generated MJCF.
        object_scale (tuple[float, float, float]): Mesh scale used for generation.

    Returns:
        tuple[Path, np.ndarray]: Generated XML path and local geom position.
    """
    scale_tag = _format_scale_tag(object_scale)
    xml_path = (
        mesh_path.parent
        / "_generated_mjcf"
        / collision_mode
        / scale_tag
        / f"{mesh_path.stem}.xml"
    )
    root = ET.parse(xml_path).getroot()

    geom_elem = root.find(".//geom[@name='object_visual']")
    if geom_elem is None:
        raise ValueError(f"No visual mesh geom found in generated XML: {xml_path}")

    geom_pos_text = geom_elem.attrib.get("pos")
    if geom_pos_text is None:
        raise ValueError(f"Mesh geom is missing pos in generated XML: {xml_path}")

    geom_pos = np.fromstring(geom_pos_text, sep=" ")
    if geom_pos.shape != (3,):
        raise ValueError(f"Invalid geom pos in generated XML: {xml_path}")

    return xml_path, geom_pos


def get_object_geom_ids(env):
    """Collect MuJoCo geom IDs that appear to belong to the debug object.

    Args:
        env (TargetEnvironment): Active robosuite environment.

    Returns:
        list[tuple[int, str]]: Matching `(geom_id, geom_name)` pairs.
    """
    geom_ids = []
    for geom_id in range(env.sim.model.ngeom):
        geom_name = env.sim.model.geom_id2name(geom_id)
        if geom_name is None:
            continue
        if "obj" in geom_name or "object" in geom_name:
            geom_ids.append((geom_id, geom_name))
    return geom_ids


def print_pose_debug(
    env,
    mesh_path,
    object_qpos,
    object_scale,
    collision_mode=DEFAULT_COLLISION_MODE,
):
    """Print mesh alignment diagnostics for the currently loaded object.

    Args:
        env (TargetEnvironment): Active robosuite environment.
        mesh_path (str | Path): Mesh being visualized.
        object_qpos (sequence[float]): Object joint pose used for the scene.
        object_scale (tuple[float, float, float]): Per-axis mesh scale.
    """
    body_pos = env.sim.data.body_xpos[env.object_body_id].copy()
    body_rot = env.sim.data.body_xmat[env.object_body_id].reshape(3, 3).copy()

    xml_path, local_geom_pos = load_generated_geom_pos(
        Path(mesh_path),
        collision_mode=collision_mode,
        object_scale=object_scale,
    )
    expected_geom_world = body_pos + body_rot @ local_geom_pos

    min_corner, max_corner = get_mesh_bounds(mesh_path)
    scale = np.asarray(object_scale, dtype=float)
    scaled_extent = (max_corner - min_corner) * scale
    expected_bottom_world = body_pos.copy()
    expected_top_world = body_pos + body_rot @ np.array([0.0, 0.0, scaled_extent[2]])

    print("=" * 80)
    print("Pose debug")
    print("=" * 80)
    print("mesh path:", mesh_path)
    print("generated xml:", xml_path)
    print("object_qpos [x, y, z, qw, qx, qy, qz]:", object_qpos)
    print("body world pos:", body_pos)
    print("body world rot:")
    print(body_rot)
    print("local geom pos from XML:", local_geom_pos)
    print("expected geom world pos:", expected_geom_world)
    print("scaled extent [x, y, z]:", scaled_extent)
    print("expected bottom-center world pos:", expected_bottom_world)
    print("expected top-center world pos:", expected_top_world)

    geom_ids = get_object_geom_ids(env)
    if not geom_ids:
        print("No object geoms found by name filter.")
    else:
        for geom_id, geom_name in geom_ids:
            actual_geom_world = env.sim.data.geom_xpos[geom_id].copy()
            print(f"geom {geom_id} ({geom_name}) world pos:", actual_geom_world)
            if "visual" in geom_name:
                error = actual_geom_world - expected_geom_world
                print(f"geom {geom_id} ({geom_name}) expected error:", error)
    print("=" * 80)


def hold_scene(env, obs, steps=400):
    """Advance the simulation while keeping the current scene visible.

    Args:
        env (TargetEnvironment): Active robosuite environment.
        obs (OrderedDict): Latest environment observation.
        steps (int): Number of simulation steps to render.

    Returns:
        OrderedDict: Final observation after holding the scene.
    """
    for _ in range(steps):
        action = np.zeros(7)
        obs, _, _, _ = env.step(action)
        env.render()
    return obs


def main():
    """Iterate through test orientations to inspect generated mesh alignment."""
    parser = GraspFactoryParser(gripper="robotiq_2f85")
    sample = parser[0]

    print("Testing mesh:", sample.mesh_path)

    for name, quat in TEST_QUATS:
        print(f"\nShowing orientation: {name}  quat[w, x, y, z]={quat}")
        object_qpos = [0.0, 0.0, 1.2, *quat]
        object_scale = DEFAULT_OBJECT_SCALE

        env, obs = arm_controller.initialize_environment(
            dataset_mesh_path=str(sample.mesh_path),
            object_qpos=object_qpos,
            object_scale=object_scale,
            object_collision_approximation=DEFAULT_COLLISION_MODE,
        )

        print_pose_debug(
            env=env,
            mesh_path=sample.mesh_path,
            object_qpos=object_qpos,
            object_scale=object_scale,
            collision_mode=DEFAULT_COLLISION_MODE,
        )

        hold_scene(env, obs, steps=250)

        try:
            env.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
