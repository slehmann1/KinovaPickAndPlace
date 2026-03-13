import numpy as np

from src.controllers import arm_controller
from src.utils.graspfactory_parser import GraspFactoryParser
from src.utils.transforms import (
    transform_to_pose,
    world_grasp_from_object,
    offset_transform_along_local_axis,
    body_pose_to_transform,
    quat_wxyz_to_xyzw,
)
from src.utils.dataset_xml_builder import get_mesh_bounds


PREGRASP_OFFSET_M = 0.10
LIFT_DISTANCE_M = 0.10

# If the gripper appears rotated wrong relative to the dataset grasp,
# tune this correction transform. Start with identity.
GRASP_FRAME_CORRECTION = np.eye(4)


def get_sim_object_transform(env):
    """Return the current simulated object's world pose as a transform.

    Returns:
        np.ndarray: Homogeneous transform `world_T_object`.
    """
    body_pos = env.sim.data.body_xpos[env.object_body_id]
    body_xmat = env.sim.data.body_xmat[env.object_body_id]
    return body_pose_to_transform(body_pos, body_xmat)


def load_dataset_grasp(parser, object_idx=0, grasp_idx=0):
    """Load a successful dataset grasp for the requested object entry.

    Args:
        parser (GraspFactoryParser): Dataset parser instance.
        object_idx (int): Index of the object sample to load.
        grasp_idx (int): Index within the successful grasps for that sample.

    Returns:
        tuple: `(sample, object_T_grasp, grasp_width)`.
    """
    sample = parser[object_idx]

    if len(sample.success_indices) == 0:
        raise ValueError(f"No successful grasps found for object {sample.object_id}")

    if grasp_idx < 0 or grasp_idx >= len(sample.successful_grasps):
        raise IndexError(
            f"grasp_idx {grasp_idx} out of range for {len(sample.successful_grasps)} successful grasps"
        )

    object_T_grasp = sample.successful_grasps[grasp_idx]
    grasp_width = sample.successful_widths[grasp_idx]

    return sample, object_T_grasp, grasp_width


def print_grasp_summary(sample, object_T_grasp, world_T_object, world_T_grasp, grasp_width):
    """Print debug information for a selected grasp candidate.

    Args:
        sample (GraspFactoryObj): Dataset sample containing mesh and grasp metadata.
        object_T_grasp (np.ndarray): Grasp pose expressed in the object frame.
        world_T_object (np.ndarray): Object pose expressed in the world frame.
        world_T_grasp (np.ndarray): Grasp pose expressed in the world frame.
        grasp_width (float): Finger width stored with the selected grasp.
    """
    grasp_pos_obj, grasp_quat_obj = transform_to_pose(object_T_grasp)
    grasp_pos_world, grasp_quat_world = transform_to_pose(world_T_grasp)

    print("=" * 80)
    print("Grasp visualization summary")
    print("=" * 80)
    print(f"Object ID: {sample.object_id}")
    print(f"Mesh path: {sample.mesh_path}")
    print(f"Successful grasps available: {len(sample.successful_grasps)}")
    print(f"Selected grasp width: {grasp_width}")
    print()
    print("Object-frame grasp pose:")
    print("  position:", grasp_pos_obj)
    print("  quaternion [w, x, y, z]:", grasp_quat_obj)
    print()
    print("World-frame object transform:")
    print(world_T_object)
    print()
    print("World-frame grasp pose:")
    print("  position:", grasp_pos_world)
    print("  quaternion [w, x, y, z]:", grasp_quat_world)
    print("=" * 80)


def visualize_grasp_sequence(env, obs, world_T_grasp):
    """Execute a simple pre-grasp, grasp, and lift visualization sequence.

    Args:
        env (TargetEnvironment): Active robosuite environment.
        obs (OrderedDict): Latest environment observation.
        world_T_grasp (np.ndarray): Target grasp pose in the world frame.

    Returns:
        OrderedDict: Final observation after the visualization sequence.
    """
    print("Moving to safe start pose...")
    obs = arm_controller.move_ee_to_position(
        step_count=100,
        goal_pos=[0.1, 0.1, 1.2],
        gripper_state=-1,
        env=env,
        obs=obs,
    )

    world_T_pregrasp = offset_transform_along_local_axis(
        world_T_grasp,
        axis_index=2,
        distance=PREGRASP_OFFSET_M,
        sign=-1.0,
    )

    pregrasp_pos, pregrasp_quat = transform_to_pose(world_T_pregrasp)
    grasp_pos, grasp_quat = transform_to_pose(world_T_grasp)

    print("Moving to pre-grasp pose...")
    obs = arm_controller.move_ee_to_pose(
        step_count=150,
        goal_pos=pregrasp_pos,
        goal_quat=quat_wxyz_to_xyzw(pregrasp_quat),
        gripper_state=-1,
        env=env,
        obs=obs,
    )

    print("Moving to grasp pose...")
    obs = arm_controller.move_ee_to_pose(
        step_count=150,
        goal_pos=grasp_pos,
        goal_quat=quat_wxyz_to_xyzw(grasp_quat),
        gripper_state=-1,
        env=env,
        obs=obs,
    )

    print("Closing gripper...")
    obs = arm_controller.set_gripper(
        step_count=40,
        gripper_state=1,
        env=env,
        obs=obs,
    )

    lift_pos = grasp_pos.copy()
    lift_pos[2] += LIFT_DISTANCE_M

    print("Lifting...")
    obs = arm_controller.move_ee_to_pose(
        step_count=120,
        goal_pos=lift_pos,
        goal_quat=quat_wxyz_to_xyzw(grasp_quat),
        gripper_state=1,
        env=env,
        obs=obs,
    )

    return obs


def compute_table_resting_qpos(
    mesh_path,
    object_xy=(-0.2, 0.0),
    object_quat=(1, 0, 0, 0),
    object_scale=(1.0, 1.0, 1.0),
    table_height=0.8,
    table_thickness=0.05,
    clearance=0.001,
):
    """Compute a MuJoCo joint pose that places a mesh just above the table.

    Args:
        mesh_path (str | Path): Mesh file used to compute local bounds.
        object_xy (tuple[float, float]): Desired object x-y position on the table.
        object_quat (tuple[float, float, float, float]): MuJoCo quaternion `[w, x, y, z]`.
        object_scale (tuple[float, float, float]): Per-axis mesh scale.
        table_height (float): Table body offset in world coordinates.
        table_thickness (float): Table thickness used by the arena definition.
        clearance (float): Small positive gap to avoid initial interpenetration.

    Returns:
        list[float]: Object joint pose `[x, y, z, qw, qx, qy, qz]`.
    """
    # TODO: Re-check this clearance against more irregular meshes and non-identity rotations.
    min_corner, _ = get_mesh_bounds(mesh_path)

    scale = np.asarray(object_scale, dtype=float)
    scaled_min_corner = min_corner * scale

    table_top_z = table_height + table_thickness / 2.0
    object_origin_z = table_top_z - scaled_min_corner[2] + clearance

    return [
        object_xy[0],
        object_xy[1],
        object_origin_z,
        object_quat[0],
        object_quat[1],
        object_quat[2],
        object_quat[3],
    ]


def main():
    """Load a dataset object, derive a grasp pose, and hold the scene for inspection."""
    parser = GraspFactoryParser(gripper="robotiq_2f85")

    object_idx = 0
    grasp_idx = 0

    sample, object_T_grasp_raw, grasp_width = load_dataset_grasp(
        parser,
        object_idx=object_idx,
        grasp_idx=grasp_idx,
    )

    object_scale = (0.001, 0.001, 0.001)
    # Keep the mesh in a raised debug pose until the table-resting alignment is verified.
    # TODO: Switch back to compute_table_resting_qpos once the mesh placement path is validated.
    object_qpos = [0.2, -0.1, 1.0, 1.0, 0.0, 0.0, 0.0]

    env, obs = arm_controller.initialize_environment(
        dataset_mesh_path=str(sample.mesh_path),
        object_qpos=object_qpos,
        object_scale=object_scale,
    )

    print("Moving arm to raised start pose...")
    obs = arm_controller.move_ee_to_position(
        step_count=150,
        goal_pos=[0.15, 0.0, 1.25],
        gripper_state=-1,
        env=env,
        obs=obs,
    )

    world_T_object = get_sim_object_transform(env)
    object_T_grasp = object_T_grasp_raw @ GRASP_FRAME_CORRECTION
    world_T_grasp = world_grasp_from_object(world_T_object, object_T_grasp)

    print_grasp_summary(
        sample=sample,
        object_T_grasp=object_T_grasp,
        world_T_object=world_T_object,
        world_T_grasp=world_T_grasp,
        grasp_width=grasp_width,
    )

    # TODO: Re-enable the grasp motion sequence after the placement debug path is finished.
    # obs = visualize_grasp_sequence(env, obs, world_T_grasp)

    # TODO: testing
    print("Object loaded. Holding still for inspection...")
    try:
        for _ in range(300):
            action = np.zeros(7)
            obs, _, _, _ = env.step(action)
            env.render()
    finally:
        env.close()

    print("Visualization complete.")


if __name__ == "__main__":
    main()
