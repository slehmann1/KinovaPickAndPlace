import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler

from src.utils.dataset_mesh_object import DatasetMeshObject


class TargetEnvironment(ManipulationEnv):
    """Environment containing a table, a target zone, and a single object.

    Args:
        ManipulationEnv (RobotEnv): Base robosuite manipulation environment.
    """

    TABLE_HEIGHT = 0.8
    TARGET_ZONE_SIZE = [0.15, 0.15, 0.05]
    PRIMITIVE_BOX_HALF_SIZE = [0.02, 0.02, 0.02]
    VALID_COLLISION_APPROXIMATIONS = ("mesh", "convex_hull", "box")
    DEFAULT_OBJECT_DROP_HEIGHT = 0.02

    def __init__(
        self,
        dataset_mesh_path=None,
        object_scale=(1.0, 1.0, 1.0),
        object_qpos=None,
        object_xy=(-0.2, 0.0),
        object_rotation=0.0,
        object_collision_approximation="mesh",
        object_drop_height=DEFAULT_OBJECT_DROP_HEIGHT,
        **kwargs,
    ):
        """Initialize the environment configuration before robosuite model loading.

        Args:
            dataset_mesh_path (str | Path | None): Optional mesh path for a dataset object.
            object_scale (tuple[float, float, float]): Per-axis mesh scale in MuJoCo units.
            object_qpos (sequence[float] | None): Optional initial object joint pose.
            object_xy (tuple[float, float]): Default object spawn location on the table.
            object_rotation (float | None): Optional z-axis rotation for placement sampling.
            object_collision_approximation (str): Collision geometry type for dataset
                meshes: `mesh`, `convex_hull`, or `box`.
            object_drop_height (float): Height above the table used for sampled spawns
                so the object can settle onto the support surface.
            **kwargs: Additional robosuite environment arguments.
        """
        self.dataset_mesh_path = dataset_mesh_path
        self.object_scale = object_scale
        self.object_qpos = object_qpos
        self.object_xy = tuple(object_xy)
        self.object_rotation = object_rotation
        self.object_collision_approximation = object_collision_approximation
        self.object_drop_height = float(object_drop_height)
        if self.object_collision_approximation not in self.VALID_COLLISION_APPROXIMATIONS:
            raise ValueError(
                "object_collision_approximation must be one of "
                f"{self.VALID_COLLISION_APPROXIMATIONS}, got "
                f"{self.object_collision_approximation!r}"
            )
        if self.object_drop_height < 0.0:
            raise ValueError(
                f"object_drop_height must be non-negative, got {self.object_drop_height}"
            )
        super().__init__(**kwargs)

    def _load_model(self):
        """Build the MuJoCo scene with the robot, table, target, and object."""
        super()._load_model()

        # Create environment
        self.mujoco_arena = TableArena(
            table_full_size=(0.8, 0.8, 0.05), table_offset=(0, 0, self.TABLE_HEIGHT)
        )

        self.target_zone = BoxObject(
            name="target_zone",
            size=self.TARGET_ZONE_SIZE,
            rgba=[0, 1, 0, 1],
            joints=None,
            obj_type="all",  # Ensures both visual and collision meshes are created
            duplicate_collision_geoms=True,
        )

        # Reposition for environment offset
        self.robots[0].robot_model.set_base_xpos([-0.5, 0, 0])

        # Keep a named reference to the manipulated object so the placement sampler,
        # state queries, and future extensions all use the same instance.
        self.manipulated_object = self._get_object()
        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=self.manipulated_object,
            x_range=[self.object_xy[0], self.object_xy[0]],
            y_range=[self.object_xy[1], self.object_xy[1]],
            rotation=self.object_rotation,
            rotation_axis="z",
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=np.array([0.0, 0.0, self.TABLE_HEIGHT]),
            z_offset=self.object_drop_height,
        )

        robot_models = [robot.robot_model for robot in self.robots]
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=robot_models,
            mujoco_objects=[self.target_zone, self.manipulated_object],
        )

    def _get_object(self):
        """Create the object that should be manipulated in the scene.

        Returns:
            MujocoObject: Dataset-backed mesh object when available, otherwise a box.
        """
        if self.dataset_mesh_path is not None:
            # TODO: Sample dataset objects and scales through a reusable scene config instead.
            return DatasetMeshObject(
                name="obj",
                mesh_path=self.dataset_mesh_path,
                scale=self.object_scale,
                collision_approximation=self.object_collision_approximation,
            )

        # Use a small box as the default object because it gives a cleaner baseline
        # for debugging scripted pick-and-place than arbitrary mesh geometry.
        return BoxObject(
            name="obj",
            size=self.PRIMITIVE_BOX_HALF_SIZE,
            rgba=[1, 0, 0, 1],
            joints="default",  # Environment interaction
            obj_type="all",  # "all" creates both a visual mesh and a collision geom
            duplicate_collision_geoms=True,
        )

    def _setup_references(self):
        """Cache MuJoCo IDs used during resets and state queries."""
        super()._setup_references()
        self.table_top_site_id = self.sim.model.site_name2id("table_top")
        self.target_zone_body_id = self.sim.model.body_name2id("target_zone_main")
        self.object_body_id = self.sim.model.body_name2id("obj_main")
        self.object_joint_name = "obj_joint0"

    def _reset_internal(self):
        """Reset dynamic state and place the target zone and object for a new episode."""
        super()._reset_internal()

        # Place target zone on table
        self.sim.model.body_pos[self.target_zone_body_id] = np.array(
            [0.2, 0.0, self.TABLE_HEIGHT + 0.02]
        )

        # Prefer robosuite's placement sampler so the environment, not the playground
        # script, owns stable object placement on the table.
        if self.object_qpos is None:
            object_placements = self.placement_initializer.sample()
            object_pos, object_quat, _ = next(iter(object_placements.values()))
            qpos = np.concatenate([np.array(object_pos), np.array(object_quat)])
        else:
            qpos = np.array(self.object_qpos, dtype=float)

        self.sim.data.set_joint_qpos(self.object_joint_name, qpos)
        self.sim.forward()

    def get_object_pose(self):
        """Return the manipulated object's world position and rotation matrix.

        Returns:
            tuple[np.ndarray, np.ndarray]: Object position and 3x3 rotation matrix.
        """
        obj_pos = self.sim.data.body_xpos[self.object_body_id].copy()
        obj_rot = self.sim.data.body_xmat[self.object_body_id].reshape(3, 3).copy()
        return obj_pos, obj_rot

    def get_table_surface_position(self, xy=None):
        """Return the table-top position from the live simulation state.

        Args:
            xy (sequence[float] | None): Optional x-y position to project onto the table surface.

        Returns:
            np.ndarray: World position on the table surface.
        """
        table_surface_pos = self.sim.data.site_xpos[self.table_top_site_id].copy()
        if xy is not None:
            table_surface_pos[:2] = np.asarray(xy, dtype=float)
        return table_surface_pos

    def reward(self, action=None):
        """Return the task reward required by robosuite.

        Args:
            action (np.ndarray | None): Unused action vector from the simulator.

        Returns:
            float: Placeholder reward value.
        """
        # TODO: add reward for training
        return 0.0

    def _check_success(self):
        """Check whether the object is resting inside the target zone.

        Returns:
            bool: True when the object is inside the zone and approximately stationary.
        """

        # Tolerances used to define success. Avoid geometry-specific z checks here so
        # the same logic still works for primitive boxes and dataset-backed meshes.
        vel_tol = 0.05  # m/s
        z_tol = 0.1  # m

        obj_pos = self.sim.data.body_xpos[self.object_body_id]
        target_pos = self.sim.data.body_xpos[self.target_zone_body_id]
        obj_vel = self.sim.data.get_body_xvelp("obj_main")  # Linear velocity

        if np.linalg.norm(obj_vel) >= vel_tol:
            print("Object is still moving. Velocity:", obj_vel)
            return False  # Still moving

        # Check that the object has been transported onto the same support surface
        # height as the target zone without assuming any particular object geometry.
        z_diff = abs(obj_pos[2] - target_pos[2])

        if z_diff >= z_tol:
            print("Object is not close enough in z. Difference:", z_diff)
            return False  # Not close enough in z

        x_dist = abs(obj_pos[0] - target_pos[0])
        y_dist = abs(obj_pos[1] - target_pos[1])

        if x_dist >= self.TARGET_ZONE_SIZE[0] / 2 or y_dist >= self.TARGET_ZONE_SIZE[1] / 2:
            print("Object is not close enough in x or y. Distances:", x_dist, y_dist)
            return False  # Not close enough in x or y

        print("Success!")
        return True
