import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask

from src.utils.dataset_mesh_object import DatasetMeshObject


class TargetEnvironment(ManipulationEnv):
    """Environment containing a table, a target zone, and a single object.

    Args:
        ManipulationEnv (RobotEnv): Base robosuite manipulation environment.
    """

    TABLE_HEIGHT = 0.8
    TARGET_ZONE_SIZE = [0.15, 0.15, 0.05]
    OBJECT_HEIGHT = 0.05

    def __init__(
        self,
        dataset_mesh_path=None,
        object_scale=(1.0, 1.0, 1.0),
        object_qpos=None,
        obstacle_configs=None,
        **kwargs,
    ):
        """Initialize the environment configuration before robosuite model loading.

        Args:
            dataset_mesh_path (str | Path | None): Optional mesh path for a dataset object.
            object_scale (tuple[float, float, float]): Per-axis mesh scale in MuJoCo units.
            object_qpos (sequence[float] | None): Optional initial object joint pose.
            obstacle_configs (list[dict] | None): Optional fixed box obstacles.
            **kwargs: Additional robosuite environment arguments.
        """
        self.dataset_mesh_path = dataset_mesh_path
        self.object_scale = object_scale
        self.object_qpos = object_qpos
        self.obstacle_configs = obstacle_configs or []
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

        robot_models = [robot.robot_model for robot in self.robots]
        mujoco_objects = [self.target_zone, self._get_object()]
        mujoco_objects.extend(self._build_obstacles())
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=robot_models,
            mujoco_objects=mujoco_objects,
        )

    def _get_object(self):
        """Create the object that should be manipulated in the scene.

        Returns:
            MujocoObject: Dataset-backed mesh object when available, otherwise a cylinder.
        """
        if self.dataset_mesh_path is not None:
            # TODO: Sample dataset objects and scales through a reusable scene config instead.
            return DatasetMeshObject(
                name="obj",
                mesh_path=self.dataset_mesh_path,
                scale=self.object_scale,
            )

        return CylinderObject(
            name="obj",
            size=[0.02, self.OBJECT_HEIGHT],
            rgba=[1, 0, 0, 1],
            joints="default",  # Environment interaction
            obj_type="all",  # "all" creates both a visual mesh and a collision geom
        )

    def _build_obstacles(self):
        """Create any fixed box obstacles requested for the scene."""
        obstacles = []
        for index, config in enumerate(self.obstacle_configs):
            size = list(config.get("size", [0.04, 0.04, 0.08]))
            rgba = list(config.get("rgba", [0.2, 0.2, 0.2, 1.0]))
            obstacles.append(
                BoxObject(
                    name=f"obstacle_{index}",
                    size=size,
                    rgba=rgba,
                    joints=None,
                    obj_type="all",
                    duplicate_collision_geoms=True,
                )
            )
        return obstacles

    def _setup_references(self):
        """Cache MuJoCo IDs used during resets and state queries."""
        super()._setup_references()
        self.target_zone_body_id = self.sim.model.body_name2id("target_zone_main")
        self.object_body_id = self.sim.model.body_name2id("obj_main")
        self.object_joint_name = "obj_joint0"
        self.obstacle_body_ids = []
        for index, _ in enumerate(self.obstacle_configs):
            body_id = self.sim.model.body_name2id(f"obstacle_{index}_main")
            self.obstacle_body_ids.append(body_id)

    def _reset_internal(self):
        """Reset dynamic state and place the target zone and object for a new episode."""
        super()._reset_internal()
        # Place target zone on table
        self.sim.model.body_pos[self.target_zone_body_id] = np.array(
            [0.2, 0.0, self.TABLE_HEIGHT + 0.02]
        )

        # Place object on table
        if self.object_qpos is None:
            # TODO: Randomize spawn pose once the grasp and placement pipeline is stable.
            qpos = np.array([-0.2, 0.0, self.TABLE_HEIGHT + 0.05, 1, 0, 0, 0])
        else:
            qpos = np.array(self.object_qpos, dtype=float)

        self.sim.data.set_joint_qpos(self.object_joint_name, qpos)

        for body_id, config in zip(self.obstacle_body_ids, self.obstacle_configs):
            self.sim.model.body_pos[body_id] = np.array(config["pos"], dtype=float)
            obstacle_quat = config.get("quat")
            if obstacle_quat is not None:
                self.sim.model.body_quat[body_id] = np.array(obstacle_quat, dtype=float)

        self.sim.forward()

    def get_object_pose(self):
        """Return the manipulated object's world position and rotation matrix.

        Returns:
            tuple[np.ndarray, np.ndarray]: Object position and 3x3 rotation matrix.
        """
        obj_pos = self.sim.data.body_xpos[self.object_body_id].copy()
        obj_rot = self.sim.data.body_xmat[self.object_body_id].reshape(3, 3).copy()
        return obj_pos, obj_rot

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

        # Tolerances used to define success
        z_tol = 0.1  # m
        vel_tol = 0.05  # m/s

        obj_pos = self.sim.data.body_xpos[self.object_body_id]
        target_pos = self.sim.data.body_xpos[self.target_zone_body_id]
        obj_vel = self.sim.data.get_body_xvelp("obj_main")  # Linear velocity

        if np.linalg.norm(obj_vel) >= vel_tol:
            print("Object is still moving. Velocity:", obj_vel)
            return False  # Still moving

        z_diff = abs(
            obj_pos[2]
            + self.OBJECT_HEIGHT / 2
            - (target_pos[2] + self.TARGET_ZONE_SIZE[2] / 2)
        )

        if z_diff >= z_tol:
            print("Object is not close enough in z. Difference:", z_diff)
            return False  # Not close enough in z

        x_dist = abs(obj_pos[0] - target_pos[0])
        y_dist = abs(obj_pos[1] - target_pos[1])

        if (
            x_dist >= self.TARGET_ZONE_SIZE[0] / 2
            or y_dist >= self.TARGET_ZONE_SIZE[1] / 2
        ):
            print("Object is not close enough in x or y. Distances:", x_dist, y_dist)
            return False  # Not close enough in x or y

        print("Success!")
        return True
