import numpy as np
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask


class TargetEnvironment(ManipulationEnv):
    """An Environment containing a green target area for an object to be placed in.

    Args:
        ManipulationEnv (RobotEnv): Robosuite Manipulation Environment
    """

    TABLE_HEIGHT = 0.8
    TARGET_ZONE_SIZE = [0.15, 0.15, 0.05]
    OBJECT_HEIGHT = 0.05

    def _load_model(self):
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
        self.model = ManipulationTask(
            mujoco_arena=self.mujoco_arena,
            mujoco_robots=robot_models,
            mujoco_objects=[self.target_zone, self._get_object()],
        )

    def _get_object(self):
        """To be replaced with a function that loads a random object. For now it just returns a cylinder"""
        return CylinderObject(
            name="obj",
            size=[0.02, self.OBJECT_HEIGHT],
            rgba=[1, 0, 0, 1],
            joints="default",  # Environment interaction
            obj_type="all",  # "all" creates both a visual mesh and a collision geom
        )

    def _setup_references(self):
        super()._setup_references()
        self.target_zone_body_id = self.sim.model.body_name2id("target_zone_main")
        self.object_body_id = self.sim.model.body_name2id("obj_main")
        self.object_joint_name = "obj_joint0"

    def _reset_internal(self):
        super()._reset_internal()
        # Place target zone on table
        self.sim.model.body_pos[self.target_zone_body_id] = np.array(
            [0.2, 0.0, self.TABLE_HEIGHT + 0.02]
        )
        # Place object on table
        self.sim.data.set_joint_qpos(
            self.object_joint_name,
            np.array([-0.2, 0.0, self.TABLE_HEIGHT + 0.05, 1, 0, 0, 0]),
        )
        self.sim.forward()

    # Abstract methods that need to be implemented
    def reward(self, action=None):
        """
        Required by robosuite. Returning 0 for now.
        """
        return 0.0

    def _check_success(self):
        """
        Success if the object is on top of the target zone and stationary.
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

        print("Success and Tested!")
        return True
