import numpy as np
import robosuite as suite
from scipy.interpolate import CubicSpline

from environments import TargetEnvironment

# PID control gains
K_P = 4.0
K_I = 0.1
K_D = 0.2

# Initialization
env = TargetEnvironment(
    robots="Panda",
    has_renderer=True,
    has_offscreen_renderer=False,
    use_camera_obs=False,
    control_freq=20,
    gripper_types="Robotiq85Gripper",
)

obs = env.reset()
env.viewer.set_camera(camera_id=-1)

print("Start")

target_bin_pos = obs.get("target_zone_pos", np.array([0.0, 0.2, 0.82]))

integral_error = np.zeros(3)
prev_error = np.zeros(3)

# Define custom points through which splines are used to create a trajectory for the gripper to follow. Each point is (x, y, z).
trajectory_points = np.array(
    [
        [0.0, 0.0, 1.0],
        [0.1, 0.0, 1.0],
        [0.1, 0.1, 1.0],
        [0.0, 0.1, 1.0],
        [-0.1, 0.1, 1.0],
        [-0.1, 0.0, 1.0],
        [0.0, 0.0, 1.0],
    ]
)

# Create cubic splines for x, y, z
t_points = np.arange(len(trajectory_points))
num_steps = 1000
t_interp = np.linspace(0, len(trajectory_points) - 1, num_steps)
cs_x = CubicSpline(t_points, trajectory_points[:, 0])
cs_y = CubicSpline(t_points, trajectory_points[:, 1])
cs_z = CubicSpline(t_points, trajectory_points[:, 2])

for i in range(num_steps):
    goal_pos = np.array([cs_x(t_interp[i]), cs_y(t_interp[i]), cs_z(t_interp[i])])
    gripper_pos = obs["robot0_eef_pos"]

    action = np.zeros(7)
    error = goal_pos - gripper_pos

    # PID calculations, assuming time step is 1
    integral_error += error
    derivative_error = error - prev_error
    prev_error = error

    action[:3] = K_P * error + K_I * integral_error + K_D * derivative_error

    action[-1] = 1  # Open

    obs, reward, done, info = env.step(action)
    env.render()
