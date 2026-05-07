
# **Kinova Pick And Place**

#### Overview

This is a robotic simulation and motion planning pipeline designed for a pick-and-place task using a Kinova arm. It integrates imitation learning for object grasping with obstacle-aware motion planning to navigate and manipulate objects in obstacle laden environments.

---

#### Example Showcase
Full video of grasping an object and avoiding obstacles to place it in the target location.
 [![Demonstration of a pick and place task in an obstacle laden environment](https://github.com/slehmann1/KinovaPickAndPlace/blob/main/demo/ImitationLearningAndObstacleAvoidance.gif?raw=true)](#) 

---

#### Methodology

The project addresses the pick-and-place problem by breaking it into distinct components and bringing them together in an integrated pipeline:

* **Grasping:** A Robitq 2F-85 gripper is used, paired with Imitation learning to be able to grasp a diverse variety of objects. Grasping policies are trained on a large dataset of grasps using the [GraspFactory](https://graspfactory.github.io/) dataset.
The below image demonstrates grasping for a range of objects.
 [![Grasping a range of objects](https://github.com/slehmann1/KinovaPickAndPlace/blob/main/demo/GraspDemonstration.png?raw=true)](#) 

* **Obstacle navigation:** An RRT (Rapidly exploring Random Trees) based obstacle avoidance routine charts a collision-free path for the robotic arm to move from the grasp location to the target drop zone. This approach is shown to be successful for environments with a wide array of objects. 
* **Control:** The robot control system is split into two distinct levels: low-level and high-level controllers. The high-level controller consists of the imitation learning model for grasping and the path planning algorithm that outlines how the end effector should travel to the goal. The low-level controller is responsible for implementing these commands and enabling the joints of the robot. This system functions through a PID controller implemented at the end effector; inverse kinematics are used to translate end effector positions to joint angles.

---
**Dependencies:**

 1. Mujoco
 2. GraspFactory
 3. Numpy
 4. Scipy
 5. Robosuite

