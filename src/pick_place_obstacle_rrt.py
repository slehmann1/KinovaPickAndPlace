from __future__ import annotations

import numpy as np

from src.controllers import arm_controller
from src.obstacle_navigation_rrt import (
    planRrtPath,
    safeZ,
    xyPathToCartesian,
)

# Grasp target
graspZOffset = 0.03
postPlaceLift = 0.12
startPos = np.array([0.10, -0.22, safeZ], dtype=float)
planningBounds = np.array([[-0.20, 0.30], [-0.30, 0.30]], dtype=float)
placeZOffset = 0.08
placeClearanceRadius = 0.08
# Default obstacle layout for pick and place
defaultPickPlaceObstacles = [
    {
        "pos": [0.02, -0.08, 0.92],
        "size": [0.05, 0.05, 0.10],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "rgba": [0.85, 0.35, 0.2, 1.0],
    },
    {
        "pos": [0.10, 0.14, 0.92],
        "size": [0.05, 0.05, 0.10],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "rgba": [0.2, 0.45, 0.85, 1.0],
    },
]

def getObjectPosition(env) -> np.ndarray:
    objectPos, _ = env.get_object_pose()
    return np.asarray(objectPos, dtype=float)

def getTargetPosition(env) -> np.ndarray:
    return env.sim.data.body_xpos[env.target_zone_body_id].copy()

# Check if one point is clear of all obstacles
def pointClearOfObstacles(pointXy: np.ndarray, obstacleConfigs: list[dict], clearanceRadius: float) -> bool:
    for obstacle in obstacleConfigs:
        obstacleXy = np.asarray(obstacle["pos"][:2], dtype=float)
        paddedHalfExtents = np.asarray(obstacle["size"][:2], dtype=float) + clearanceRadius
        lower = obstacleXy - paddedHalfExtents
        upper = obstacleXy + paddedHalfExtents
        if np.all(pointXy >= lower) and np.all(pointXy <= upper):
            return False
    return True

# Pick closest point to target
def chooseSafePlaceXy(preferredXy: np.ndarray, obstacleConfigs: list[dict]) -> np.ndarray:
    if pointClearOfObstacles(preferredXy, obstacleConfigs, placeClearanceRadius):
        return preferredXy

    xOffsets = np.linspace(-0.12, 0.12, 7)
    yOffsets = np.linspace(-0.12, 0.12, 7)
    candidates = []
    for xOffset in xOffsets:
        for yOffset in yOffsets:
            candidate = preferredXy + np.array([xOffset, yOffset], dtype=float)
            if not (
                planningBounds[0, 0] <= candidate[0] <= planningBounds[0, 1]
                and planningBounds[1, 0] <= candidate[1] <= planningBounds[1, 1]
            ):
                continue
            if pointClearOfObstacles(candidate, obstacleConfigs, placeClearanceRadius):
                candidates.append(candidate)

    if not candidates:
        raise RuntimeError(
            "Unable to find obstacle-free placement point"
        )

    return min(candidates, key=lambda candidate: np.linalg.norm(candidate - preferredXy))

# Move target zone to new xy position
def setTargetZoneXy(env, targetXy: np.ndarray):
    env.sim.model.body_pos[env.target_zone_body_id][:2] = np.asarray(targetXy, dtype=float)
    env.sim.forward()

# Plan an obstacle avoiding cartesian path at safe z
def planCartesianRrt(startPos: np.ndarray, goalPos: np.ndarray, obstacleConfigs: list[dict]) -> np.ndarray:
    pathXy = planRrtPath(startXy=np.asarray(startPos[:2], dtype=float), goalXy=np.asarray(goalPos[:2], dtype=float), obstacles=obstacleConfigs, bounds=planningBounds)
    return xyPathToCartesian(pathXy, safeZ)

# Follow an RRT planned path
def moveAlongRrt(env, obs, startPos: np.ndarray, goalPos: np.ndarray, gripperState: int, obstacleConfigs: list[dict]):
    trajectory = planCartesianRrt(startPos, goalPos, obstacleConfigs)
    print("Planned transport waypoints:")
    for waypoint in trajectory:
        print(waypoint.tolist())

    obs = arm_controller.move_ee_along_trajectory(trajectory_points=trajectory, step_count=max(240, len(trajectory) * 80), gripper_state=gripperState, env=env, obs=obs)
    return obs

# Refine final place motion above and into the target
def alignForPlace(env, obs, targetPos: np.ndarray):
    preplacePos = np.array([targetPos[0], targetPos[1], safeZ], dtype=float)
    placePos = np.array([targetPos[0], targetPos[1], targetPos[2] + placeZOffset], dtype=float)

    print("Refining alignment above target zone")
    obs = arm_controller.move_ee_to_position(step_count=140, goal_pos=preplacePos, gripper_state=1, env=env, obs=obs)

    print("Lowering with final place alignment")
    obs = arm_controller.move_ee_to_position(step_count=180, goal_pos=placePos, gripper_state=1, env=env, obs=obs)
    return obs

# Measure final task success and placement error
def evaluateTestResult(env) -> dict:
    objectPos = getObjectPosition(env)
    targetPos = getTargetPosition(env)
    xyError = float(np.linalg.norm(objectPos[:2] - targetPos[:2]))
    zError = float(abs(objectPos[2] - targetPos[2]))
    success = bool(env._check_success())
    return {
        "success": success,
        "objectPosition": objectPos.tolist(),
        "targetPosition": targetPos.tolist(),
        "xyError": xyError,
        "zError": zError,
    }

# Run full pick and place demo with obstacles
def runDemoWithObstacles(obstacleConfigs: list[dict], hasRenderer: bool = True) -> dict:
    env, obs = arm_controller.initialize_environment(has_renderer=hasRenderer, obstacle_configs=obstacleConfigs)

    try:
        objectPos = getObjectPosition(env)
        targetPos = getTargetPosition(env)
        preferredTargetXy = targetPos[:2].copy()
        placeTargetXy = chooseSafePlaceXy(preferredTargetXy, obstacleConfigs)
        setTargetZoneXy(env, placeTargetXy)
        targetPos = getTargetPosition(env)
        print("Selected obstacle-free placement target:", targetPos.tolist())

        pregraspPos = np.array([objectPos[0], objectPos[1], safeZ], dtype=float)
        graspPos = np.array([objectPos[0], objectPos[1], objectPos[2] + graspZOffset], dtype=float)
        retreatPos = np.array([targetPos[0], targetPos[1], safeZ + postPlaceLift], dtype=float)

        print("Moving to start")
        obs = arm_controller.move_ee_to_position(step_count=120, goal_pos=startPos, gripper_state=-1, env=env, obs=obs)

        print("Navigating to pre-grasp above the object")
        obs = moveAlongRrt(env=env, obs=obs, startPos=startPos, goalPos=pregraspPos, gripperState=-1, obstacleConfigs=obstacleConfigs)

        print("Descending to grasp")
        obs = arm_controller.move_ee_to_position(step_count=120, goal_pos=graspPos, gripper_state=-1, env=env, obs=obs)

        print("Closing gripper")
        obs = arm_controller.set_gripper(step_count=50, gripper_state=1, env=env, obs=obs)

        print("Lifting object")
        obs = arm_controller.move_ee_to_position(step_count=120, goal_pos=pregraspPos, gripper_state=1, env=env, obs=obs)

        print("Transporting object to target")
        obs = moveAlongRrt(env=env, obs=obs, startPos=pregraspPos, goalPos=np.array([targetPos[0], targetPos[1], safeZ], dtype=float), gripperState=1, obstacleConfigs=obstacleConfigs)

        targetPos = getTargetPosition(env)
        obs = alignForPlace(env, obs, targetPos)

        print("Opening gripper")
        obs = arm_controller.set_gripper(step_count=50, gripper_state=-1, env=env, obs=obs)

        print("Retreating upward")
        obs = arm_controller.move_ee_to_position(step_count=120, goal_pos=retreatPos, gripper_state=-1, env=env, obs=obs)
        result = evaluateTestResult(env)
        result["placementTarget"] = targetPos.tolist()
        result["obstacles"] = obstacleConfigs
        return result
    finally:
        env.close()

def main():
    # Run the default pick and place demo
    runDemoWithObstacles(obstacleConfigs=defaultPickPlaceObstacles, hasRenderer=True)

# Run main when file is executed
if __name__ == "__main__":
    main()
