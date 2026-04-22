from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from src.controllers import arm_controller

safeZ = 1.05
startPos = np.array([0.10, -0.20, safeZ], dtype=float)
goalPos = np.array([0.22, 0.18, safeZ], dtype=float)
defaultBounds = np.array([[-0.05, 0.30], [-0.30, 0.30]], dtype=float)
# Default obstacle layout
defaultObstacles = [
    {
        "pos": [0.10, -0.02, 0.92],
        "size": [0.05, 0.05, 0.10],
        "quat": [1.0, 0.0, 0.0, 0.0],
        "rgba": [0.85, 0.35, 0.2, 1.0],
    },
    {
        "pos": [0.17, 0.08, 0.92],
        "size": [0.04, 0.06, 0.10],
        "quat": [0.9238795, 0.0, 0.0, 0.3826834],
        "rgba": [0.2, 0.45, 0.85, 1.0],
    },
]
@dataclass
class RRTNode:
    point: np.ndarray
    parentIndex: int | None

# Check if xy point is inside any obstacle
def pointInCollision(pointXy: np.ndarray, obstacles: list[dict], robotRadius: float) -> bool:
    for obstacle in obstacles:
        obstacleXy = np.asarray(obstacle["pos"][:2], dtype=float)
        halfExtents = np.asarray(obstacle["size"][:2], dtype=float) + robotRadius
        lower = obstacleXy - halfExtents
        upper = obstacleXy + halfExtents
        if np.all(pointXy >= lower) and np.all(pointXy <= upper):
            return True
    return False

# Check if straight line hits any obstacle
def segmentInCollision(startXy: np.ndarray, endXy: np.ndarray, obstacles: list[dict], robotRadius: float, samples: int = 30) -> bool:
    for alpha in np.linspace(0.0, 1.0, samples):
        point = startXy + alpha * (endXy - startXy)
        if pointInCollision(point, obstacles, robotRadius):
            return True
    return False

# Sample a random collision free xy point
def sampleFreePoint(rng: np.random.Generator, bounds: np.ndarray, obstacles: list[dict], robotRadius: float) -> np.ndarray:
    while True:
        candidate = np.array(
            [
                rng.uniform(bounds[0, 0], bounds[0, 1]),
                rng.uniform(bounds[1, 0], bounds[1, 1]),
            ],
            dtype=float,
        )
        if not pointInCollision(candidate, obstacles, robotRadius):
            return candidate

# Build final path by walking back through parent nodes
def reconstructPath(nodes: list[RRTNode], endIndex: int) -> np.ndarray:
    points = []
    currentIndex = endIndex
    while currentIndex is not None:
        node = nodes[currentIndex]
        points.append(node.point)
        currentIndex = node.parentIndex
    points.reverse()
    return np.asarray(points, dtype=float)

# Shorten path by removing unnecessary middle waypoints
def simplifyPath(pathXy: np.ndarray, obstacles: list[dict], robotRadius: float) -> np.ndarray:
    if len(pathXy) <= 2:
        return pathXy

    simplified = [pathXy[0]]
    anchorIndex = 0
    while anchorIndex < len(pathXy) - 1:
        nextIndex = len(pathXy) - 1
        while nextIndex > anchorIndex + 1:
            if not segmentInCollision(pathXy[anchorIndex], pathXy[nextIndex], obstacles, robotRadius):
                break
            nextIndex -= 1
        simplified.append(pathXy[nextIndex])
        anchorIndex = nextIndex

    return np.asarray(simplified, dtype=float)

# Plan an obstacle avoiding path in xy using RRT
def planRrtPath(startXy: np.ndarray, goalXy: np.ndarray, obstacles: list[dict], bounds: np.ndarray, stepSize: float = 0.05,
                maxIterations: int = 3000, goalBias: float = 0.15, robotRadius: float = 0.035, seed: int = 7) -> np.ndarray:
    if pointInCollision(startXy, obstacles, robotRadius):
        raise ValueError("Start point is inside obstacle")
    if pointInCollision(goalXy, obstacles, robotRadius):
        raise ValueError("Goal point is inside obstacle")

    rng = np.random.default_rng(seed)
    nodes = [RRTNode(point=np.asarray(startXy, dtype=float), parentIndex=None)]

    for _ in range(maxIterations):
        if rng.random() < goalBias:
            sampleXy = goalXy
        else:
            sampleXy = sampleFreePoint(rng, bounds, obstacles, robotRadius)

        distances = [np.linalg.norm(node.point - sampleXy) for node in nodes]
        nearestIndex = int(np.argmin(distances))
        nearestXy = nodes[nearestIndex].point

        direction = sampleXy - nearestXy
        distance = np.linalg.norm(direction)
        if distance == 0:
            continue

        step = min(stepSize, distance)
        candidateXy = nearestXy + (direction / distance) * step

        if segmentInCollision(nearestXy, candidateXy, obstacles, robotRadius):
            continue

        nodes.append(RRTNode(point=candidateXy, parentIndex=nearestIndex))
        newIndex = len(nodes) - 1

        if np.linalg.norm(candidateXy - goalXy) <= stepSize and not segmentInCollision(candidateXy, goalXy, obstacles, robotRadius):
            nodes.append(RRTNode(point=np.asarray(goalXy, dtype=float), parentIndex=newIndex))
            return simplifyPath(reconstructPath(nodes, len(nodes) - 1), obstacles, robotRadius)

    raise RuntimeError("RRT failed to find a collision free path")

# Add fixed z height to each xy waypoint
def xyPathToCartesian(pathXy: np.ndarray, zHeight: float) -> np.ndarray:
    zColumn = np.full((len(pathXy), 1), zHeight, dtype=float)
    return np.hstack([pathXy, zColumn])

# Run demo with the given obstacle setup
def runDemoWithObstacles(obstacleConfigs: list[dict], hasRenderer: bool = True) -> np.ndarray:
    pathXy = planRrtPath(startXy=np.asarray(startPos[:2], dtype=float), goalXy=np.asarray(goalPos[:2], dtype=float), obstacles=obstacleConfigs, bounds=defaultBounds)
    cartesianPath = xyPathToCartesian(pathXy, safeZ)

    print("Planned RRT waypoints:")
    for waypoint in cartesianPath:
        print(waypoint.tolist())

    env, obs = arm_controller.initialize_environment(has_renderer=hasRenderer, obstacle_configs=obstacleConfigs)

    try:
        obs = arm_controller.move_ee_to_position(step_count=120, goal_pos=startPos, gripper_state=-1, env=env, obs=obs)
        obs = arm_controller.move_ee_along_trajectory(trajectory_points=cartesianPath, step_count=max(240, len(cartesianPath) * 80), gripper_state=-1, env=env, obs=obs)
        obs = arm_controller.move_ee_to_position(step_count=80, goal_pos=goalPos, gripper_state=-1, env=env, obs=obs)
    finally:
        env.close()

    return cartesianPath

def main() -> None:
    # Run the obstacle navigation demo
    runDemoWithObstacles(obstacleConfigs=defaultObstacles, hasRenderer=True)

if __name__ == "__main__":
    main()
