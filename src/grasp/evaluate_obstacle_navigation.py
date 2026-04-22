from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

from src.obstacle.pick_place_obstacle_rrt import (
    planningBounds,
    runDemoWithObstacles,
)

obstacleZ = 0.92
obstacleSizes = [
    [0.04, 0.04, 0.10],
    [0.05, 0.05, 0.10],
    [0.06, 0.04, 0.10],
]
# Keep areas free when sampling obstacles
startClearancePoint = np.array([0.10, -0.22], dtype=float)
objectClearancePoint = np.array([-0.20, 0.00], dtype=float)
targetClearancePoint = np.array([0.20, 0.00], dtype=float)
keepOutRadius = 0.10
reportPath = Path("data/evaluations/obstacle_navigation_eval.json")

# Reject points which fall inside a protected radius
def forbiddenRadius(candidateXy: np.ndarray, protectedXy: np.ndarray, radius: float) -> bool:
    return float(np.linalg.norm(candidateXy - protectedXy)) < radius

def yawDegreesToQuat(yawDegrees: float) -> list[float]:
    yawRadians = np.deg2rad(yawDegrees)
    halfAngle = yawRadians / 2.0
    return [float(np.cos(halfAngle)), 0.0, 0.0, float(np.sin(halfAngle))]

def sampleRandomObstacle(rng: np.random.Generator, obstacleCount: int) -> list[dict]:
    obstacles = []
    attempts = 0
    while len(obstacles) < obstacleCount:
        attempts += 1
        if attempts > 3000:
            raise RuntimeError("Failed to sample valid obstacle layout")

        candidateXy = np.array(
            [
                rng.uniform(planningBounds[0, 0] + 0.04, planningBounds[0, 1] - 0.04),
                rng.uniform(planningBounds[1, 0] + 0.04, planningBounds[1, 1] - 0.04),
            ],
            dtype=float,
        )
        if forbiddenRadius(candidateXy, startClearancePoint, keepOutRadius):
            continue
        if forbiddenRadius(candidateXy, objectClearancePoint, keepOutRadius):
            continue
        if forbiddenRadius(candidateXy, targetClearancePoint, keepOutRadius):
            continue

        size = copy.deepcopy(obstacleSizes[int(rng.integers(0, len(obstacleSizes)))])
        obstacle = {
            "pos": [float(candidateXy[0]), float(candidateXy[1]), obstacleZ],
            "size": size,
            "quat": yawDegreesToQuat(float(rng.uniform(0.0, 90.0))),
            "rgba": [
                float(rng.uniform(0.1, 0.9)),
                float(rng.uniform(0.1, 0.9)),
                float(rng.uniform(0.1, 0.9)),
                1.0,
            ],
        }

        overlapsExisting = False
        for existingObstacle in obstacles:
            existingXy = np.asarray(existingObstacle["pos"][:2], dtype=float)
            # Leaving some gap between obstacles for safety
            minDistance = max(
                existingObstacle["size"][0] + obstacle["size"][0],
                existingObstacle["size"][1] + obstacle["size"][1],
            ) + 0.02
            if np.linalg.norm(candidateXy - existingXy) < minDistance:
                overlapsExisting = True
                break
        if overlapsExisting:
            continue

        obstacles.append(obstacle)
    return obstacles

def runEvaluation(testCount: int, seed: int, obstacleCount: int, hasRenderer: bool) -> dict:
    rng = np.random.default_rng(seed)
    results = []

    for testIndex in range(testCount):
        # Build obstacle layout for each test
        obstacles = sampleRandomObstacle(rng, obstacleCount=obstacleCount)
        result = runDemoWithObstacles(obstacleConfigs=obstacles, hasRenderer=hasRenderer)
        result["testIndex"] = testIndex
        results.append(result)

    successCount = sum(1 for result in results if result["success"])
    return {
        "testCount": testCount,
        "successCount": successCount,
        "successRate": successCount / testCount if testCount else 0.0,
        "seed": seed,
        "obstacleCount": obstacleCount,
        "results": results,
    }

def main() -> None:
    report = runEvaluation(testCount=10, seed=7, obstacleCount=2, hasRenderer=True)
    fullReportPath = Path(__file__).resolve().parents[1] / reportPath
    fullReportPath.parent.mkdir(parents=True, exist_ok=True)
    fullReportPath.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Tests run: {report['testCount']}")
    print(f"Successes: {report['successCount']}")
    print(f"Success rate: {report['successRate']:.2%}")
    print(f"Report written to: {fullReportPath}")

if __name__ == "__main__":
    main()
