from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from src.controllers import arm_controller
from src.grasp.evaluate_imitation_grasp import (
    LIFT_SUCCESS_THRESHOLD_M,
    _build_predicted_grasp,
    evaluate_object,
)
from src.grasp.grasp_playground import (
    compute_table_resting_qpos,
    print_grasp_summary,
    visualize_grasp_sequence,
)
from src.grasp.imitation_grasp import DATASET_ROOT, OBJECT_SCALE, load_dataset_policy
from src.obstacle.pick_place_obstacle_rrt import (
    alignForPlace,
    chooseSafePlaceXy,
    defaultPickPlaceObstacles,
    evaluateTestResult,
    getTargetPosition,
    moveAlongRrt,
    postPlaceLift,
    safeZ,
    setTargetZoneXy,
)
from src.utils.graspfactory_parser import GraspFactoryParser

# TODO: update overall implementation interface to reduce obstacle pipeline duplication
def chooseRandomObjectIndex(
    parser: GraspFactoryParser,
    seed: int | None = None,
    objectIdx: int | None = None,
) -> int:
    """Choose an object with at least one successful grasp."""
    if objectIdx is not None:
        if len(parser[objectIdx].success_indices) == 0:
            raise ValueError(f"Selected object {parser[objectIdx].object_id} has no successful grasps.")
        return objectIdx

    rng = np.random.default_rng(seed)
    for candidateIndex in rng.permutation(len(parser)):
        if len(parser[int(candidateIndex)].success_indices) > 0:
            return int(candidateIndex)

    raise RuntimeError("No dataset objects with successful grasps were found.")


def buildDatasetEnvironment(sample, hasRenderer: bool, obstacleConfigs=None):
    """Create a scene for one dataset object."""
    objectQpos = compute_table_resting_qpos(
        mesh_path=sample.mesh_path,
        object_xy=(-0.20, 0.0),
        object_scale=OBJECT_SCALE,
    )
    return arm_controller.initialize_environment(
        dataset_mesh_path=str(sample.mesh_path),
        object_qpos=objectQpos,
        object_scale=OBJECT_SCALE,
        has_renderer=hasRenderer,
        obstacle_configs=obstacleConfigs,
    )


def runImitationGraspTrial(parser, policy, objectIdx: int, hasRenderer: bool) -> dict:
    """Run the existing imitation-grasp evaluation for one object."""
    return evaluate_object(
        parser=parser,
        policy=policy,
        object_idx=objectIdx,
        has_renderer=hasRenderer,
        visualize_grasp=hasRenderer,
        object_scale=OBJECT_SCALE,
        model_source="loaded",
        model_path="",
    )


def runIntegratedObstacleTask(parser, policy, objectIdx: int, hasRenderer: bool, obstacleConfigs: list[dict]) -> dict:
    """Retry the grasp in an obstacle scene and continue to placement."""
    sample = parser[objectIdx]
    env, obs = buildDatasetEnvironment(sample, hasRenderer=hasRenderer, obstacleConfigs=obstacleConfigs)

    try:
        initialObjectPos, _ = env.get_object_pose()
        preferredTargetXy = getTargetPosition(env)[:2].copy()
        placeTargetXy = chooseSafePlaceXy(preferredTargetXy, obstacleConfigs)
        setTargetZoneXy(env, placeTargetXy)
        targetPos = getTargetPosition(env)

        (
            _,
            predictedWidth,
            selectedPrototype,
            prototypeScore,
            objectTGrasp,
            worldTObject,
            worldTGrasp,
        ) = _build_predicted_grasp(env, obs, policy, OBJECT_SCALE)

        if hasRenderer:
            print_grasp_summary(
                sample=sample,
                object_T_grasp=objectTGrasp,
                world_T_object=worldTObject,
                world_T_grasp=worldTGrasp,
                grasp_width=predictedWidth,
            )

        obs = visualize_grasp_sequence(env, obs, worldTGrasp)

        liftedObjectPos, _ = env.get_object_pose()
        liftDelta = float(liftedObjectPos[2] - initialObjectPos[2])
        liftSuccess = liftDelta > LIFT_SUCCESS_THRESHOLD_M

        if not liftSuccess:
            return {
                "object_idx": objectIdx,
                "object_id": sample.object_id,
                "selected_prototype": selectedPrototype,
                "prototype_score": prototypeScore,
                "predicted_width": predictedWidth,
                "initial_object_pos": np.asarray(initialObjectPos, dtype=float).tolist(),
                "lifted_object_pos": np.asarray(liftedObjectPos, dtype=float).tolist(),
                "lift_delta": liftDelta,
                "lift_success": False,
                "status": "grasp_failed",
                "obstacles": obstacleConfigs,
                "placementTarget": targetPos.tolist(),
            }

        transportStartPos = np.asarray(obs["robot0_eef_pos"], dtype=float).copy()
        transportStartPos[2] = safeZ

        obs = arm_controller.move_ee_to_position(
            step_count=120,
            goal_pos=transportStartPos,
            gripper_state=1,
            env=env,
            obs=obs,
        )

        obs = moveAlongRrt(
            env=env,
            obs=obs,
            startPos=transportStartPos,
            goalPos=np.array([targetPos[0], targetPos[1], safeZ], dtype=float),
            gripperState=1,
            obstacleConfigs=obstacleConfigs,
        )

        targetPos = getTargetPosition(env)
        obs = alignForPlace(env, obs, targetPos)

        obs = arm_controller.set_gripper(
            step_count=50,
            gripper_state=-1,
            env=env,
            obs=obs,
        )

        retreatPos = np.array([targetPos[0], targetPos[1], safeZ + postPlaceLift], dtype=float)
        obs = arm_controller.move_ee_to_position(
            step_count=120,
            goal_pos=retreatPos,
            gripper_state=-1,
            env=env,
            obs=obs,
        )

        result = evaluateTestResult(env)
        result["object_idx"] = objectIdx
        result["object_id"] = sample.object_id
        result["selected_prototype"] = selectedPrototype
        result["prototype_score"] = prototypeScore
        result["predicted_width"] = predictedWidth
        result["initial_object_pos"] = np.asarray(initialObjectPos, dtype=float).tolist()
        result["lifted_object_pos"] = np.asarray(liftedObjectPos, dtype=float).tolist()
        result["lift_delta"] = liftDelta
        result["lift_success"] = True
        result["status"] = "ok"
        result["placementTarget"] = targetPos.tolist()
        result["obstacles"] = obstacleConfigs
        return result
    finally:
        env.close()


def writeResults(report: dict, outputPath: str | None) -> Path | None:
    """Write a JSON report if requested."""
    if outputPath is None:
        return None

    path = Path(outputPath)
    if not path.is_absolute():
        projectRoot = Path(__file__).resolve().parents[1]
        path = projectRoot / path

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path

# TODO: streamline flow for continuous execution
# TODO: enable selection from known subset of successful grasps. Seed 17 works as an example
def buildArgParser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run imitation grasping on one object, then retry it in an obstacle scene and place it if the grasp succeeds."
    )
    parser.add_argument(
        "--object-idx",
        type=int,
        default=None,
        help="Fixed dataset object index. Defaults to a random valid object.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Seed for object sampling.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DATASET_ROOT,
        help="Override the GraspFactory dataset root.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Saved imitation model file or directory.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without the MuJoCo viewer.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Write the combined run report to JSON.",
    )
    return parser


def main() -> None:
    args = buildArgParser().parse_args()
    parser = GraspFactoryParser(root_dir=args.dataset_root, gripper="robotiq")
    policy = load_dataset_policy(args.model_path)
    objectIdx = chooseRandomObjectIndex(parser, seed=args.seed, objectIdx=args.object_idx)
    sample = parser[objectIdx]

    print(f"Object index: {objectIdx}")
    print(f"Object id: {sample.object_id}")

    graspTrial = runImitationGraspTrial(
        parser=parser,
        policy=policy,
        objectIdx=objectIdx,
        hasRenderer=not args.headless,
    )

    integratedTrial = None
    if graspTrial["lift_success"]:
        print("Initial grasp succeeded. Running obstacle scene.")
        integratedTrial = runIntegratedObstacleTask(
            parser=parser,
            policy=policy,
            objectIdx=objectIdx,
            hasRenderer=not args.headless,
            obstacleConfigs=defaultPickPlaceObstacles,
        )
    else:
        print("Initial grasp failed. Skipping obstacle scene.")

    report = {
        "object_idx": objectIdx,
        "object_id": sample.object_id,
        "initial_grasp_trial": graspTrial,
        "obstacle_retry_trial": integratedTrial,
    }
    reportPath = writeResults(report, args.output)

    print(f"Initial grasp success: {graspTrial['lift_success']}")
    if integratedTrial is None:
        print("Obstacle retry: skipped")
    else:
        print(f"Obstacle grasp success: {integratedTrial['lift_success']}")
        print(f"Placement success: {integratedTrial.get('success', False)}")
    if reportPath is not None:
        print(f"Report written to: {reportPath}")


if __name__ == "__main__":
    main()
