from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.controllers import arm_controller
from src.grasp_pipeline.config import ExecutionConfig, SceneConfig
from src.grasp_pipeline.models import GraspCandidate, GraspObjectSample, TrialRecord
from src.utils.transforms import (
    body_pose_to_transform,
    offset_transform_along_local_axis,
    quat_wxyz_to_xyzw,
    transform_to_pose,
    world_grasp_from_object,
)


def _step_scene(env, obs, step_count: int):
    """Advance the simulation while keeping the current controls at zero."""
    for _ in range(step_count):
        action = np.zeros(7)
        obs, _, _, _ = env.step(action)
        env.render()
    return obs


def _get_sim_object_transform(env):
    """Return the current object-body pose in the world frame."""
    body_pos = env.sim.data.body_xpos[env.object_body_id]
    body_xmat = env.sim.data.body_xmat[env.object_body_id]
    return body_pose_to_transform(body_pos, body_xmat)


def _evaluate_trial(
    env,
    candidate: GraspCandidate,
    initial_object_pos: np.ndarray,
    execution_config: ExecutionConfig,
) -> TrialRecord:
    """Measure whether the object was lifted and retained near the gripper trajectory."""
    final_object_pos = env.sim.data.body_xpos[env.object_body_id].copy()
    lift_delta_m = float(final_object_pos[2] - initial_object_pos[2])
    xy_slip_m = float(np.linalg.norm(final_object_pos[:2] - initial_object_pos[:2]))

    success = (
        lift_delta_m >= execution_config.min_successful_lift_m
        and xy_slip_m <= execution_config.max_post_lift_xy_slip_m
    )

    failure_reason = "success"
    if not success:
        if lift_delta_m < execution_config.min_successful_lift_m:
            failure_reason = "insufficient_lift"
        else:
            failure_reason = "excessive_xy_slip"

    return TrialRecord(
        object_id="",
        grasp_id=candidate.grasp_id,
        success=success,
        failure_reason=failure_reason,
        initial_object_pos=tuple(float(v) for v in initial_object_pos),
        final_object_pos=tuple(float(v) for v in final_object_pos),
        lift_delta_m=lift_delta_m,
        grasp_width_m=candidate.grasp_width_m,
        score=candidate.score,
        metadata={"xy_slip_m": xy_slip_m, **candidate.metadata},
    )


def execute_grasp_candidate(
    sample: GraspObjectSample,
    candidate: GraspCandidate,
    scene_config: SceneConfig,
    execution_config: ExecutionConfig,
) -> TrialRecord:
    """Create an environment, execute one scripted grasp, and return the trial record."""
    object_scale = scene_config.object_scale
    if object_scale is None:
        object_scale = tuple(sample.metadata.get("object_scale", (0.001, 0.001, 0.001)))

    env, obs = arm_controller.initialize_environment(
        dataset_mesh_path=str(sample.mesh_path),
        object_scale=object_scale,
        object_xy=scene_config.object_xy,
        object_collision_approximation=scene_config.collision_mode,
        object_drop_height=scene_config.object_drop_height,
    )

    try:
        # Let the object settle after the initial drop so the scripted grasp starts
        # from a repeatable resting pose.
        obs = _step_scene(env, obs, execution_config.settle_steps)

        initial_object_pos = env.sim.data.body_xpos[env.object_body_id].copy()
        world_T_object = _get_sim_object_transform(env)
        world_T_grasp = world_grasp_from_object(world_T_object, candidate.object_T_grasp)
        world_T_pregrasp = offset_transform_along_local_axis(
            world_T_grasp,
            axis_index=2,
            distance=execution_config.pregrasp_offset_m,
            sign=-1.0,
        )

        pregrasp_pos, pregrasp_quat = transform_to_pose(world_T_pregrasp)
        grasp_pos, grasp_quat = transform_to_pose(world_T_grasp)

        obs = arm_controller.move_ee_to_position(
            step_count=execution_config.move_steps,
            goal_pos=list(execution_config.safe_start_pos),
            gripper_state=-1,
            env=env,
            obs=obs,
        )
        obs = arm_controller.move_ee_to_pose(
            step_count=execution_config.move_steps,
            goal_pos=pregrasp_pos,
            goal_quat=quat_wxyz_to_xyzw(grasp_quat),
            gripper_state=-1,
            env=env,
            obs=obs,
        )
        obs = arm_controller.move_ee_to_pose(
            step_count=execution_config.move_steps,
            goal_pos=grasp_pos,
            goal_quat=quat_wxyz_to_xyzw(grasp_quat),
            gripper_state=-1,
            env=env,
            obs=obs,
        )
        obs = arm_controller.set_gripper(
            step_count=execution_config.gripper_close_steps,
            gripper_state=1,
            env=env,
            obs=obs,
        )

        lift_pos = grasp_pos.copy()
        lift_pos[2] += execution_config.lift_distance_m
        obs = arm_controller.move_ee_to_pose(
            step_count=execution_config.lift_steps,
            goal_pos=lift_pos,
            goal_quat=quat_wxyz_to_xyzw(grasp_quat),
            gripper_state=1,
            env=env,
            obs=obs,
        )
        obs = _step_scene(env, obs, 20)

        result = _evaluate_trial(
            env=env,
            candidate=candidate,
            initial_object_pos=initial_object_pos,
            execution_config=execution_config,
        )
        return TrialRecord(
            object_id=sample.object_id,
            grasp_id=result.grasp_id,
            success=result.success,
            failure_reason=result.failure_reason,
            initial_object_pos=result.initial_object_pos,
            final_object_pos=result.final_object_pos,
            lift_delta_m=result.lift_delta_m,
            grasp_width_m=result.grasp_width_m,
            score=result.score,
            metadata={
                **sample.metadata,
                **result.metadata,
                "scene_config": {
                    "object_xy": list(scene_config.object_xy),
                    "object_scale": list(object_scale),
                    "collision_mode": scene_config.collision_mode,
                    "object_drop_height": scene_config.object_drop_height,
                },
                "object_T_grasp": candidate.object_T_grasp.tolist(),
            },
        )
    finally:
        env.close()


def append_trial_record(output_path: str | Path, record: TrialRecord):
    """Append one trial record to a JSONL benchmark log."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record.__dict__) + "\n")
