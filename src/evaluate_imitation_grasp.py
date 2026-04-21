from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np

from src.controllers import arm_controller
from src.grasp_playground import (GRASP_FRAME_CORRECTION,
                                  compute_table_resting_qpos,
                                  get_sim_object_transform,
                                  print_grasp_summary, scale_grasp_translation,
                                  visualize_grasp_sequence)
from src.imitation_grasp import (DATASET_ROOT, DEFAULT_MODEL_DIR,
                                 DEFAULT_MODEL_NAME, OBJECT_SCALE,
                                 GraspImitationPolicy, load_dataset_policy)
from src.utils.graspfactory_parser import GraspFactoryParser
from src.utils.transforms import world_grasp_from_object

LIFT_SUCCESS_THRESHOLD_M = 0.01  # Consider it a success if the object is lifted by at least this distance in meters.

"""Evaluation script for simplified imitation grasping on the GraspFactory dataset. Evaluates a pretrained imitation policy by executing grasp attempts in simulation and measuring lift success. 
A saved model must exist before running this script. This is done in imitation_grasp.py."""

def _choose_object_indices(total: int, num_objects: int, seed: int | None) -> list[int]:
    """Choose which object indices to evaluate, either sequentially or randomly based on a seed.

    Args:
        total (int): Total number of available objects.
        num_objects (int): Number of objects to evaluate.
        seed (int | None): Random seed for sampling, or None for sequential selection.

    Raises:
        ValueError: If num_objects is less than 1.

    Returns:
        list[int]: List of chosen object indices.
    """
    if total == 0:
        return []
    if num_objects <= 0:
        raise ValueError(f"num_objects must be >= 1, got {num_objects}.")

    if seed is None:
        seed = random.randint(0,100000)

    rng = np.random.default_rng(seed)
    count = min(total, num_objects)
    sampled = rng.choice(np.arange(total), size=count, replace=False)
    return [int(index) for index in sampled]


def _build_predicted_grasp(
    env,
    obs,
    policy: GraspImitationPolicy,
    object_scale: tuple[float, float, float],
):
    """Build the predicted grasp for a given object sample.

    Args:
        env (_type_): The simulation environment.
        obs (_type_): The observation from the simulation environment.
        policy (GraspImitationPolicy): The imitation policy to use for predicting grasps.
        object_scale (tuple[float, float, float]): The scale of the object in the simulation.

    Returns:
        tuple: A tuple containing the policy, object_T_grasp, world_T_object, and world_T_grasp.
    """
    # Compute the predicted grasp in the object's frame and convert it to the world frame.
    world_T_object = get_sim_object_transform(env)
    world_T_object_inv = np.linalg.inv(world_T_object)

    # Convert the end-effector position from the world frame to the object frame for policy input.
    eef_world_pos = np.asarray(obs["robot0_eef_pos"], dtype=float)
    eef_world_h = np.concatenate([eef_world_pos, np.array([1.0], dtype=float)])
    eef_object_pos = (world_T_object_inv @ eef_world_h)[:3]

    # Predict the grasp pose in the object frame, convert it to the world frame, and apply the necessary corrections.
    object_T_grasp, predicted_width, selected_prototype, prototype_score = (
        policy.predict_object_frame_grasp(current_eef_object_pos=eef_object_pos)
    )
    object_T_grasp = scale_grasp_translation(object_T_grasp, object_scale)
    object_T_grasp = object_T_grasp @ GRASP_FRAME_CORRECTION
    world_T_grasp = world_grasp_from_object(world_T_object, object_T_grasp)

    return (
        policy,
        predicted_width,
        selected_prototype,
        prototype_score,
        object_T_grasp,
        world_T_object,
        world_T_grasp,
    )


def evaluate_object(
    parser: GraspFactoryParser,
    policy: GraspImitationPolicy,
    object_idx: int,
    has_renderer: bool,
    visualize_grasp: bool,
    object_scale: tuple[float, float, float] = OBJECT_SCALE,
    model_source: str = "unknown",
    model_path: str = "",
) -> dict:
    """Evaluate a single grasp attempt on an object from the dataset by executing the predicted grasp in simulation and measuring lift success.

    Args:
        parser (GraspFactoryParser): The GraspFactory dataset parser.
        object_idx (int): The index of the object to evaluate.
        has_renderer (bool): Whether to enable the renderer.
        visualize_grasp (bool): Whether to visualize the grasp sequence.
        object_scale (tuple[float, float, float], optional): The scale of the object in the simulation. Defaults to OBJECT_SCALE.

    Returns:
        dict: A dictionary containing the evaluation results for the object.
    """
    sample = parser[object_idx]

    if len(sample.success_indices) == 0:
        return {
            "object_idx": object_idx,
            "object_id": sample.object_id,
            "demo_count": 0,
            "status": "skipped",
            "reason": "no_successful_grasps",
        }

    # Setup simulation environment
    object_qpos = compute_table_resting_qpos(
        mesh_path=sample.mesh_path,
        object_xy=(-0.12, 0.0),
        object_scale=object_scale,
    )

    env, obs = arm_controller.initialize_environment(
        dataset_mesh_path=str(sample.mesh_path),
        object_qpos=object_qpos,
        object_scale=object_scale,
        has_renderer=has_renderer,
    )

    # Execute the predicted grasp and evaluate lift success
    try:
        initial_object_pos, _ = env.get_object_pose()
        (
            policy,
            predicted_width,
            selected_prototype,
            prototype_score,
            object_T_grasp,
            world_T_object,
            world_T_grasp,
        ) = _build_predicted_grasp(
            env,
            obs,
            policy,
            object_scale,
        )

        if visualize_grasp:
            print_grasp_summary(
                sample=sample,
                object_T_grasp=object_T_grasp,
                world_T_object=world_T_object,
                world_T_grasp=world_T_grasp,
                grasp_width=predicted_width,
            )

        obs = visualize_grasp_sequence(env, obs, world_T_grasp)

        final_object_pos, _ = env.get_object_pose()
        lift_delta = float(final_object_pos[2] - initial_object_pos[2])
        # Consider it a success if the object was lifted by at least the threshold distance.
        lift_success = lift_delta > LIFT_SUCCESS_THRESHOLD_M

        return {
            "object_idx": object_idx,
            "object_id": sample.object_id,
            "demo_count": policy.demo_count,
            "cluster_count": policy.cluster_count,
            "selected_prototype": selected_prototype,
            "prototype_score": prototype_score,
            "predicted_width": predicted_width,
            "initial_object_pos": np.asarray(initial_object_pos, dtype=float).tolist(),
            "final_object_pos": np.asarray(final_object_pos, dtype=float).tolist(),
            "lift_delta": lift_delta,
            "lift_success": lift_success,
            "model_source": model_source,
            "model_path": model_path,
            "status": "ok",
        }
    finally:
        env.close()


def run_evaluation(
    num_objects: int,
    start_idx: int,
    seed: int | None,
    dataset_root: str | None,
    render: bool,
    visualize_count: int,
    model_path: str | None,
) -> dict:
    """Evaluate a pretrained imitation policy across a subset of dataset objects.

    Raises:
        FileNotFoundError: If no pretrained model exists at the resolved model path.

    Returns:
        dict: Evaluation results including per-object outcomes and aggregate success rate.
    """
    policy = load_dataset_policy(model_path)
    parser = GraspFactoryParser(root_dir=dataset_root, gripper="robotiq")

    object_indices = _choose_object_indices(
        total=len(parser),
        num_objects=num_objects,
        seed=seed,
    )

    total_eval = len(object_indices)

    # Visualize the last N = visualize_count objects in the run.
    visualize_start = max(0, total_eval - visualize_count)
    print(f"[Progress] Evaluating objects: 0/{total_eval} (0%)")

    results = []
    for eval_index, object_idx in enumerate(object_indices):
        visualize_grasp = eval_index >= visualize_start
        has_renderer = render or visualize_grasp
        try:
            result = evaluate_object(
                parser=parser,
                policy=policy,
                object_idx=object_idx,
                has_renderer=has_renderer,
                visualize_grasp=visualize_grasp,
                model_source="loaded",
                model_path=str(model_path or ""),
            )
        except Exception as exc:
            result = {
                "object_idx": object_idx,
                "status": "error",
                "error": str(exc),
            }
        results.append(result)

        pct = int(((eval_index + 1) * 100) / max(1, total_eval))
        print(f"[Progress] Evaluating objects: {eval_index + 1}/{total_eval} ({pct}%)")

    attempted = [entry for entry in results if entry.get("status") == "ok"]
    success_count = sum(entry["lift_success"] for entry in attempted)
    attempted_count = len(attempted)

    return {
        "num_requested": num_objects,
        "num_evaluated": total_eval,
        "num_attempted": attempted_count,
        "success_count": success_count,
        "success_rate": success_count / attempted_count if attempted_count else 0.0,
        "lift_success_threshold_m": LIFT_SUCCESS_THRESHOLD_M,
        "dataset_root": dataset_root,
        "seed": seed,
        "start_idx": start_idx,
        "render": render,
        "visualize_count": visualize_count,
        "model_path": str(model_path or ""),
        "policy_prototypes": policy.cluster_count,
        "policy_demo_count": policy.demo_count,
        "results": results,
    }


def write_results(report: dict, output_path: str | None) -> Path:
    """Write the results to a JSON file.

    Args:
        report (dict): The evaluation report dictionary.
        output_path (str | None): The output file path. If None, a default path is used.

    Returns:
        Path: The path to the written results file.
    """

    if output_path is None:
        output_path = "data/evaluations/imitation_grasp_eval.json"

    path = Path(output_path)
    if not path.is_absolute():
        project_root = Path(__file__).resolve().parents[1]
        path = project_root / path

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path


def build_arg_parser() -> argparse.ArgumentParser:
    """Build the argument parser for the evaluation script.

    Returns:
        argparse.ArgumentParser: The argument parser instance.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate a pretrained imitation grasp policy across dataset objects. "
            "A saved model must exist before running this script."
        )
    )
    parser.add_argument(
        "--num-objects",
        type=int,
        default=10,
        help="Number of objects to test (default: 10).",
    )
    parser.add_argument(
        "--start-idx",
        type=int,
        default=0,
        help="Starting object index for sequential evaluation (ignored when --seed is set).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="If set, evaluate a random subset of objects using this seed.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DATASET_ROOT,
        help="Optional override for the GraspFactory dataset root directory.",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render all evaluation rollouts (the last --visualize-count are always rendered).",
    )
    parser.add_argument(
        "--visualize-count",
        type=int,
        default=10,
        help="Number of final grasps to visualise in the MuJoCo viewer (default: 10).",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional JSON output path for the evaluation report.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            f"Path to a saved .npz model file or directory containing "
            f"'{DEFAULT_MODEL_NAME}.npz'. Defaults to {DEFAULT_MODEL_DIR}/."
        ),
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    report = run_evaluation(
        num_objects=args.num_objects,
        start_idx=args.start_idx,
        seed=args.seed,
        dataset_root=args.dataset_root,
        render=args.render,
        visualize_count=max(0, args.visualize_count),
        model_path=args.model_path,
    )
    report_path = write_results(report, args.output)

    print()
    print("=" * 50)
    print(f"  Grasps tested:    {report['num_attempted']} / {report['num_evaluated']}")
    print(f"  Lift successes:   {report['success_count']}")
    print(f"  Success rate:     {report['success_rate']:.1%}")
    print(f"  Policy:           {report['policy_prototypes']} prototypes, "
          f"{report['policy_demo_count']} training demos")
    print("=" * 50)
    print(f"Report written to: {report_path}")


if __name__ == "__main__":
    main()
