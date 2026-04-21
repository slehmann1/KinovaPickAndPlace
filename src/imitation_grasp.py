from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation as R

from src.utils.graspfactory_parser import GraspFactoryParser

OBJECT_SCALE = (0.001, 0.001, 0.001)
DATASET_ROOT = "C:/Users/samue/PycharmProjects/GraspFactory/graspfactory/robotiq/robotiq"
DEFAULT_MODEL_DIR = "data/models/imitation_grasp"
DEFAULT_MODEL_NAME = "Model_B"
DEFAULT_MAX_GRASPS_PER_OBJECT = 8
DEFAULT_MAX_TRAINING_GRASPS = 300000

"""Imitation learning for grasping using the GraspFactory dataset this script trains an imitation learngin policy by clustering successful grasp demonstrations into a specified number of prototypes. 
The policy then predicts the best prototype grasp for a given object pose, which is executed in simulation to evaluate its performance."""

@dataclass
class GraspImitationPolicy:
    """Imitation policy for grasping using the GraspFactory dataset.
    """

    prototype_grasps: np.ndarray
    prototype_widths: np.ndarray
    demo_count: int
    cluster_count: int

    def predict_object_frame_grasp(
        self,
        current_eef_object_pos: np.ndarray | None = None,
    ) -> tuple[np.ndarray, float, int, float]:
        """Select the best grasp for the current state of the end effector.

        Returns:
            tuple[np.ndarray, float, int, float]:
                (grasp transform, predicted width, selected prototype index, score)
        """
        scores = []
        for grasp in self.prototype_grasps:
            grasp_translation = grasp[:3, 3]
            approach_axis = grasp[:3, 2]

            # If the current end effector position relative to the object is known, prefer grasps that require less translation from the current position.
            # Otherwise, consider distance from grasp to the object.
            if current_eef_object_pos is None:
                translation_cost = float(np.linalg.norm(grasp_translation))
            else:
                translation_cost = float(np.linalg.norm(grasp_translation - current_eef_object_pos))

            to_center = -grasp_translation
            norm_to_center = float(np.linalg.norm(to_center))
            if norm_to_center > 1e-8:
                to_center = to_center / norm_to_center
                center_alignment = float(np.dot(approach_axis, to_center))
            else:
                center_alignment = 0.0

            # Lower score is better. Reward grasps that approach toward object center.
            score = translation_cost + 0.25 * (1.0 - center_alignment)
            scores.append(score)

        best_index = int(np.argmin(scores))
        return (
            np.asarray(self.prototype_grasps[best_index], dtype=float).copy(),
            float(self.prototype_widths[best_index]),
            best_index,
            float(scores[best_index]),
        )

    def save(self, path: str | Path) -> Path:
        """Save the policy as an np archive.

        Args:
            path (str | Path): The file path to save the policy.

        Returns:
            Path: The path to the saved policy file.
        """

        model_path = Path(path)
        if not model_path.is_absolute():
            project_root = Path(__file__).resolve().parents[1]
            model_path = project_root / model_path
        model_path.parent.mkdir(parents=True, exist_ok=True)

        np.savez_compressed(
            model_path,
            prototype_grasps=np.asarray(self.prototype_grasps, dtype=float),
            prototype_widths=np.asarray(self.prototype_widths, dtype=float),
            demo_count=np.array([self.demo_count], dtype=np.int32),
            cluster_count=np.array([self.cluster_count], dtype=np.int32),
        )
        return model_path

    @classmethod
    def load(cls, path: str | Path) -> "GraspImitationPolicy":
        """Load a previously saved policy from disk.

        Args:
            path (str | Path): The file path to load the policy from.

        Raises:
            ValueError: If the loaded data is invalid.

        Returns:
            GraspImitationPolicy: The loaded policy instance.
        """
        model_path = Path(path)
        data = np.load(model_path, allow_pickle=False)

        policy = cls(
            prototype_grasps=np.asarray(data["prototype_grasps"], dtype=float),
            prototype_widths=np.asarray(data["prototype_widths"], dtype=float),
            demo_count=int(np.asarray(data["demo_count"]).reshape(-1)[0]),
            cluster_count=int(np.asarray(data["cluster_count"]).reshape(-1)[0]),
        )

        if len(policy.prototype_grasps) != len(policy.prototype_widths):
            raise ValueError("Invalid model file: prototype grasp and width counts differ.")
        return policy


def _unambiguous_quaternions(quaternions: np.ndarray) -> np.ndarray:
    """Resolve quaternion sign ambiguity.

    Args:
        quaternions (np.ndarray): Array of quaternions to canonicalize.

    Returns:
        np.ndarray: Array of canonicalized quaternions.
    """
    quats = np.asarray(quaternions, dtype=float).copy()
    quats[quats[:, 3] < 0.0] *= -1.0
    return quats


def _cluster_prototypes(
    successful_grasps: np.ndarray,
    successful_widths: np.ndarray,
    num_prototypes: int,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """Cluster successful grasps. This preserves grasp orientation.

    Args:
        successful_grasps (np.ndarray): Array of successful grasp poses.
        successful_widths (np.ndarray): Array of successful grasp widths.
        num_prototypes (int): Number of prototypes to generate.
        seed (int, optional): Random seed for reproducibility. Defaults to 0.

    Returns:
        tuple[np.ndarray, np.ndarray]: Tuple containing arrays of prototype grasps and their corresponding widths.
    """
    demo_count = len(successful_grasps)
    k = min(max(1, num_prototypes), demo_count)

    translations = np.asarray(successful_grasps[:, :3, 3], dtype=np.float32)
    quaternions = _unambiguous_quaternions(
        R.from_matrix(successful_grasps[:, :3, :3]).as_quat()
    ).astype(np.float32, copy=False)

    # Scale translation and rotation features to balance their influence in clustering. 
    # Scales were chosen quickly and can be tuned.
    translation_scale = 2.5
    rotation_scale = 0.8

    features = np.concatenate(
        [translation_scale * translations, rotation_scale * quaternions],
        axis=1,
    ).astype(np.float32, copy=False)

    rng = np.random.default_rng(seed)
    init_ids = rng.choice(np.arange(demo_count), size=k, replace=False)
    centroids = features[init_ids].copy()

    def _assign_clusters_chunked(
        x: np.ndarray,
        c: np.ndarray,
        chunk_size: int = 4096,
    ) -> np.ndarray:
        """Assign each feature row to nearest centroid with bounded memory."""
        assignments_local = np.empty(len(x), dtype=np.int32)
        c_norm = np.sum(c * c, axis=1)

        for start in range(0, len(x), chunk_size):
            end = min(start + chunk_size, len(x))
            x_chunk = x[start:end]
            x_norm = np.sum(x_chunk * x_chunk, axis=1, keepdims=True)
            # Squared Euclidean distance: ||x-c||^2 = ||x||^2 + ||c||^2 - 2 x.c
            d2_chunk = x_norm + c_norm[None, :] - 2.0 * (x_chunk @ c.T)
            assignments_local[start:end] = np.argmin(d2_chunk, axis=1).astype(np.int32)

        return assignments_local

    # Lloyd's algorithm for k-means clustering.
    assignments = np.zeros(demo_count, dtype=np.int32)
    for _ in range(40):
        new_assignments = _assign_clusters_chunked(features, centroids)
        if np.array_equal(new_assignments, assignments):
            break
        assignments = new_assignments

        for cluster_idx in range(k):
            mask = assignments == cluster_idx
            if np.any(mask):
                centroids[cluster_idx] = np.mean(features[mask], axis=0)
            else:
                # Re-seed empty clusters from a random demo.
                centroids[cluster_idx] = features[rng.integers(0, demo_count)]

    # For each cluster, select the grasp closest to the centroid and average widths of all grasps in the cluster for the width.
    prototype_grasps = []
    prototype_widths = []
    for cluster_idx in range(k):
        member_ids = np.where(assignments == cluster_idx)[0]
        if len(member_ids) == 0:
            continue

        member_features = features[member_ids]
        centroid = centroids[cluster_idx]
        medoid_local_idx = int(np.argmin(np.sum((member_features - centroid) ** 2, axis=1)))
        medoid_idx = int(member_ids[medoid_local_idx])

        prototype_grasps.append(successful_grasps[medoid_idx])
        prototype_widths.append(float(np.mean(successful_widths[member_ids])))

    return (
        np.asarray(prototype_grasps, dtype=float),
        np.asarray(prototype_widths, dtype=float),
    )


def train_imitation_policy(
    successful_grasps: np.ndarray,
    successful_widths: np.ndarray,
    num_prototypes: int = 12,
) -> GraspImitationPolicy:
    """Train an imitation learning policy from successful demonstrations.

    Args:
        successful_grasps (np.ndarray): Array of successful grasp poses.
        successful_widths (np.ndarray): Array of successful grasp widths.
        num_prototypes (int, optional): Number of prototypes to generate. Defaults to 12.

    Returns:
        GraspImitationPolicy: Policy with grasp prototypes and widths.
    """
    if successful_grasps.ndim != 3 or successful_grasps.shape[1:] != (4, 4):
        raise ValueError(
            f"Expected successful_grasps with shape (N, 4, 4), got {successful_grasps.shape}."
        )
    if len(successful_grasps) == 0:
        raise ValueError("No successful grasps available for imitation learning.")
    if len(successful_widths) != len(successful_grasps):
        raise ValueError(
            "successful_widths and successful_grasps must have the same number of samples."
        )

    prototype_grasps, prototype_widths = _cluster_prototypes(
        successful_grasps=successful_grasps,
        successful_widths=successful_widths,
        num_prototypes=num_prototypes,
    )

    return GraspImitationPolicy(
        prototype_grasps=prototype_grasps,
        prototype_widths=prototype_widths,
        demo_count=len(successful_grasps),
        cluster_count=len(prototype_grasps),
    )


def _resolve_model_path(model_path: str | None, model_name: str = DEFAULT_MODEL_NAME) -> Path:
    """Resolve model path for saving or loading.

    Args:
        model_path (str | None): The path to the model file or directory.
        model_name (str): Model filename stem used when a directory is provided.

    Returns:
        Path: The resolved path to the model file.
    """
    default_dir = Path(DEFAULT_MODEL_DIR)
    requested = Path(model_path) if model_path else default_dir
    if requested.suffix.lower() == ".npz":
        return requested
    return requested / f"{model_name}.npz"


def _collect_successes_from_parser(
    parser: GraspFactoryParser,
    max_grasps_per_object: int = DEFAULT_MAX_GRASPS_PER_OBJECT,
    max_training_grasps: int = DEFAULT_MAX_TRAINING_GRASPS,
    seed: int = 0,
) -> tuple[np.ndarray, np.ndarray, int]:
    """Aggregate successful grasps across all objects

    Args:
        parser (GraspFactoryParser): The GraspFactory dataset parser.
        max_grasps_per_object (int, optional): Maximum number of grasps to consider per object. Defaults to DEFAULT_MAX_GRASPS_PER_OBJECT.
        max_training_grasps (int, optional): Maximum number of grasps to use for training. Defaults to DEFAULT_MAX_TRAINING_GRASPS.
        seed (int, optional): Random seed for sampling. Defaults to 0.

    Returns:
        tuple[np.ndarray, np.ndarray, int]: A tuple containing the aggregated successful grasps, their corresponding widths, and the number of objects used.
    """
    all_grasps = []
    all_widths = []
    used_objects = 0
    skipped_files = 0
    total = len(parser.grasp_files)
    total_selected = 0
    total_success_available = 0
    rng = np.random.default_rng(seed)

    print(f"Start aggregating successful grasps from {total} objects...")

    for idx, grasp_file in enumerate(parser.grasp_files):
        pct = int((idx * 100) / max(1, total))
        print(f"[Progress] Aggregating successful grasps: {idx}/{total} ({pct}%)")
        try:
            data = np.load(grasp_file, allow_pickle=True)
            grasps = np.asarray(data["grasps"], dtype=float)
            widths = np.asarray(data["grasp_widths"], dtype=float)
            success_indices = np.asarray(data["success_indices"], dtype=np.int64).reshape(-1)
        except Exception:
            skipped_files += 1
            continue

        if grasps.ndim != 3 or grasps.shape[1:] != (4, 4) or len(widths) != len(grasps):
            skipped_files += 1
            continue

        # Filter to valid success indices that are within bounds of the grasps array.
        valid_success_indices = success_indices[
            (success_indices >= 0) & (success_indices < len(grasps))
        ]
        total_success_available += int(len(valid_success_indices))
        if len(valid_success_indices) == 0:
            continue

        # Enforce per-object and global caps on the number of grasps to consider for training.
        remaining_budget = max_training_grasps - total_selected
        if remaining_budget <= 0:
            continue
        
        # If the number of valid successful grasps exceeds the per-object cap, randomly sample a subset to consider.
        take_count = min(len(valid_success_indices), max_grasps_per_object, remaining_budget)
        if take_count <= 0:
            continue

        if take_count == len(valid_success_indices):
            chosen_indices = valid_success_indices
        else:
            selected_local = rng.choice(len(valid_success_indices), size=take_count, replace=False)
            chosen_indices = valid_success_indices[selected_local]

        # Append successful grasps and widths to the aggregate list.
        all_grasps.append(np.asarray(grasps[chosen_indices], dtype=np.float32))
        all_widths.append(np.asarray(widths[chosen_indices], dtype=np.float32))
        used_objects += 1
        total_selected += int(take_count)

    print(
        f"[Progress] Aggregation complete: {used_objects} objects contributed "
        f"{total_selected} sampled successful grasps "
        f"(from {total_success_available} available)."
    )
    if skipped_files:
        print(f"[Progress] Aggregation skipped {skipped_files} malformed/unreadable files.")
    if total_selected >= max_training_grasps:
        print(f"[Progress] Reached global training sample cap: {max_training_grasps} grasps.")

    return np.concatenate(all_grasps, axis=0), np.concatenate(all_widths, axis=0), used_objects


def load_dataset_policy(model_path: str | None) -> GraspImitationPolicy:
    """Load a pre-trained dataset-level policy; raise FileNotFoundError if not found.

    Args:
        model_path (str | None): Model file or directory override. Defaults to DEFAULT_MODEL_DIR.

    Raises:
        FileNotFoundError: If no saved model exists at the resolved path.

    Returns:
        GraspImitationPolicy: The loaded policy.
    """
    resolved = _resolve_model_path(model_path, DEFAULT_MODEL_NAME)
    if not resolved.exists():
        raise FileNotFoundError(
            f"No pretrained model found at: {resolved}\n"
            "Train a model first by running imitation_grasp.py (with --save-model)."
        )
    print(f"[Progress] Loading pretrained model from {resolved}...")
    policy = GraspImitationPolicy.load(resolved)
    print(
        f"[Progress] Model loaded — {policy.cluster_count} prototypes, "
        f"{policy.demo_count} training demos."
    )
    return policy


def _load_or_train_dataset_policy(
    parser: GraspFactoryParser,
    model_path: str | None,
    use_saved_model: bool,
    save_model: bool,
    num_prototypes: int,
    max_grasps_per_object: int = DEFAULT_MAX_GRASPS_PER_OBJECT,
    max_training_grasps: int = DEFAULT_MAX_TRAINING_GRASPS,
) -> tuple[GraspImitationPolicy, str, Path, int]:
    """Load or train one global policy using successful grasps from all dataset objects."""
    if use_saved_model:
        try:
            policy = load_dataset_policy(model_path)
            return policy, "loaded", _resolve_model_path(model_path, DEFAULT_MODEL_NAME), -1
        except FileNotFoundError:
            print("[Progress] No existing dataset-level model found. Proceeding to train a new model from the dataset...")

    resolved_model_path = _resolve_model_path(model_path, DEFAULT_MODEL_NAME)
    print("[Progress] Building dataset-level training set from all objects...")
    successful_grasps, successful_widths, used_objects = _collect_successes_from_parser(
        parser,
        max_grasps_per_object=max_grasps_per_object,
        max_training_grasps=max_training_grasps,
    )
    print(
        f"[Progress] Training dataset-level policy with {len(successful_grasps)} successful grasps "
        f"from {used_objects} objects..."
    )
    policy = train_imitation_policy(
        successful_grasps=successful_grasps,
        successful_widths=successful_widths,
        num_prototypes=num_prototypes,
    )
    print("[Progress] Dataset-level policy training complete.")

    model_source = "trained"
    if save_model:
        print(f"[Progress] Saving dataset-level model to {resolved_model_path}...")
        resolved_model_path = policy.save(resolved_model_path)
        print("[Progress] Model saved.")
        model_source = "trained_and_saved"

    return policy, model_source, resolved_model_path, used_objects

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a simplified imitation policy from successful grasps and execute one attempt."
    )
    parser.add_argument(
        "--object-idx",
        type=int,
        default=0,
        help="Dataset object index to imitate and execute.",
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Disable the MuJoCo viewer.",
    )
    parser.add_argument(
        "--dataset-root",
        type=str,
        default=DATASET_ROOT,
        help="Optional override for the GraspFactory dataset root directory.",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help=(
            "Model file path or directory. If directory, the model is saved/loaded as "
            f"{DEFAULT_MODEL_NAME}.npz inside it."
        ),
    )
    parser.add_argument(
        "--num-prototypes",
        type=int,
        default=1000,
        help="Number of clustered grasp prototypes to learn from successful demos.",
    )
    parser.add_argument(
        "--max-grasps-per-object",
        type=int,
        default=DEFAULT_MAX_GRASPS_PER_OBJECT,
        help="Maximum successful grasp samples to keep per object during dataset-level training.",
    )
    parser.add_argument(
        "--max-training-grasps",
        type=int,
        default=DEFAULT_MAX_TRAINING_GRASPS,
        help="Global cap on successful grasp samples kept in memory for training.",
    )
    parser.add_argument(
        "--single-object-training",
        action="store_true",
        help="Train only on the selected object (default trains on all dataset objects).",
    )
    parser.add_argument(
        "--load-model",
        action="store_true",
        default=False,
        help="Enable loading an existing saved model before training.",
    )
    parser.add_argument(
        "--save-model",
        action="store_true",
        default=True,
        help="Enable saving the trained model to disk.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    _load_or_train_dataset_policy(
            parser=GraspFactoryParser(root_dir=args.dataset_root, gripper="robotiq"),
            model_path=args.model_path,
            use_saved_model=args.load_model,
            save_model=args.save_model,
            num_prototypes=max(1, args.num_prototypes),
            max_grasps_per_object=max(1, args.max_grasps_per_object),
            max_training_grasps=max(1, args.max_training_grasps),
        )


if __name__ == "__main__":
    main()