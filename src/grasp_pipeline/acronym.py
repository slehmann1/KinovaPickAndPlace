from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path

import h5py
import numpy as np

from src.grasp_pipeline.models import GraspCandidate, GraspObjectSample
from src.utils.dataset_xml_builder import convert_mesh_frame_translation_to_body_frame


def _decode_string(value):
    """Decode HDF5 byte strings and normalize path-like values."""
    if isinstance(value, bytes):
        return value.decode("utf-8")
    if isinstance(value, np.ndarray) and value.shape == ():
        return _decode_string(value.item())
    if isinstance(value, np.ndarray) and value.size == 1:
        return _decode_string(value.reshape(-1)[0])
    return str(value)


def _load_first_available(container, candidate_paths):
    """Return the first dataset value that exists from a list of HDF5-style paths."""
    for candidate in candidate_paths:
        try:
            return container[candidate][()]
        except Exception:
            continue
    raise KeyError(f"None of the candidate paths were found: {candidate_paths}")


def _load_optional_first_available(container, candidate_paths, default=None):
    """Return the first available dataset value, or a default when none exist."""
    try:
        return _load_first_available(container, candidate_paths)
    except KeyError:
        return default


@dataclass(frozen=True)
class AcronymGraspEntry:
    """Indexed information for one ACRONYM grasp file."""

    grasp_path: Path
    object_id: str
    object_scale: float
    mesh_rel_path: str | None


class AcronymMeshResolver:
    """Resolve ACRONYM object meshes from the dataset metadata and mesh root."""

    COMMON_MESH_DIR_NAMES = (
        "meshes",
        "models",
        "models-OBJ",
        "ShapeNetSem",
        "shapenetsem",
        "shapenet",
        "watertight",
    )

    def __init__(self, grasp_root: str | Path, mesh_root: str | Path | None = None):
        self.grasp_root = Path(grasp_root).resolve()
        self.mesh_root = Path(mesh_root).resolve() if mesh_root is not None else None

    def _candidate_mesh_roots(self):
        """Yield likely mesh roots based on the provided dataset paths."""
        yielded = set()
        if self.mesh_root is not None:
            yielded.add(self.mesh_root)
            yield self.mesh_root

        for parent in (self.grasp_root, *self.grasp_root.parents):
            for name in self.COMMON_MESH_DIR_NAMES:
                candidate = parent / name
                if candidate.exists() and candidate not in yielded:
                    yielded.add(candidate)
                    yield candidate

    def _mesh_candidates_from_rel_path(self, rel_path: str):
        """Generate likely filesystem candidates from a relative mesh path."""
        rel_path = rel_path.replace("\\", "/").strip()
        rel = Path(rel_path)
        basename = rel.name
        stem = rel.stem

        for root in self._candidate_mesh_roots():
            yield root / rel
            yield root / basename
            yield root / stem / "model.obj"
            yield root / stem / f"{stem}.obj"
            yield root / stem / "model_watertight.obj"
            yield root / f"{stem}.obj"
            yield root / f"{stem}_watertight.obj"

    def _mesh_candidates_from_object_id(self, object_id: str):
        """Generate likely filesystem candidates when the HDF5 lacks a mesh path."""
        stem = object_id
        if "_" in object_id:
            maybe_scale = object_id.rsplit("_", 1)[-1]
            try:
                float(maybe_scale)
                stem = object_id.rsplit("_", 1)[0]
            except ValueError:
                stem = object_id

        for root in self._candidate_mesh_roots():
            yield root / f"{stem}.obj"
            yield root / stem / "model.obj"
            yield root / stem / f"{stem}.obj"
            yield root / stem / "model_watertight.obj"
            yield root / f"{stem}_watertight.obj"

    def resolve(self, object_id: str, mesh_rel_path: str | None = None):
        """Resolve a mesh file for one ACRONYM object entry."""
        checked = []

        if mesh_rel_path:
            for candidate in self._mesh_candidates_from_rel_path(mesh_rel_path):
                checked.append(candidate)
                if candidate.exists():
                    return candidate.resolve()

        for candidate in self._mesh_candidates_from_object_id(object_id):
            checked.append(candidate)
            if candidate.exists():
                return candidate.resolve()

        # Fall back to a recursive basename search under the mesh root candidates.
        mesh_names = []
        if mesh_rel_path:
            mesh_names.append(Path(mesh_rel_path).name)
        mesh_names.append(f"{object_id}.obj")
        if "_" in object_id:
            mesh_names.append(f"{object_id.rsplit('_', 1)[0]}.obj")

        for root in self._candidate_mesh_roots():
            for mesh_name in mesh_names:
                matches = list(root.rglob(mesh_name))
                if matches:
                    return matches[0].resolve()

        checked_str = "\n".join(str(path) for path in checked[:20])
        raise FileNotFoundError(
            "Unable to resolve an ACRONYM mesh path.\n"
            f"object_id={object_id}\n"
            f"mesh_rel_path={mesh_rel_path}\n"
            f"First checked candidates:\n{checked_str}"
        )


class AcronymParser:
    """Load ACRONYM grasp files and convert them into benchmark-ready object samples."""

    TRANSFORM_PATHS = (
        "grasps/transforms",
        "transforms",
        "grasps",
    )
    QUALITY_PATHS = (
        "grasps/qualities/flex/object_in_gripper",
        "grasps/qualities/flex/success",
        "grasps/qualities/object_in_gripper",
        "qualities/flex/object_in_gripper",
        "qualities/object_in_gripper",
        "success",
    )
    WIDTH_PATHS = (
        "grasps/widths",
        "grasp_widths",
        "widths",
    )
    MESH_PATHS = (
        "object/file",
        "object/path",
        "object/mesh/file",
        "mesh/file",
        "mesh/path",
        "object_file",
    )
    SCALE_PATHS = (
        "object/scale",
        "mesh/scale",
        "object_scale",
        "scale",
    )

    def __init__(self, root_dir: str | Path, mesh_root: str | Path | None = None):
        self.root_dir = Path(root_dir).resolve()
        self.grasp_dir = self._resolve_grasp_dir(self.root_dir)
        self.mesh_resolver = AcronymMeshResolver(self.root_dir, mesh_root=mesh_root)
        self.grasp_files = self._index_grasp_files(self.grasp_dir)

    @staticmethod
    def _resolve_grasp_dir(root_dir: Path):
        """Find the directory containing the ACRONYM grasp files."""
        candidates = (
            root_dir / "grasps",
            root_dir / "acronym" / "grasps",
            root_dir,
        )
        for candidate in candidates:
            if candidate.exists() and any(candidate.rglob("*.h5")):
                return candidate
        raise FileNotFoundError(f"Could not find any ACRONYM grasp files under {root_dir}")

    @staticmethod
    def _index_grasp_files(grasp_dir: Path):
        """Return a stable list of HDF5 grasp files."""
        grasp_files = sorted(grasp_dir.rglob("*.h5"))
        if not grasp_files:
            raise FileNotFoundError(f"No ACRONYM .h5 grasp files found under {grasp_dir}")
        return grasp_files

    def __len__(self):
        return len(self.grasp_files)

    def _read_entry_metadata(self, grasp_path: Path):
        """Load identifying mesh metadata from one grasp file."""
        with h5py.File(grasp_path, "r") as handle:
            mesh_rel_path = _load_optional_first_available(handle, self.MESH_PATHS)
            object_scale = _load_optional_first_available(handle, self.SCALE_PATHS, default=1.0)

        object_id = grasp_path.stem
        return AcronymGraspEntry(
            grasp_path=grasp_path,
            object_id=object_id,
            object_scale=float(np.asarray(object_scale).reshape(-1)[0]),
            mesh_rel_path=None if mesh_rel_path is None else _decode_string(mesh_rel_path),
        )

    def _load_quality_mask(self, handle, grasp_count: int):
        """Return a boolean mask of successful grasps, or keep all when unavailable."""
        quality_values = _load_optional_first_available(handle, self.QUALITY_PATHS)
        if quality_values is None:
            return np.ones(grasp_count, dtype=bool)

        quality_values = np.asarray(quality_values).reshape(-1)
        if quality_values.shape[0] != grasp_count:
            raise ValueError(
                "ACRONYM quality array length does not match transform count: "
                f"{quality_values.shape[0]} vs {grasp_count}"
            )
        return quality_values.astype(float) > 0.0

    def _load_widths(self, handle, grasp_count: int):
        """Return optional grasp widths in meters, or a list of None values."""
        widths = _load_optional_first_available(handle, self.WIDTH_PATHS)
        if widths is None:
            return [None] * grasp_count

        widths = np.asarray(widths, dtype=float).reshape(-1)
        if widths.shape[0] != grasp_count:
            raise ValueError(
                "ACRONYM width array length does not match transform count: "
                f"{widths.shape[0]} vs {grasp_count}"
            )
        return [float(width) for width in widths]

    def get_object_sample(self, object_index: int, max_grasps: int | None = None):
        """Load one ACRONYM object and convert its grasps into body-frame candidates."""
        entry = self._read_entry_metadata(self.grasp_files[object_index])
        mesh_path = self.mesh_resolver.resolve(
            object_id=entry.object_id,
            mesh_rel_path=entry.mesh_rel_path,
        )
        object_scale = (entry.object_scale, entry.object_scale, entry.object_scale)

        with h5py.File(entry.grasp_path, "r") as handle:
            raw_transforms = np.asarray(_load_first_available(handle, self.TRANSFORM_PATHS), dtype=float)
            if raw_transforms.ndim != 3 or raw_transforms.shape[1:] != (4, 4):
                raise ValueError(
                    f"Unexpected ACRONYM transform shape for {entry.grasp_path}: "
                    f"{raw_transforms.shape}"
                )

            widths = self._load_widths(handle, raw_transforms.shape[0])
            success_mask = self._load_quality_mask(handle, raw_transforms.shape[0])

        success_indices = np.flatnonzero(success_mask)
        if max_grasps is not None:
            success_indices = success_indices[:max_grasps]
        if len(success_indices) == 0:
            raise ValueError(f"No successful ACRONYM grasps found for {entry.object_id}")

        candidates = []
        for local_idx, raw_idx in enumerate(success_indices):
            object_T_grasp = raw_transforms[raw_idx].copy()
            object_T_grasp[:3, 3] = convert_mesh_frame_translation_to_body_frame(
                translation=object_T_grasp[:3, 3],
                mesh_path=mesh_path,
                scale=object_scale,
            )

            width_value = widths[raw_idx]
            candidates.append(
                GraspCandidate(
                    grasp_id=f"grasp_{local_idx:04d}",
                    object_T_grasp=object_T_grasp,
                    grasp_width_m=width_value,
                    metadata={
                        "source": "acronym",
                        "raw_grasp_index": int(raw_idx),
                        "grasp_file": str(entry.grasp_path),
                    },
                )
            )

        return GraspObjectSample(
            object_id=entry.object_id,
            mesh_path=mesh_path,
            candidates=tuple(candidates),
            metadata={
                "source": "acronym",
                "grasp_path": str(entry.grasp_path),
                "mesh_path": str(mesh_path),
                "mesh_rel_path": entry.mesh_rel_path,
                "object_scale": object_scale,
            },
        )

    def inspect_object(self, object_index: int):
        """Return a JSON-serializable description of one ACRONYM object entry."""
        entry = self._read_entry_metadata(self.grasp_files[object_index])
        mesh_path = self.mesh_resolver.resolve(
            object_id=entry.object_id,
            mesh_rel_path=entry.mesh_rel_path,
        )

        with h5py.File(entry.grasp_path, "r") as handle:
            raw_transforms = np.asarray(_load_first_available(handle, self.TRANSFORM_PATHS), dtype=float)
            success_mask = self._load_quality_mask(handle, raw_transforms.shape[0])

        return {
            "object_id": entry.object_id,
            "grasp_path": str(entry.grasp_path),
            "mesh_rel_path": entry.mesh_rel_path,
            "mesh_path": str(mesh_path),
            "object_scale": entry.object_scale,
            "grasp_count": int(raw_transforms.shape[0]),
            "successful_grasp_count": int(np.count_nonzero(success_mask)),
        }


def dump_acronym_inspection(parser: AcronymParser, object_index: int):
    """Return a pretty-printed description for one indexed ACRONYM object."""
    return json.dumps(parser.inspect_object(object_index), indent=2)
