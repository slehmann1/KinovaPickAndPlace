from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

from src.grasp_pipeline.acronym import AcronymParser
from src.grasp_pipeline.models import GraspCandidate, GraspObjectSample
from src.utils.dataset_xml_builder import convert_mesh_frame_translation_to_body_frame
from src.utils.graspfactory_parser import GraspFactoryParser


class GraspSource(ABC):
    """Abstract adapter for any dataset that provides known grasp descriptions."""

    @abstractmethod
    def get_object_sample(self, object_index: int, max_grasps: int | None = None) -> GraspObjectSample:
        """Return one object sample and a subset of candidate grasps."""


class GraspFactorySource(GraspSource):
    """Adapter that turns GraspFactory samples into executable body-frame grasps."""

    def __init__(
        self,
        root_dir: str | Path | None = None,
        gripper: str = "robotiq_2f85",
        object_scale: tuple[float, float, float] = (0.001, 0.001, 0.001),
    ):
        self.parser = GraspFactoryParser(root_dir=root_dir, gripper=gripper)
        self.object_scale = tuple(float(v) for v in object_scale)

    def _convert_candidate(
        self,
        grasp_index: int,
        raw_grasp_index: int,
        object_T_grasp_raw: np.ndarray,
        grasp_width: float,
        mesh_path: Path,
    ) -> GraspCandidate:
        """Convert one raw GraspFactory grasp into the MuJoCo object body frame."""
        object_T_grasp = np.asarray(object_T_grasp_raw, dtype=float).copy()
        object_T_grasp[:3, 3] = convert_mesh_frame_translation_to_body_frame(
            translation=object_T_grasp_raw[:3, 3],
            mesh_path=mesh_path,
            scale=self.object_scale,
        )
        grasp_width_m = float(grasp_width) * float(self.object_scale[0])
        return GraspCandidate(
            grasp_id=f"grasp_{grasp_index:04d}",
            object_T_grasp=object_T_grasp,
            grasp_width_m=grasp_width_m,
            metadata={"source": "graspfactory", "raw_grasp_index": int(raw_grasp_index)},
        )

    def get_object_sample(self, object_index: int, max_grasps: int | None = None) -> GraspObjectSample:
        """Load one object and convert its successful grasps into executable candidates."""
        sample = self.parser[object_index]
        successful_indices = sample.success_indices
        if len(successful_indices) == 0:
            raise ValueError(f"No successful grasps found for object {sample.object_id}")

        if max_grasps is not None:
            successful_indices = successful_indices[:max_grasps]

        candidates = []
        for local_id, raw_idx in enumerate(successful_indices):
            object_T_grasp_raw = sample.grasps[raw_idx]
            grasp_width = sample.grasp_widths[raw_idx]
            candidates.append(
                self._convert_candidate(
                    grasp_index=local_id,
                    raw_grasp_index=int(raw_idx),
                    object_T_grasp_raw=object_T_grasp_raw,
                    grasp_width=grasp_width,
                    mesh_path=sample.mesh_path,
                )
            )

        return GraspObjectSample(
            object_id=sample.object_id,
            mesh_path=sample.mesh_path,
            candidates=tuple(candidates),
            metadata={
                "source": "graspfactory",
                "mesh_uid": sample.mesh_uid,
                "mesh_path": str(sample.mesh_path),
                "object_scale": self.object_scale,
            },
        )


class AcronymSource(GraspSource):
    """Adapter that loads ACRONYM grasp files and resolves their meshes."""

    def __init__(self, root_dir: str | Path, mesh_root: str | Path | None = None):
        self.root_dir = Path(root_dir)
        self.parser = AcronymParser(root_dir=root_dir, mesh_root=mesh_root)

    def get_object_sample(self, object_index: int, max_grasps: int | None = None) -> GraspObjectSample:
        return self.parser.get_object_sample(object_index=object_index, max_grasps=max_grasps)
