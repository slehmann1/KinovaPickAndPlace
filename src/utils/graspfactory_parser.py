from dataclasses import dataclass
from pathlib import Path

import numpy as np

# TODO: update to local folder
DATA_ROOT = Path("D:/graspfactory/robotiq")


@dataclass
class GraspFactoryObj:
    """Container for one GraspFactory object sample and its associated grasps."""

    object_id: str
    mesh_path: Path
    grasps: np.ndarray
    grasp_widths: np.ndarray
    success_indices: np.ndarray
    mesh_uid: str

    @property
    def successful_grasps(self):
        """Return only grasps marked successful by the dataset."""
        return self.grasps[self.success_indices]

    @property
    def successful_widths(self):
        """Return gripper widths corresponding to successful grasps."""
        return self.grasp_widths[self.success_indices]


class GraspFactoryParser:
    """Load meshes and grasp annotations from the local GraspFactory dataset."""

    def __init__(self, root_dir=None, gripper="robotiq"):
        """Index the requested GraspFactory gripper subset.

        Args:
            root_dir (str | Path | None): Dataset root directory override.
            gripper (str): Gripper subset to load, such as ``robotiq``.
        """
        self.root_dir = Path(root_dir) if root_dir is not None else DATA_ROOT

        self.grasp_dir = self.root_dir / gripper / "grasps"
        self.mesh_dir = self.root_dir / gripper / "meshes"

        if not self.grasp_dir.exists():
            raise FileNotFoundError(f"Missing grasp directory: {self.grasp_dir}")
        if not self.mesh_dir.exists():
            raise FileNotFoundError(f"Missing mesh directory: {self.mesh_dir}")

        self.grasp_files = sorted(self.grasp_dir.glob("*.npz"))

    def __len__(self):
        """Return the number of indexed grasp files."""
        return len(self.grasp_files)

    def _resolve_mesh_path(self, object_id, mesh_uid):
        """Resolve the mesh file corresponding to a grasp annotation file.

        Args:
            object_id (str): Basename of the grasp file.
            mesh_uid (str): Mesh UID stored inside the dataset archive.

        Returns:
            Path: Resolved mesh file path.
        """
        cands = [
            self.mesh_dir / f"{object_id}.obj",
            self.mesh_dir / f"{mesh_uid}.obj",
        ]
        for c in cands:
            if c.exists():
                return c
        raise FileNotFoundError(
            f"No mesh found for object_id={object_id}, mesh_uid={mesh_uid}"
        )

    def __getitem__(self, index):
        """Load a dataset sample by index.

        Args:
            index (int): Object item index.

        Returns:
            GraspFactoryObj: Loaded dataset sample.
        """
        grasp_file = self.grasp_files[index]
        object_id = grasp_file.stem

        data = np.load(grasp_file, allow_pickle=True)

        grasps = data["grasps"]
        grasp_widths = data["grasp_widths"]
        success_indices = data["success_indices"]
        mesh_uid = str(data["mesh_uid"])

        mesh_path = self._resolve_mesh_path(object_id, mesh_uid)

        # Build Grasp Factory object from mesh
        return GraspFactoryObj(
            object_id=object_id,
            mesh_path=mesh_path,
            grasps=grasps,
            grasp_widths=grasp_widths,
            success_indices=success_indices,
            mesh_uid=mesh_uid,
        )
