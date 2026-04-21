from pathlib import Path
from robosuite.models.objects import MujocoXMLObject

from src.utils.dataset_xml_builder import build_dataset_object_xml


class DatasetMeshObject(MujocoXMLObject):
    """Wrap a dataset mesh as a robosuite-compatible MuJoCo XML object."""

    def __init__(
        self,
        name,
        mesh_path,
        scale=(1.0, 1.0, 1.0),
        generated_xml_dir=None,
    ):
        """Generate MJCF for a mesh and initialize it as a MuJoCo object.

        Args:
            name (str): Object name used inside the scene.
            mesh_path (str | Path): Source OBJ mesh path.
            scale (tuple[float, float, float]): Per-axis mesh scale.
            generated_xml_dir (str | Path | None): Optional output directory for MJCF files.
        """
        mesh_path = Path(mesh_path).resolve()

        if generated_xml_dir is None:
            generated_xml_dir = mesh_path.parent / "_generated_mjcf"
        generated_xml_dir = Path(generated_xml_dir).resolve()
        generated_xml_dir.mkdir(parents=True, exist_ok=True)

        xml_path = generated_xml_dir / f"{mesh_path.stem}.xml"

        # TODO: Cache generated XML by mesh path and scale to avoid rebuilding on every reset.
        build_dataset_object_xml(
            mesh_path=mesh_path,
            xml_out_path=xml_path,
            model_name=f"{mesh_path.stem}_model",
            mesh_name=f"{mesh_path.stem}_mesh",
            scale=scale,
        )

        super().__init__(
            fname=str(xml_path),
            name=name,
            joints="default",
            obj_type="all",
            duplicate_collision_geoms=True,
        )
