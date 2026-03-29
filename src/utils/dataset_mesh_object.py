from pathlib import Path
from robosuite.models.objects import MujocoXMLObject

from src.utils.dataset_xml_builder import build_dataset_object_xml


class DatasetMeshObject(MujocoXMLObject):
    """Wrap a dataset mesh as a robosuite-compatible MuJoCo XML object."""

    @staticmethod
    def _format_scale_tag(scale):
        """Create a filesystem-safe scale suffix for generated asset names."""
        scale = tuple(float(v) for v in scale)
        return "scale_" + "_".join(f"{v:.6f}".replace(".", "p").replace("-", "m") for v in scale)

    def __init__(
        self,
        name,
        mesh_path,
        scale=(1.0, 1.0, 1.0),
        generated_xml_dir=None,
        collision_approximation="mesh",
    ):
        """Generate MJCF for a mesh and initialize it as a MuJoCo object.

        Args:
            name (str): Object name used inside the scene.
            mesh_path (str | Path): Source OBJ mesh path.
            scale (tuple[float, float, float]): Per-axis mesh scale.
            generated_xml_dir (str | Path | None): Optional output directory for MJCF files.
            collision_approximation (str): Collision geometry type:
                `mesh`, `convex_hull`, or `box`.
        """
        mesh_path = Path(mesh_path).resolve()

        if generated_xml_dir is None:
            generated_xml_dir = mesh_path.parent / "_generated_mjcf"
        generated_xml_dir = Path(generated_xml_dir).resolve()
        scale_tag = self._format_scale_tag(scale)
        generated_xml_dir = generated_xml_dir / collision_approximation / scale_tag
        generated_xml_dir.mkdir(parents=True, exist_ok=True)

        xml_path = generated_xml_dir / f"{mesh_path.stem}.xml"
        mesh_asset_name = f"{mesh_path.stem}_{collision_approximation}_{scale_tag}_mesh"
        model_name = f"{mesh_path.stem}_{collision_approximation}_{scale_tag}_model"

        # TODO: Cache generated XML by mesh path and scale to avoid rebuilding on every reset.
        build_dataset_object_xml(
            mesh_path=mesh_path,
            xml_out_path=xml_path,
            model_name=model_name,
            mesh_name=mesh_asset_name,
            scale=scale,
            collision_approximation=collision_approximation,
        )

        super().__init__(
            fname=str(xml_path),
            name=name,
            joints="default",
            obj_type="all",
            duplicate_collision_geoms=False,
        )
