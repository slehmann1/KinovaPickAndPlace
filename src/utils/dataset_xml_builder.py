import os
from pathlib import Path

import numpy as np
import trimesh


def _fmt3(vec):
    """Format a length-3 vector for inline MJCF attributes."""
    return f"{vec[0]:.8f} {vec[1]:.8f} {vec[2]:.8f}"


def get_mesh_bounds(mesh_path):
    """Return the raw mesh bounds in the mesh's local coordinates.

    Args:
        mesh_path (str | Path): Mesh file to inspect.

    Returns:
        tuple[np.ndarray, np.ndarray]: Minimum and maximum mesh corners.
    """
    mesh = trimesh.load_mesh(Path(mesh_path), process=False)
    if len(mesh.vertices) == 0:
        raise ValueError(f"Mesh is empty or invalid: {mesh_path}")

    vertices = np.asarray(mesh.vertices)
    min_corner = vertices.min(axis=0)
    max_corner = vertices.max(axis=0)
    return min_corner, max_corner


def build_dataset_object_xml(
    mesh_path,
    xml_out_path,
    model_name="dataset_object",
    mesh_name="dataset_mesh",
    scale=(0.001, 0.001, 0.001), # TODO: mm scaling by default, need to verify
    density=300.0,
    friction=(1.0, 0.3, 0.1),
    rgba=(0.8, 0.2, 0.2, 1.0),
    should_log_info=False,
):
    """Build a robosuite-compatible MJCF XML file for a dataset OBJ mesh.

    The mesh geom is repositioned so that x and y are centered while the
    bottom of the mesh sits at local `z = 0`.

    Args:
        mesh_path (str | Path): Source OBJ mesh path.
        xml_out_path (str | Path): Destination path for the generated MJCF file.
        model_name (str): MJCF model name.
        mesh_name (str): MJCF mesh asset name.
        scale (tuple[float, float, float]): Per-axis scale applied to the mesh.
        density (float): Density value written to the collision geom.
        friction (tuple[float, float, float]): MuJoCo friction tuple.
        rgba (tuple[float, float, float, float]): Display color used by the geom.

    Returns:
        Path: Output XML path.
    """
    mesh_path = Path(mesh_path).resolve()
    xml_out_path = Path(xml_out_path).resolve()
    xml_out_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load_mesh(mesh_path, process=False)
    vertices = np.asarray(mesh.vertices)
    if len(vertices) == 0:
        raise ValueError(f"Mesh has no vertices: {mesh_path}")

    # TODO: verify object geometry center and offset
    min_corner = vertices.min(axis=0)
    max_corner = vertices.max(axis=0)
    extent = max_corner - min_corner
    scale = np.asarray(scale, dtype=float)

    if should_log_info:
        print(f"Building XML for mesh: {mesh_path.name}")
        print(f"Mesh bounds min: {min_corner}")
        print(f"Mesh bounds max: {max_corner}")
        print(f"Mesh extent: {extent}")
        print(f"Mesh scale: {scale}")

    # Raw-mesh bottom-center
    center_xy_bottom = np.array([
        0.5 * (min_corner[0] + max_corner[0]),
        0.5 * (min_corner[1] + max_corner[1]),
        min_corner[2],
    ])

    # Convert offset into scaled MuJoCo units
    # TODO: verify
    geom_pos = -(center_xy_bottom * scale)

    # Scaled extent for local sites
    scaled_extent = extent * scale

    # Horizontal radius should also be computed in scaled units
    centered_xy = vertices[:, :2] - center_xy_bottom[:2]
    scaled_centered_xy = centered_xy * scale[:2]
    horizontal_radius = float(
        np.max(np.linalg.norm(scaled_centered_xy, axis=1))
    )

    local_bottom_site = [0.0, 0.0, 0.0]
    local_top_site = [0.0, 0.0, float(scaled_extent[2])]
    local_horizontal_radius_site = [horizontal_radius, 0.0, 0.0]

    rel_mesh_path = os.path.relpath(mesh_path, start=xml_out_path.parent).replace(
        "\\", "/"
    )
    
    if should_log_info:
        print(f"Computed geom_pos: {geom_pos}")
        print(f"Scaled extent: {scaled_extent}")
        print(f"Horizontal radius: {horizontal_radius}")

    xml_text = f"""<mujoco model="{model_name}">
  <asset>
    <mesh file="{rel_mesh_path}" name="{mesh_name}" scale="{_fmt3(scale)}"/>
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <geom
          name="object_geom"
          type="mesh"
          mesh="{mesh_name}"
          pos="{_fmt3(geom_pos)}"
          density="{density}"
          friction="{friction[0]} {friction[1]} {friction[2]}"
          rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"
          solimp="0.998 0.998 0.001"
          solref="0.001 1"
          condim="4"
          group="0"
        />
      </body>
      <site name="bottom_site" rgba="0 0 0 0" size="0.005" pos="{_fmt3(local_bottom_site)}"/>
      <site name="top_site" rgba="0 0 0 0" size="0.005" pos="{_fmt3(local_top_site)}"/>
      <site name="horizontal_radius_site" rgba="0 0 0 0" size="0.005" pos="{_fmt3(local_horizontal_radius_site)}"/>
    </body>
  </worldbody>
</mujoco>
"""
    xml_out_path.write_text(xml_text, encoding="utf-8")
    return xml_out_path
