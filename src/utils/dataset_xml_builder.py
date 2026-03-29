from pathlib import Path
import os
import numpy as np
import trimesh


def _fmt3(vec):
    """Format a length-3 vector for inline MJCF attributes."""
    return f"{vec[0]:.8f} {vec[1]:.8f} {vec[2]:.8f}"


def _build_mesh_geom_xml(
    name,
    mesh_name,
    pos,
    friction,
    rgba,
    group,
    contype=None,
    conaffinity=None,
    shellinertia=None,
):
    """Return a mesh geom XML snippet with consistent formatting."""
    extra_attrs = []
    if contype is not None:
        extra_attrs.append(f'contype="{contype}"')
    if conaffinity is not None:
        extra_attrs.append(f'conaffinity="{conaffinity}"')
    if shellinertia is not None:
        extra_attrs.append(f'shellinertia="{str(shellinertia).lower()}"')

    extra_attr_text = ""
    if extra_attrs:
        extra_attr_text = "\n          " + "\n          ".join(extra_attrs)

    return f"""        <geom
          name="{name}"
          type="mesh"
          mesh="{mesh_name}"
          pos="{_fmt3(pos)}"
          friction="{friction[0]} {friction[1]} {friction[2]}"
          rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"
          solimp="0.998 0.998 0.001"
          solref="0.001 1"
          condim="4"
          group="{group}"{extra_attr_text}
        />"""


def get_mesh_bounds(mesh_path):
    """Return the raw mesh bounds in the mesh's local coordinates.

    Args:
        mesh_path (str | Path): Mesh file to inspect.

    Returns:
        tuple[np.ndarray, np.ndarray]: Minimum and maximum mesh corners.
    """
    mesh = trimesh.load_mesh(Path(mesh_path), process=False)
    if mesh.is_empty or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh is empty or invalid: {mesh_path}")

    vertices = np.asarray(mesh.vertices)
    min_corner = vertices.min(axis=0)
    max_corner = vertices.max(axis=0)
    return min_corner, max_corner


def get_mesh_body_frame_metadata(mesh_path, scale=(1.0, 1.0, 1.0)):
    """Compute the generated object's local-frame metadata for a source mesh.

    The generated MJCF shifts the raw mesh so that the MuJoCo body frame sits at the
    bottom-center of the object's footprint. This helper centralizes that transform so
    the XML builder, placement helpers, and dataset grasp conversion all use the same
    convention.

    Args:
        mesh_path (str | Path): Source OBJ mesh path.
        scale (tuple[float, float, float]): Per-axis mesh scale used in MJCF.

    Returns:
        dict[str, np.ndarray | float]: Precomputed local-frame metadata.
    """
    mesh = trimesh.load_mesh(Path(mesh_path), process=False)
    if mesh.is_empty or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh is empty or invalid: {mesh_path}")

    vertices = np.asarray(mesh.vertices, dtype=float)
    scale = np.asarray(scale, dtype=float)

    min_corner = vertices.min(axis=0)
    max_corner = vertices.max(axis=0)
    extent = max_corner - min_corner

    # Use the bottom-center of the raw mesh as the MuJoCo body-frame origin.
    center_xy_bottom = np.array(
        [
            0.5 * (min_corner[0] + max_corner[0]),
            0.5 * (min_corner[1] + max_corner[1]),
            min_corner[2],
        ],
        dtype=float,
    )

    # Convert the raw mesh into scaled MuJoCo coordinates.
    geom_pos = -(center_xy_bottom * scale)
    scaled_extent = extent * scale

    centered_xy = vertices[:, :2] - center_xy_bottom[:2]
    scaled_centered_xy = centered_xy * scale[:2]
    horizontal_radius = float(np.max(np.linalg.norm(scaled_centered_xy, axis=1)))

    return {
        "vertices": vertices,
        "scale": scale,
        "min_corner": min_corner,
        "max_corner": max_corner,
        "extent": extent,
        "center_xy_bottom": center_xy_bottom,
        "geom_pos": geom_pos,
        "scaled_extent": scaled_extent,
        "horizontal_radius": horizontal_radius,
    }


def get_mesh_body_frame_vertices(mesh_path, scale=(1.0, 1.0, 1.0)):
    """Return mesh vertices expressed in the object body frame used by the generated MJCF.

    The generated MJCF shifts the raw mesh so that the body's local origin sits at the
    bottom-center of the mesh footprint. This helper reproduces that same transform.

    Args:
        mesh_path (str | Path): Source OBJ mesh path.
        scale (tuple[float, float, float]): Per-axis mesh scale used in MJCF.

    Returns:
        np.ndarray: Mesh vertices in the generated object's local body frame.
    """
    metadata = get_mesh_body_frame_metadata(mesh_path, scale=scale)
    vertices = metadata["vertices"]
    scale = metadata["scale"]
    geom_pos = metadata["geom_pos"]

    scaled_vertices = vertices * scale
    return scaled_vertices + geom_pos


def convert_mesh_frame_translation_to_body_frame(
    translation,
    mesh_path,
    scale=(1.0, 1.0, 1.0),
):
    """Convert a raw mesh-frame translation into the generated MuJoCo body frame.

    This is used for dataset grasp poses, whose translations are expressed in the
    original mesh coordinates rather than the shifted bottom-center body frame used
    by the generated MJCF object.

    Args:
        translation (array-like): Raw mesh-frame translation.
        mesh_path (str | Path): Source mesh path.
        scale (tuple[float, float, float]): Per-axis mesh scale used in MJCF.

    Returns:
        np.ndarray: Translation expressed in the MuJoCo object body frame.
    """
    translation = np.asarray(translation, dtype=float)
    if translation.shape != (3,):
        raise ValueError(f"translation must have shape (3,), got {translation.shape}")

    metadata = get_mesh_body_frame_metadata(mesh_path, scale=scale)
    return translation * metadata["scale"] + metadata["geom_pos"]


def build_convex_hull_mesh_file(mesh_path, hull_out_path):
    """Write a convex-hull OBJ derived from the source mesh."""
    mesh_path = Path(mesh_path).resolve()
    hull_out_path = Path(hull_out_path).resolve()
    hull_out_path.parent.mkdir(parents=True, exist_ok=True)

    mesh = trimesh.load_mesh(mesh_path, process=False)
    if mesh.is_empty or len(mesh.vertices) == 0:
        raise ValueError(f"Mesh is empty or invalid: {mesh_path}")

    convex_hull = mesh.convex_hull
    if convex_hull.is_empty or len(convex_hull.vertices) == 0:
        raise ValueError(f"Convex hull is empty or invalid: {mesh_path}")

    convex_hull.export(hull_out_path)
    return hull_out_path


def build_dataset_object_xml(
    mesh_path,
    xml_out_path,
    model_name="dataset_object",
    mesh_name="dataset_mesh",
    scale=(0.001, 0.001, 0.001),  # TODO: mm scaling by default, need to verify
    density=300.0,
    friction=(1.0, 0.3, 0.1),
    rgba=(0.8, 0.2, 0.2, 1.0),
    collision_approximation="mesh",
):
    """Build a robosuite-compatible MJCF XML file for a dataset OBJ mesh.

    The mesh geom is repositioned so that x and y are centered while the
    bottom of the mesh sits at local `z = 0`.

    IMPORTANT:
    The geom offset must be scaled into MuJoCo world units, because mesh scale
    does not automatically scale geom positions.

    Args:
        mesh_path (str | Path): Source OBJ mesh path.
        xml_out_path (str | Path): Destination path for the generated MJCF file.
        model_name (str): MJCF model name.
        mesh_name (str): MJCF mesh asset name.
        scale (tuple[float, float, float]): Per-axis scale applied to the mesh.
        density (float): Density value written to the collision geom.
        friction (tuple[float, float, float]): MuJoCo friction tuple.
        rgba (tuple[float, float, float, float]): Display color used by the geom.
        collision_approximation (str): Collision mode: `mesh`, `convex_hull`, or `box`.

    Returns:
        Path: Output XML path.
    """
    mesh_path = Path(mesh_path).resolve()
    xml_out_path = Path(xml_out_path).resolve()
    xml_out_path.parent.mkdir(parents=True, exist_ok=True)

    metadata = get_mesh_body_frame_metadata(mesh_path, scale=scale)
    min_corner = metadata["min_corner"]
    max_corner = metadata["max_corner"]
    extent = metadata["extent"]
    scale = metadata["scale"]
    geom_pos = metadata["geom_pos"]
    scaled_extent = metadata["scaled_extent"]
    horizontal_radius = metadata["horizontal_radius"]

    print(f"Building XML for mesh: {mesh_path.name}")
    print(f"Mesh bounds min: {min_corner}")
    print(f"Mesh bounds max: {max_corner}")
    print(f"Mesh extent: {extent}")
    print(f"Mesh scale: {scale}")

    local_bottom_site = [0.0, 0.0, 0.0]
    local_top_site = [0.0, 0.0, float(scaled_extent[2])]
    local_horizontal_radius_site = [horizontal_radius, 0.0, 0.0]
    proxy_box_center = [0.0, 0.0, float(scaled_extent[2] / 2.0)]
    proxy_box_half_size = np.maximum(scaled_extent / 2.0, 1e-6)
    proxy_box_full_size = np.maximum(scaled_extent, 2e-6)

    # Use a conservative box-based inertial model so thin or non-watertight meshes do
    # not destabilize the body when mesh collision is enabled.
    proxy_volume = float(np.prod(proxy_box_full_size))
    body_mass = max(density * proxy_volume, 0.02)
    hx, hy, hz = proxy_box_half_size
    body_inertia = np.array(
        [
            (body_mass / 3.0) * (hy**2 + hz**2),
            (body_mass / 3.0) * (hx**2 + hz**2),
            (body_mass / 3.0) * (hx**2 + hy**2),
        ],
        dtype=float,
    )

    rel_mesh_path = os.path.relpath(mesh_path, start=xml_out_path.parent).replace(
        "\\", "/"
    )

    print(f"Computed geom_pos: {geom_pos}")
    print(f"Scaled extent: {scaled_extent}")
    print(f"Horizontal radius: {horizontal_radius}")

    if collision_approximation not in {"box", "mesh", "convex_hull"}:
        raise ValueError(
            "collision_approximation must be one of 'mesh', 'convex_hull', or 'box', "
            f"got {collision_approximation!r}"
        )

    if collision_approximation == "box":
        collision_geom_xml = f"""        <geom
          name="object_collision"
          type="box"
          pos="{_fmt3(proxy_box_center)}"
          size="{_fmt3(proxy_box_half_size)}"
          friction="{friction[0]} {friction[1]} {friction[2]}"
          rgba="0 0 0 0"
          solimp="0.998 0.998 0.001"
          solref="0.001 1"
          condim="4"
          group="0"
        />"""
        extra_asset_xml = ""
    elif collision_approximation == "convex_hull":
        hull_mesh_name = f"{mesh_name}_convex_hull"
        hull_out_path = xml_out_path.parent / f"{mesh_path.stem}_convex_hull.obj"
        build_convex_hull_mesh_file(mesh_path=mesh_path, hull_out_path=hull_out_path)
        rel_hull_mesh_path = os.path.relpath(hull_out_path, start=xml_out_path.parent).replace(
            "\\", "/"
        )
        extra_asset_xml = (
            f'\n    <mesh file="{rel_hull_mesh_path}" name="{hull_mesh_name}" scale="{_fmt3(scale)}"/>'
        )
        collision_geom_xml = _build_mesh_geom_xml(
            name="object_collision",
            mesh_name=hull_mesh_name,
            pos=geom_pos,
            friction=friction,
            rgba=(0.0, 0.0, 0.0, 0.0),
            group="0",
        )
    else:
        extra_asset_xml = ""
        collision_geom_xml = _build_mesh_geom_xml(
            name="object_collision",
            mesh_name=mesh_name,
            pos=geom_pos,
            friction=friction,
            rgba=(0.0, 0.0, 0.0, 0.0),
            group="0",
        )

    # Keep the visual mesh declaration explicit so rendered geometry remains easy
    # to inspect independently of whichever collision mode is active.
    visual_geom_xml = f"""        <geom
          name="object_visual"
          type="mesh"
          mesh="{mesh_name}"
          pos="{_fmt3(geom_pos)}"
          rgba="{rgba[0]} {rgba[1]} {rgba[2]} {rgba[3]}"
          contype="0"
          conaffinity="0"
          group="1"
        />"""

    xml_text = f"""<mujoco model="{model_name}">
  <asset>
    <mesh file="{rel_mesh_path}" name="{mesh_name}" scale="{_fmt3(scale)}"/>
{extra_asset_xml}
  </asset>
  <worldbody>
    <body>
      <body name="object">
        <inertial
          pos="{_fmt3(proxy_box_center)}"
          mass="{body_mass:.8f}"
          diaginertia="{_fmt3(body_inertia)}"
        />
{collision_geom_xml}
{visual_geom_xml}
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
