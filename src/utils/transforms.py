import numpy as np
from scipy.spatial.transform import Rotation as R


def quat_wxyz_to_xyzw(quaternion_wxyz):
    """
    Converts a MuJoCo-style quaternion [w, x, y, z] to SciPy / robosuite order [x, y, z, w].

    Args:
        quaternion_wxyz (array-like): [w, x, y, z]

    Returns:
        np.ndarray: [x, y, z, w]
    """
    quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=float)
    return quaternion_wxyz[[1, 2, 3, 0]]


def quat_xyzw_to_wxyz(quaternion_xyzw):
    """
    Converts a SciPy / robosuite quaternion [x, y, z, w] to MuJoCo order [w, x, y, z].

    Args:
        quaternion_xyzw (array-like): [x, y, z, w]

    Returns:
        np.ndarray: [w, x, y, z]
    """
    quaternion_xyzw = np.asarray(quaternion_xyzw, dtype=float)
    return quaternion_xyzw[[3, 0, 1, 2]]


def make_transform(rotation=None, translation=None):
    """
    Builds a 4x4 homogeneous transform from a 3x3 rotation matrix and 3D translation.

    Args:
        rotation (np.ndarray | None): 3x3 rotation matrix
        translation (np.ndarray | list | tuple | None): length-3 translation

    Returns:
        np.ndarray: 4x4 homogeneous transform
    """
    T = np.eye(4)

    if rotation is not None:
        rotation = np.asarray(rotation, dtype=float)
        if rotation.shape != (3, 3):
            raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
        T[:3, :3] = rotation

    if translation is not None:
        translation = np.asarray(translation, dtype=float)
        if translation.shape != (3,):
            raise ValueError(
                f"translation must have shape (3,), got {translation.shape}"
            )
        T[:3, 3] = translation

    return T


def pose_to_transform(position, quaternion_wxyz):
    """
    Converts position + quaternion to a 4x4 homogeneous transform.

    Args:
        position (array-like): [x, y, z]
        quaternion_wxyz (array-like): [w, x, y, z]

    Returns:
        np.ndarray: 4x4 homogeneous transform
    """
    position = np.asarray(position, dtype=float)
    quaternion_wxyz = np.asarray(quaternion_wxyz, dtype=float)

    if position.shape != (3,):
        raise ValueError(f"position must have shape (3,), got {position.shape}")
    if quaternion_wxyz.shape != (4,):
        raise ValueError(
            f"quaternion_wxyz must have shape (4,), got {quaternion_wxyz.shape}"
        )

    rotation = R.from_quat(quat_wxyz_to_xyzw(quaternion_wxyz)).as_matrix()
    return make_transform(rotation=rotation, translation=position)


def transform_to_pose(T):
    """
    Splits a 4x4 transform into position and quaternion.

    Args:
        T (np.ndarray): 4x4 homogeneous transform

    Returns:
        tuple[np.ndarray, np.ndarray]:
            position shape (3,)
            quaternion [w, x, y, z] shape (4,)
    """
    T = np.asarray(T, dtype=float)

    position = T[:3, 3].copy()
    rotation = T[:3, :3].copy()
    quaternion_xyzw = R.from_matrix(rotation).as_quat()
    return position, quat_xyzw_to_wxyz(quaternion_xyzw)


def compose_transforms(A, B):
    """
    Returns A @ B for two 4x4 transforms.

    Args:
        A (np.ndarray): 4x4 transform
        B (np.ndarray): 4x4 transform

    Returns:
        np.ndarray: 4x4 transform
    """
    A = np.asarray(A, dtype=float)
    B = np.asarray(B, dtype=float)

    return A @ B


def invert_transform(T):
    """
    Inverts a rigid 4x4 homogeneous transform.

    Args:
        T (np.ndarray): 4x4 transform

    Returns:
        np.ndarray: inverse transform
    """
    T = np.asarray(T, dtype=float)

    R_part = T[:3, :3]
    t_part = T[:3, 3]

    T_inv = np.eye(4)
    T_inv[:3, :3] = R_part.T
    T_inv[:3, 3] = -R_part.T @ t_part
    return T_inv


def offset_transform_along_local_axis(T, axis_index=2, distance=0.1, sign=1.0):
    """
    Offsets a transform translation along one of its own local axes.

    Args:
        T (np.ndarray): 4x4 transform
        axis_index (int): 0=x, 1=y, 2=z local axis
        distance (float): offset magnitude in meters
        sign (float): +1 or -1

    Returns:
        np.ndarray: shifted 4x4 transform
    """
    T = np.asarray(T, dtype=float)

    T_shifted = T.copy()
    local_axis_world = T[:3, axis_index]
    T_shifted[:3, 3] = T_shifted[:3, 3] + sign * distance * local_axis_world
    return T_shifted


def world_grasp_from_object(world_T_object, object_T_grasp):
    """
    Converts an object-frame grasp into a world-frame grasp.

    Args:
        world_T_object (np.ndarray): 4x4 object pose in world
        object_T_grasp (np.ndarray): 4x4 grasp pose in object frame

    Returns:
        np.ndarray: 4x4 grasp pose in the world frame.
    """
    return compose_transforms(world_T_object, object_T_grasp)


def body_pose_to_transform(body_pos, body_xmat):
    """
    Converts MuJoCo body position and flattened rotation matrix into 4x4 transform.

    Args:
        body_pos (array-like): shape (3,)
        body_xmat (array-like): shape (9,) or (3, 3)

    Returns:
        np.ndarray: 4x4 homogeneous transform
    """
    body_pos = np.asarray(body_pos, dtype=float)
    body_xmat = np.asarray(body_xmat, dtype=float)

    if body_pos.shape != (3,):
        raise ValueError(f"body_pos must have shape (3,), got {body_pos.shape}")

    if body_xmat.shape == (9,):
        rotation = body_xmat.reshape(3, 3)
    elif body_xmat.shape == (3, 3):
        rotation = body_xmat
    else:
        raise ValueError(
            f"body_xmat must have shape (9,) or (3, 3), got {body_xmat.shape}"
        )

    return make_transform(rotation=rotation, translation=body_pos)
