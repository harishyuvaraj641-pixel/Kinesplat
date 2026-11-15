"""Camera and quaternion helpers (NumPy + optional Torch)."""

from __future__ import annotations

import numpy as np


def normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        return v
    return v / n


def look_at_c2w(eye: np.ndarray, target: np.ndarray, up: np.ndarray | None = None) -> np.ndarray:
    """Build a 3x3 camera-to-world rotation for an OpenCV camera (x right, y down, z forward)."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.array([0.0, 1.0, 0.0]) if up is None else np.asarray(up, dtype=np.float64)
    forward = normalize(target - eye)
    right = normalize(np.cross(up, forward))
    down = normalize(np.cross(right, forward))
    return np.stack([right, down, forward], axis=1)


def rotation_to_quaternion(R: np.ndarray) -> np.ndarray:
    """Return (w, x, y, z) quaternion for a rotation matrix."""
    m = np.asarray(R, dtype=np.float64)
    trace = m[0, 0] + m[1, 1] + m[2, 2]
    if trace > 0:
        s = np.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(max(1e-12, 1.0 + m[0, 0] - m[1, 1] - m[2, 2])) * 2.0
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(max(1e-12, 1.0 + m[1, 1] - m[0, 0] - m[2, 2])) * 2.0
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(max(1e-12, 1.0 + m[2, 2] - m[0, 0] - m[1, 1])) * 2.0
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    return q / max(1e-12, np.linalg.norm(q))


def project_points(points: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray):
    """Project world points into pixels.  Returns u, v, z."""
    Rcw = R.T
    tcw = -Rcw @ t
    cam = points @ Rcw.T + tcw
    z = cam[:, 2]
    zz = np.where(np.abs(z) < 1e-9, 1e-9, z)
    u = K[0, 0] * cam[:, 0] / zz + K[0, 2]
    v = K[1, 1] * cam[:, 1] / zz + K[1, 2]
    return u, v, z


def unproject_pixels(u: np.ndarray, v: np.ndarray, depth: np.ndarray, K: np.ndarray,
                     R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Unproject pixel centres with per-pixel depth into world space."""
    x = (u - K[0, 2]) / K[0, 0] * depth
    y = (v - K[1, 2]) / K[1, 1] * depth
    cam = np.stack([x, y, depth], axis=-1)
    world = cam @ R.T + t
    return world


def rotmat_to_quat_torch(R):
    """(N,3,3) rotation matrices -> normalised (N,4) w,x,y,z quaternions (torch)."""
    import torch

    m00, m01, m02 = R[:, 0, 0], R[:, 0, 1], R[:, 0, 2]
    m10, m11, m12 = R[:, 1, 0], R[:, 1, 1], R[:, 1, 2]
    m20, m21, m22 = R[:, 2, 0], R[:, 2, 1], R[:, 2, 2]
    trace = m00 + m11 + m22
    q = torch.zeros((R.shape[0], 4), dtype=R.dtype, device=R.device)
    # A branch-free-ish form is fine here because initial rotations are identity.
    s = torch.sqrt(torch.clamp(trace + 1.0, min=1e-8)) * 2.0
    q[:, 0] = 0.25 * s
    q[:, 1] = (m21 - m12) / s
    q[:, 2] = (m02 - m20) / s
    q[:, 3] = (m10 - m01) / s
    q = q / torch.clamp(q.norm(dim=1, keepdim=True), min=1e-8)
    return q


def quat_to_rotmat_torch(q):
    """(N,4) w,x,y,z quaternions -> (N,3,3) rotation matrices."""
    import torch

    q = q / torch.clamp(q.norm(dim=1, keepdim=True), min=1e-8)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    two = 2.0
    R = torch.stack(
        [
            1 - two * (y * y + z * z), two * (x * y - w * z), two * (x * z + w * y),
            two * (x * y + w * z), 1 - two * (x * x + z * z), two * (y * z - w * x),
            two * (x * z - w * y), two * (y * z + w * x), 1 - two * (x * x + y * y),
        ],
        dim=1,
    ).reshape(-1, 3, 3)
    return R


def to_three_coords(points: np.ndarray) -> np.ndarray:
    """Convert OpenCV camera/world axes (y down, z forward) to three.js y-up axes."""
    p = np.asarray(points).copy()
    p[..., 1] *= -1.0
    p[..., 2] *= -1.0
    return p
