"""Small sparse bundle-adjustment helper (SciPy) for the built-in SfM.

COLMAP does this internally; without it incremental PnP drifts quickly.  We
optimise camera extrinsics and 3D points jointly with a Huber loss, using a
sparse Jacobian so laptop CPUs can handle a few hundred points per window.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import coo_matrix

import cv2


def _rodrigues(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv2.Rodrigues(np.asarray(rvec, np.float64))
    return R


def refine(
    poses: dict[int, tuple[np.ndarray, np.ndarray]],
    points: dict[int, np.ndarray],
    point_obs: dict[int, dict[int, int]],
    feats: list[dict],
    K: np.ndarray,
    camera_ids: list[int],
    fixed_id: int,
    max_nfev: int = 15,
    log=print,
) -> None:
    cam_ids = [c for c in camera_ids if c in poses]
    if len(cam_ids) < 2:
        return

    selected: list[tuple[int, int, float, float]] = []
    for pid, obs in point_obs.items():
        if pid not in points:
            continue
        cams = [c for c in obs if c in cam_ids]
        if len(cams) < 2:
            continue
        for c in cams:
            kp = obs[c]
            u, v = feats[c]["pts"][kp]
            selected.append((c, pid, float(u), float(v)))
    if len(selected) < 20:
        return

    pids = sorted({s[1] for s in selected})
    pid_col = {pid: i for i, pid in enumerate(pids)}
    free_cams = [c for c in cam_ids if c != fixed_id]
    cam_col = {c: i for i, c in enumerate(free_cams)}
    n_cam_vars = 6 * len(free_cams)
    n_pt_vars = 3 * len(pids)
    n_vars = n_cam_vars + n_pt_vars

    x0 = np.zeros(n_vars, np.float64)
    for c in free_cams:
        R, t = poses[c]
        Rcw = R.T
        rvec, _ = cv2.Rodrigues(Rcw)
        base = 6 * cam_col[c]
        x0[base:base + 3] = rvec.ravel()
        x0[base + 3:base + 6] = (-Rcw @ t).ravel()
    for pid in pids:
        base = n_cam_vars + 3 * pid_col[pid]
        x0[base:base + 3] = points[pid]

    by_cam: dict[int, list[int]] = {c: [] for c in cam_ids}
    for q, (c, _pid, _u, _v) in enumerate(selected):
        by_cam[c].append(q)

    rows: list[int] = []
    cols: list[int] = []
    for q, (c, pid, _u, _v) in enumerate(selected):
        vcols: list[int] = []
        if c in cam_col:
            base = 6 * cam_col[c]
            vcols.extend(base + j for j in range(6))
        pbase = n_cam_vars + 3 * pid_col[pid]
        vcols.extend(pbase + j for j in range(3))
        for r in (2 * q, 2 * q + 1):
            rows.extend([r] * len(vcols))
            cols.extend(vcols)
    sparsity = coo_matrix(
        (np.ones(len(rows), np.int8), (rows, cols)), shape=(2 * len(selected), n_vars)
    ).tocsr()

    fixed_R = poses[fixed_id][0]
    fixed_Rcw = fixed_R.T
    fixed_tcw = -fixed_Rcw @ poses[fixed_id][1]
    uv = np.array([[s[2], s[3]] for s in selected], np.float64)
    obs_cam = np.array([s[0] for s in selected], np.int32)
    obs_pid = np.array([pid_col[s[1]] for s in selected], np.int32)
    fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]

    def unpack(x):
        cams = {}
        for c in cam_ids:
            if c == fixed_id:
                cams[c] = (fixed_Rcw, fixed_tcw)
            else:
                base = 6 * cam_col[c]
                Rcw = _rodrigues(x[base:base + 3])
                cams[c] = (Rcw, x[base + 3:base + 6])
        pts = x[n_cam_vars:].reshape(-1, 3)
        return cams, pts

    def residual(x):
        cams, pts = unpack(x)
        out = np.zeros(2 * len(selected), np.float64)
        for c in cam_ids:
            idx = np.asarray(by_cam[c], np.int32)
            if idx.size == 0:
                continue
            Rcw, tcw = cams[c]
            X = pts[obs_pid[idx]]
            Xc = X @ Rcw.T + tcw
            z = Xc[:, 2]
            zz = np.where(np.abs(z) < 1e-8, 1e-8, z)
            up = fx * Xc[:, 0] / zz + cx
            vp = fy * Xc[:, 1] / zz + cy
            out[2 * idx] = up - uv[idx, 0]
            out[2 * idx + 1] = vp - uv[idx, 1]
        return out

    try:
        result = least_squares(
            residual, x0, jac_sparsity=sparsity, method="trf",
            loss="huber", f_scale=2.0, max_nfev=max_nfev,
            xtol=1e-5, ftol=1e-5, gtol=1e-5, verbose=0,
        )
    except Exception as exc:
        log(f"    BA skipped: {exc}")
        return

    cams, pts = unpack(result.x)
    for c in free_cams:
        Rcw, tcw = cams[c]
        R = Rcw.T
        t = -R @ tcw
        poses[c] = (R, t)
    for i, pid in enumerate(pids):
        points[pid] = pts[i]
