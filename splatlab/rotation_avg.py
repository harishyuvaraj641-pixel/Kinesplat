"""Global rotation averaging for the built-in SfM.

Incremental PnP drifts because every new camera is fitted to a local map.  A
lightweight rotation-averaging pass over verified two-view relative rotations
removes most of that drift before the cameras are registered.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(a.T @ b) - 1.0) * 0.5, -1.0, 1.0))))


def average(
    n_frames: int,
    sequential: list[tuple[int, int, np.ndarray, int]],
    extra: list[tuple[int, int, np.ndarray, int]],
    consistency_deg: float = 30.0,
    log=print,
) -> dict[int, np.ndarray]:
    """Estimate absolute camera-to-world rotations in the first camera's frame.

    ``sequential`` must contain the ``(i, i+1, R_ij, inliers)`` chain, where
    ``R_ij`` maps camera-i coordinates to camera-j coordinates.  ``extra`` are
    longer-range pairs; they are only accepted when they roughly agree with the
    chain, which rejects the 180-degree flips that repetitive scenes produce.
    """
    if n_frames < 2 or len(sequential) < n_frames - 1:
        raise ValueError("rotation averaging needs a complete sequential chain")

    seq = {(i, j): (R, n) for i, j, R, n in sequential}
    rots: list[np.ndarray] = [np.eye(3)]
    for i in range(1, n_frames):
        R, _n = seq[(i - 1, i)]
        rots.append(rots[-1] @ R.T)

    edges = list(sequential)
    accepted = 0
    for i, j, R, n in extra:
        pred = rots[j].T @ rots[i]
        if _angle_deg(R, pred) < consistency_deg:
            edges.append((i, j, R, n))
            accepted += 1
    log(f"Rotation graph: {len(edges)} edges ({accepted} long-range accepted).")

    opt_idx = list(range(1, n_frames))
    pos = {i: k for k, i in enumerate(opt_idx)}
    x0 = np.concatenate([Rotation.from_matrix(rots[i]).as_rotvec() for i in opt_idx])

    def build(x: np.ndarray) -> list[np.ndarray]:
        cur = [r.copy() for r in rots]
        for i in opt_idx:
            cur[i] = Rotation.from_rotvec(x[3 * pos[i]:3 * pos[i] + 3]).as_matrix()
        return cur

    def residuals(x: np.ndarray) -> np.ndarray:
        cur = build(x)
        out: list[float] = []
        for i, j, R_ij, n in edges:
            err = R_ij.T @ (cur[j].T @ cur[i])
            out.extend(np.sqrt(min(n, 250) / 250.0) * Rotation.from_matrix(err).as_rotvec())
        return np.asarray(out, np.float64)

    result = least_squares(
        residuals, x0, loss="huber", f_scale=np.deg2rad(0.7),
        max_nfev=250, xtol=1e-8, ftol=1e-8, gtol=1e-8,
    )

    # Drop edges that still disagree strongly, then polish once more.
    cur = build(result.x)
    kept = []
    for i, j, R_ij, n in edges:
        err = R_ij.T @ (cur[j].T @ cur[i])
        if np.degrees(Rotation.from_matrix(err).magnitude()) < 4.0:
            kept.append((i, j, R_ij, n))
    if 10 < len(kept) < len(edges):
        edges = kept

        def residuals2(x: np.ndarray) -> np.ndarray:
            c = build(x)
            out: list[float] = []
            for i, j, R_ij, n in edges:
                err = R_ij.T @ (c[j].T @ c[i])
                out.extend(np.sqrt(min(n, 250) / 250.0) * Rotation.from_matrix(err).as_rotvec())
            return np.asarray(out, np.float64)

        result = least_squares(
            residuals2, result.x, loss="huber", f_scale=np.deg2rad(0.5),
            max_nfev=150, xtol=1e-8, ftol=1e-8, gtol=1e-8,
        )

    final = build(result.x)
    return {i: final[i] for i in range(n_frames)}
