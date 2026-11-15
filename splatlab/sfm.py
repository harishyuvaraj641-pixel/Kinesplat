"""Lightweight sequential Structure-from-Motion fallback (OpenCV, CPU).

This is deliberately conservative: it tracks features, solves PnP against an
incremental 3D map and triangulates new points.  It is not a COLMAP
replacement, but it makes the application fully usable offline and without
installing external binaries.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

from .capture import imread_unicode
from .types import Camera, Scene
from . import bundle_adjust
from . import rotation_avg


class SfmError(RuntimeError):
    pass


def _match(desc_a, desc_b, ratio: float = 0.8) -> np.ndarray:
    if desc_a is None or desc_b is None or len(desc_a) < 8 or len(desc_b) < 8:
        return np.empty((0, 2), np.int32)
    if desc_a.dtype == np.uint8:
        matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
        knn = matcher.knnMatch(desc_a, desc_b, k=2)
    else:
        index_params = dict(algorithm=1, trees=4)  # FLANN_INDEX_KDTREE
        search_params = dict(checks=48)
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        try:
            knn = matcher.knnMatch(desc_a, desc_b, k=2)
        except cv2.error:
            matcher = cv2.BFMatcher(cv2.NORM_L2)
            knn = matcher.knnMatch(desc_a, desc_b, k=2)
    good = []
    for pair in knn:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good.append((pair[0].queryIdx, pair[0].trainIdx))
    return np.asarray(good, dtype=np.int32).reshape(-1, 2)


def _triangulate(K, R1, t1, R2, t2, p1, p2):
    P1 = K @ np.hstack([R1.T, (-R1.T @ t1).reshape(3, 1)])
    P2 = K @ np.hstack([R2.T, (-R2.T @ t2).reshape(3, 1)])
    Xh = cv2.triangulatePoints(P1, P2, p1.T.astype(np.float64), p2.T.astype(np.float64))
    w = Xh[3]
    valid = np.abs(w) > 1e-9
    X = np.zeros((len(p1), 3), np.float64)
    X[valid] = (Xh[:3, valid] / w[valid]).T

    def reproj(Rcw, tcw, pts):
        cam = pts @ Rcw.T + tcw  # world -> camera
        z = cam[:, 2]
        zz = np.where(np.abs(z) < 1e-9, 1e-9, z)
        u = K[0, 0] * cam[:, 0] / zz + K[0, 2]
        v = K[1, 1] * cam[:, 1] / zz + K[1, 2]
        return np.stack([u, v], -1), z

    uv1, z1 = reproj(R1.T, -R1.T @ t1, X)
    uv2, z2 = reproj(R2.T, -R2.T @ t2, X)
    err = np.linalg.norm(uv1 - p1, axis=1) + np.linalg.norm(uv2 - p2, axis=1)
    ok = valid & (z1 > 0) & (z2 > 0) & (err < 5.0)
    return X, ok


def reconstruct(
    images_dir: Path,
    workspace: Path,
    focal: float | None = None,
    log=print,
    progress=None,
    cancel_event=None,
) -> Scene:
    images_dir = Path(images_dir)
    paths = sorted([
        p for p in images_dir.iterdir()
        if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ])
    if len(paths) < 2:
        raise SfmError("Structure-from-motion needs at least 2 images.")

    feats = []
    for idx, path in enumerate(paths):
        if cancel_event is not None and cancel_event.is_set():
            raise SfmError("Cancelled")
        bgr = imread_unicode(path)
        if bgr is None:
            raise SfmError(f"Could not read {path.name}")
        h, w = bgr.shape[:2]
        scale = min(1.0, 960.0 / max(h, w))
        if scale < 1.0:
            bgr = cv2.resize(bgr, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        # ORB is dramatically denser than SIFT on smooth / repetitive scenes
        # and stays fast on a CPU.  It is the workhorse of this fallback.
        orb = cv2.ORB_create(
            nfeatures=6000, scaleFactor=1.15, nlevels=10,
            edgeThreshold=12, patchSize=31, fastThreshold=6,
        )
        kps, desc = orb.detectAndCompute(gray, None)
        pts = np.array([kp.pt for kp in kps], np.float32).reshape(-1, 2)
        feats.append({"path": path, "bgr": bgr, "gray": gray, "pts": pts,
                      "desc": desc, "w": w, "h": h})
        if progress:
            progress(0.05 + 0.20 * (idx + 1) / len(paths), f"Features {idx + 1}/{len(paths)}")
    log(f"ORB: {sum(len(f['pts']) for f in feats)} features across {len(feats)} frames.")

    # Shared pinhole intrinsics (feature resolution).
    W, H = feats[0]["bgr"].shape[1], feats[0]["bgr"].shape[0]
    # Prefer a known focal length (EXIF / user override); otherwise assume a
    # 50 mm-equivalent horizontal field of view, which suits most phone photos.
    f = float(focal) if focal and focal > 1.0 else 1.2 * max(W, H)
    log(f"Intrinsics: f={f:.1f}px ({2.0 * np.degrees(np.arctan(W / (2.0 * f))):.1f} deg horizontal FOV)")
    K = np.array([[f, 0, W / 2.0], [0, f, H / 2.0], [0, 0, 1.0]], np.float64)

    matches_cache: dict[tuple[int, int], np.ndarray] = {}
    inlier_cache: dict[tuple[int, int], tuple[np.ndarray, np.ndarray | None]] = {}

    def matches(i: int, j: int) -> np.ndarray:
        key = (min(i, j), max(i, j))
        if key not in matches_cache:
            a, b = feats[key[0]], feats[key[1]]
            if key[0] == i:
                m = _match(a["desc"], b["desc"])
            else:
                m = _match(b["desc"], a["desc"])[:, ::-1]
            matches_cache[key] = m
        return matches_cache[key]

    def verified_matches(i: int, j: int):
        """Matches + two-view essential-matrix verification.

        ORB is dense but a repetitive texture can create many false matches;
        the essential-matrix inlier mask is an excellent cheap filter.
        Returns (verified_matches, R_rel, t_rel) in i -> j convention.
        """
        key = (min(i, j), max(i, j))
        if key not in inlier_cache:
            m = matches(i, j)
            if len(m) < 12:
                inlier_cache[key] = (np.empty((0, 2), np.int32), None)
            else:
                q1 = feats[i]["pts"][m[:, 0]]
                q2 = feats[j]["pts"][m[:, 1]]
                try:
                    E, mask = cv2.findEssentialMat(
                        q1, q2, K, method=cv2.RANSAC, prob=0.99999, threshold=1.0
                    )
                except cv2.error:
                    E, mask = None, None
                if E is None or mask is None or int(mask.sum()) < 10:
                    inlier_cache[key] = (np.empty((0, 2), np.int32), None)
                else:
                    mask = mask.ravel().astype(bool)
                    mv = m[mask]
                    try:
                        _, R_rel, t_rel, _ = cv2.recoverPose(E, q1[mask], q2[mask], K)
                        inlier_cache[key] = (mv, (R_rel, t_rel.ravel()))
                    except cv2.error:
                        inlier_cache[key] = (mv, None)
        mv, rel = inlier_cache[key]
        if key[0] == i:
            return mv, rel
        # Reverse the match direction.
        if rel is None:
            return mv[:, ::-1], None
        R_rel, t_rel = rel
        return mv[:, ::-1], (R_rel.T, -R_rel.T @ t_rel)

    def relative_rotation(i: int, j: int):
        m, rel = verified_matches(i, j)
        return rel

    def translation_from_correspondences(R_i, frame, corr, inlier_idx):
        """Solve camera translation only, keeping the (averaged) rotation fixed."""
        sel = [corr[k] for k in inlier_idx]
        obj = np.array([points[pid] for pid, _, _ in sel], np.float64)
        img = np.array([feats[frame]["pts"][kp] for _, _, kp in sel], np.float64)
        d = np.stack([
            (img[:, 0] - K[0, 2]) / K[0, 0],
            (img[:, 1] - K[1, 2]) / K[1, 1],
            np.ones(len(img)),
        ], axis=1)
        skew = np.zeros((len(d), 3, 3), np.float64)
        skew[:, 0, 1] = -d[:, 2]
        skew[:, 0, 2] = d[:, 1]
        skew[:, 1, 0] = d[:, 2]
        skew[:, 1, 2] = -d[:, 0]
        skew[:, 2, 0] = -d[:, 1]
        skew[:, 2, 1] = d[:, 0]
        y = obj @ R_i  # R_i.T @ X for every point
        A = skew.reshape(-1, 3)
        b = (skew @ y[:, :, None]).reshape(-1)
        c, *_ = np.linalg.lstsq(A, b, rcond=None)
        return R_i @ c

    def pnp_pose(i: int, corr: list[tuple[int, int, int]], fixed_R: np.ndarray | None = None):
        obj = np.array([points[pid] for pid, _, _ in corr], np.float32)
        img = np.array([feats[i]["pts"][kp] for _, _, kp in corr], np.float32)
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj, img, K, None, flags=cv2.SOLVEPNP_ITERATIVE,
            iterationsCount=600, reprojectionError=4.0, confidence=0.9999,
        )
        if not ok or inliers is None:
            return None
        inliers = inliers.ravel()
        if len(inliers) < 10 or len(inliers) < int(0.05 * len(corr)):
            log(f"    PnP weak: {len(inliers)} inliers / {len(corr)} matches")
            return None
        if fixed_R is not None:
            t_fixed = translation_from_correspondences(fixed_R, i, corr, inliers)
            log(f"    PnP: {len(inliers)} inliers / {len(corr)} matches (rotation fixed)")
            return fixed_R, t_fixed, inliers
        try:
            rvec, tvec = cv2.solvePnPRefineLM(
                obj[inliers], img[inliers], K, None, rvec, tvec,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 30, 1e-6),
            )
        except cv2.error:
            pass
        Rwc, _ = cv2.Rodrigues(rvec)
        twc = tvec.ravel()
        R_i = Rwc.T
        t_i = -R_i @ twc
        log(f"    PnP: {len(inliers)} inliers / {len(corr)} matches")
        return R_i, t_i, inliers

    def refine_camera(frame: int, fixed_R: np.ndarray | None = None) -> None:
        obs = point_obs.get(frame, {})
        pairs = [(pid, kp) for pid, kp in obs.items() if pid in points]
        if len(pairs) < 8:
            return
        R, t = poses[frame]
        Rcw, tcw = R.T, -R.T @ t
        obj = np.array([points[pid] for pid, _ in pairs], np.float32)
        img = np.array([feats[frame]["pts"][kp] for _, kp in pairs], np.float32)
        cam = obj @ Rcw.T + tcw
        zz = np.maximum(cam[:, 2], 1e-9)
        uv = cam[:, :2] / zz[:, None]
        uv = uv * np.array([K[0, 0], K[1, 1]]) + np.array([K[0, 2], K[1, 2]])
        err = np.linalg.norm(uv - img, axis=1)
        keep = (err < 5.0) & (cam[:, 2] > 0)
        if keep.sum() < 8:
            return
        if fixed_R is not None:
            corr = [(pid, frame, kp) for pid, kp in pairs]
            t_i = translation_from_correspondences(fixed_R, frame, corr, np.nonzero(keep)[0])
            poses[frame] = (fixed_R, t_i)
            return
        obj, img = obj[keep], img[keep]
        rvec, _ = cv2.Rodrigues(Rcw)
        tvec = tcw.reshape(3, 1)
        try:
            rvec2, tvec2 = cv2.solvePnPRefineLM(
                obj, img, K, None, rvec, tvec,
                criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_COUNT, 25, 1e-7),
            )
        except cv2.error:
            return
        Rwc, _ = cv2.Rodrigues(rvec2)
        poses[frame] = (Rwc.T, -Rwc.T @ tvec2.ravel())

    # ---------------------------------------------------- global rotations
    # A chain of pairwise relative rotations is accurate over a few frames but
    # accumulates a small bias over dozens.  Average all verified relative
    # rotations to keep the absolute camera orientation stable, which in turn
    # stops PnP from drifting.
    avg_rot: dict[int, np.ndarray] | None = None
    sequential: list[tuple[int, int, np.ndarray, int]] = []
    for i in range(1, len(feats)):
        mv, rel = verified_matches(i - 1, i)
        if rel is None:
            break
        sequential.append((i - 1, i, rel[0], max(len(mv), 1)))
    if len(sequential) == len(feats) - 1:
        extra: list[tuple[int, int, np.ndarray, int]] = []
        seen = {(i, j) for i, j, *_ in sequential}
        step = max(6, len(feats) // 8)
        for d in (2, 3):
            for i in range(0, len(feats) - d):
                mv, rel = verified_matches(i, i + d)
                if rel is not None and len(mv) >= 30:
                    extra.append((i, i + d, rel[0], len(mv)))
                    seen.add((i, i + d))
        for i in range(8, len(feats)):
            for j in range(0, i - 6, step):
                if (j, i) in seen:
                    continue
                mv, rel = verified_matches(j, i)
                if rel is not None and len(mv) >= 24:
                    extra.append((j, i, rel[0], len(mv)))
                    seen.add((j, i))
        try:
            avg_rot = rotation_avg.average(len(feats), sequential, extra, log=log)
        except Exception as exc:
            log(f"Rotation averaging skipped: {exc}")
            avg_rot = None
    else:
        log("Rotation averaging skipped: sequential chain is incomplete.")

    poses: dict[int, tuple[np.ndarray, np.ndarray]] = {0: (np.eye(3), np.zeros(3))}
    kp_point: dict[int, dict[int, int]] = {0: {}}
    points: dict[int, np.ndarray] = {}
    point_obs: dict[int, dict[int, int]] = {}
    point_rgb: dict[int, np.ndarray] = {}
    next_point = 0

    m01 = matches(0, 1)
    if len(m01) < 12:
        raise SfmError("Too few matches between the first two frames. "
                       "Capture slower / with more overlap and texture.")
    p1 = feats[0]["pts"][m01[:, 0]]
    p2 = feats[1]["pts"][m01[:, 1]]
    E, inl = cv2.findEssentialMat(p1, p2, K, method=cv2.RANSAC, prob=0.99999, threshold=1.0)
    if E is None or inl is None or int(inl.sum()) < 10:
        raise SfmError("Could not estimate the initial camera pair.")
    inl = inl.ravel().astype(bool)
    p1i, p2i = p1[inl], p2[inl]
    _, R12, t12, _ = cv2.recoverPose(E, p1i, p2i, K)
    if np.linalg.norm(t12) < 1e-9:
        raise SfmError("Initial camera baseline is zero (frames too similar).")
    t12 = t12.ravel() / np.linalg.norm(t12)
    # recoverPose gives x2 = R*x1 + t (world->camera for camera 2); convert to c2w.
    R2 = avg_rot[1] if avg_rot is not None else R12.T
    t2 = -R2 @ t12
    poses[1] = (R2, t2)

    X, good = _triangulate(K, *poses[0], *poses[1], p1i, p2i)
    kp_point[1] = {}
    for k in range(len(X)):
        if not good[k]:
            continue
        pid = next_point
        next_point += 1
        points[pid] = X[k]
        point_obs[pid] = {0: int(m01[inl][k, 0]), 1: int(m01[inl][k, 1])}
        kp_point[0][int(m01[inl][k, 0])] = pid
        kp_point[1][int(m01[inl][k, 1])] = pid
        u, v = p2i[k].astype(int)
        u = int(np.clip(u, 0, feats[1]["bgr"].shape[1] - 1))
        v = int(np.clip(v, 0, feats[1]["bgr"].shape[0] - 1))
        point_rgb[pid] = feats[1]["bgr"][v, u][::-1].astype(np.float32) / 255.0

    log(f"Initial pair: {int(good.sum())} triangulated points.")

    registered = [0, 1]
    last_step = 0.0
    for i in range(2, len(feats)):
        if cancel_event is not None and cancel_event.is_set():
            raise SfmError("Cancelled")

        # Match against several recent frames (robust tracking) plus a few
        # older frames (cheap loop closure for orbit-style captures).
        sources = list(registered[-8:])
        if len(registered) > 8:
            step = max(3, len(registered) // 8)
            sources += list(registered[:-8:step])[-4:]
        sources = sorted(set(sources))

        correspondences: list[tuple[int, int, int]] = []  # pid, source frame, kp_i
        seen_pid: set[int] = set()
        used_kp: set[int] = set()
        for src in sources:
            m, _rel = verified_matches(src, i)
            mapping = kp_point.get(src, {})
            for kp_a, kp_b in m:
                pid = mapping.get(int(kp_a))
                kp_b = int(kp_b)
                if pid is None or pid in seen_pid or kp_b in used_kp:
                    continue
                seen_pid.add(pid)
                used_kp.add(kp_b)
                correspondences.append((pid, src, kp_b))

        fixed_R = avg_rot.get(i) if avg_rot is not None else None
        result = pnp_pose(i, correspondences, fixed_R=fixed_R) if len(correspondences) >= 12 else None
        if result is not None and fixed_R is None:
            R_i, t_i, inliers = result
            # Sanity check against the latest relative rotation.  Essential
            # matrices are very reliable, so a wild disagreement usually
            # means the PnP fit locked onto repetitive features.
            rel = relative_rotation(registered[-1], i)
            if rel is not None:
                R_rel, _ = rel
                R_prev = poses[registered[-1]][0]
                R_pred = R_prev @ R_rel.T
                cosang = (np.trace(R_pred.T @ R_i) - 1.0) * 0.5
                angle = np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0)))
                if angle > 55.0:
                    log(f"  frame {i}: pose rejected ({angle:.0f} deg vs two-view motion)")
                    result = None
        if result is not None:
            R_i, t_i, inliers = result
        if result is None:
            # Bridge skipped frames with a relative pose from the last good
            # frame.  Translation scale comes from the previous successful
            # step, which is far better than assuming a unit baseline.
            rel = relative_rotation(registered[-1], i)
            if rel is None:
                log(f"  frame {i}: not enough stable matches, skipping")
                continue
            R_rel, t_rel = rel
            R_prev, t_prev = poses[registered[-1]]
            R_i = fixed_R if fixed_R is not None else R_prev @ R_rel.T
            step = last_step if last_step > 1e-6 else 0.2
            t_i = t_prev - R_i @ (t_rel / max(np.linalg.norm(t_rel), 1e-9) * step)
            inliers = None
            log(f"  frame {i}: relative-pose fallback (step {step:.3f})")

        poses[i] = (R_i, t_i)
        if len(registered) >= 1:
            step_now = float(np.linalg.norm(t_i - poses[registered[-1]][1]))
            if step_now > 1e-6 and np.isfinite(step_now):
                last_step = step_now
        kp_point.setdefault(i, {})
        registered.append(i)

        # Only keep inlier correspondences; they become map observations.
        if inliers is None:
            used = []
        else:
            used = [correspondences[k] for k in inliers]
        for pid, src, kp in used:
            point_obs[pid][i] = kp
            kp_point[i][kp] = pid

        # Triangulate fresh points against recent + loop-closure frames.
        added = 0
        reliable = inliers is not None
        new_sources = list(registered[-6:-1]) if reliable else []
        if reliable and len(registered) > 8:
            new_sources += [registered[0], registered[len(registered) // 2]]
        new_sources = sorted(set(s for s in new_sources if s != i))
        for src in new_sources:
            m, _rel = verified_matches(src, i)
            mapping = kp_point.get(src, {})
            fresh = [(int(a), int(b)) for a, b in m
                     if int(a) not in mapping and int(b) not in kp_point[i]]
            if len(fresh) < 8:
                continue
            pp = np.array(fresh, np.int32)
            a_pts = feats[src]["pts"][pp[:, 0]]
            b_pts = feats[i]["pts"][pp[:, 1]]
            X, good = _triangulate(K, *poses[src], *poses[i], a_pts, b_pts)
            if len(X) == 0:
                continue
            # Parallax guard: reject points observed along (almost) the same ray.
            C1, C2 = poses[src][1], poses[i][1]
            rays1 = X - C1
            rays2 = X - C2
            n1 = np.linalg.norm(rays1, axis=1)
            n2 = np.linalg.norm(rays2, axis=1)
            cosang = np.sum(rays1 * rays2, axis=1) / np.maximum(n1 * n2, 1e-9)
            good &= np.degrees(np.arccos(np.clip(cosang, -1.0, 1.0))) > 0.8
            if not good.any():
                continue
            # Prefer points spread over the frame: cap per source.
            take = np.nonzero(good)[0][:240]
            for k in take:
                if added >= 900:
                    break
                pid = next_point
                next_point += 1
                points[pid] = X[k]
                point_obs[pid] = {src: pp[k, 0], i: pp[k, 1]}
                kp_point[src][int(pp[k, 0])] = pid
                kp_point[i][int(pp[k, 1])] = pid
                u, v = b_pts[k].astype(int)
                u = int(np.clip(u, 0, feats[i]["bgr"].shape[1] - 1))
                v = int(np.clip(v, 0, feats[i]["bgr"].shape[0] - 1))
                point_rgb[pid] = feats[i]["bgr"][v, u][::-1].astype(np.float32) / 255.0
                added += 1

        # Refine the active window, and periodically the whole map's cameras.
        for f in registered[-4:]:
            refine_camera(f, avg_rot.get(f) if avg_rot is not None else None)
        if len(registered) >= 5 and (i % 3 == 0 or i == len(feats) - 1):
            window = registered[-7:]
            bundle_adjust.refine(
                poses, points, point_obs, feats, K, window,
                fixed_id=window[0], max_nfev=12, log=log,
            )
        if i % 8 == 0:
            for f in registered:
                refine_camera(f, avg_rot.get(f) if avg_rot is not None else None)

        log(f"  frame {i}: registered, +{added} points, {len(points)} total")
        if progress:
            progress(0.25 + 0.55 * (i - 1) / max(1, len(feats) - 2),
                     f"Tracking frame {i + 1}/{len(feats)}")

    # Final alternating refinement: global bundle adjustment, then discard
    # inconsistent map points.  This is what removes drift at loop closure.
    bundle_adjust.refine(
        poses, points, point_obs, feats, K, registered,
        fixed_id=0, max_nfev=30, log=log,
    )
    for f in registered:
        refine_camera(f, avg_rot.get(f) if avg_rot is not None else None)
    bad: list[int] = []
    for pid, obs in point_obs.items():
        if pid not in points or len(obs) < 2:
            continue
        errs = []
        for f, kp in obs.items():
            R, t = poses[f]
            Rcw = R.T
            Xc = Rcw @ (points[pid] - t)
            if Xc[2] <= 0:
                errs.append(1e3)
                continue
            uv = K[:2, :2] @ (Xc[:2] / Xc[2]) + K[:2, 2]
            errs.append(float(np.linalg.norm(uv - feats[f]["pts"][kp])))
        if len(errs) >= 3 and np.median(errs) > 6.0:
            bad.append(pid)
    for pid in bad:
        points.pop(pid, None)
        point_obs.pop(pid, None)
        point_rgb.pop(pid, None)

    if len(registered) < 3:
        raise SfmError("SfM could not register enough frames. "
                       "Try COLMAP mode or a slower capture.")

    # Convert feature-space cameras into the original image resolution.
    out_cameras: list[Camera] = []
    for idx in sorted(registered):
        feat = feats[idx]
        orig_w, orig_h = feat["w"], feat["h"]
        K_orig = K.copy()
        s = orig_w / float(feat["bgr"].shape[1])
        K_orig[0, :] *= s
        K_orig[1, :] *= s
        R, t = poses[idx]
        out_cameras.append(Camera(
            name=feat["path"].name, width=orig_w, height=orig_h,
            fx=float(K_orig[0, 0]), fy=float(K_orig[1, 1]),
            cx=float(K_orig[0, 2]), cy=float(K_orig[1, 2]),
            R=R, t=t, image_path=str(feat["path"]),
        ))

    pids = sorted(points)
    pts = np.array([points[p] for p in pids], np.float32).reshape(-1, 3)
    cols = np.array([point_rgb.get(p, (0.5, 0.5, 0.5)) for p in pids], np.float32)
    log(f"SfM: {len(out_cameras)} cameras, {len(pts)} points.")
    if progress:
        progress(0.82, "Sparse reconstruction complete")
    scene = Scene(cameras=out_cameras, points=pts, colors=cols, source="sfm")
    workspace.mkdir(parents=True, exist_ok=True)
    (workspace / "scene.json").write_text(
        json.dumps({"source": "sfm", "cameras": [c.to_dict() for c in out_cameras]}, indent=2),
        encoding="utf-8",
    )
    return scene
