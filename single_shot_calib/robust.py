"""Robust single-shot refinements (enabled with `robust.enabled: true`).

The paper core is unchanged: spherical intensity image -> derivative corner
candidates -> checkerboard lattice -> E-PnP + LM.  This module adds, for the same
single capture:

* camera: temporal median of the stationary frames, pixel-centre-correct corner
  scaling and a CLAHE fallback for glare;
* LiDAR: hole filling of the spherical image, a board search restricted to planar,
  board-sized range segments, and a plane-constrained checker pattern fit that
  returns all inner corners from every board point;
* quality gates (cell contrast, coverage, square-size consistency, stationarity).
"""
from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from scipy.optimize import minimize

from .detector import lidar_lattice, recover_xyz
from .spherical import project


# ------------------------------------------------------------------ camera
def median_image(images, max_deviation):
    """Median of the frames; frames that differ from it (motion, exposure jump) are dropped."""
    stack = np.stack(images).astype(np.float32)
    med = np.median(stack, axis=0)
    dev = np.abs(stack - med).mean(axis=(1, 2, 3))
    keep = dev <= max_deviation
    if keep.sum() >= 3 and not keep.all():
        med = np.median(stack[keep], axis=0)
    return med.astype(np.uint8), dict(frames=len(images), frames_used=int(keep.sum()),
                                      mean_frame_deviation=float(dev.mean()), max_frame_deviation=float(dev.max()))


def camera_corners(image, board, scales, clahe=True):
    """findChessboardCornersSB with upscaling; corners mapped back as (x + 0.5) / s - 0.5."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    variants = [("plain", gray)] + ([("clahe", cv2.createCLAHE(3.0, (8, 8)).apply(gray))] if clahe else [])
    for name, g in variants:
        for s in scales:
            test = g if s == 1 else cv2.resize(g, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
            ok, c = cv2.findChessboardCornersSB(test, tuple(board), cv2.CALIB_CB_EXHAUSTIVE | cv2.CALIB_CB_ACCURACY)
            if ok:
                return (c.reshape(-1, 2).astype(np.float64) + 0.5) / s - 0.5, dict(upscale=int(s), preprocessing=name)
    raise RuntimeError(f"Camera checkerboard {board} was not detected")


# ------------------------------------------------------- spherical helpers
def fill_holes(image, valid):
    """Inpaint small scan gaps only; large empty areas (outside the FOV) stay black."""
    holes = (~valid).astype(np.uint8)
    large = cv2.morphologyEx(holes, cv2.MORPH_OPEN, np.ones((4, 4), np.uint8)).astype(bool)
    return cv2.inpaint(image, (holes.astype(bool) & ~large).astype(np.uint8), 2, cv2.INPAINT_TELEA)


def range_image(xyz, uv, ids, shape):
    h, w = shape
    r = np.linalg.norm(xyz[ids], axis=1)
    s = np.zeros((h, w)); n = np.zeros((h, w))
    np.add.at(s, (uv[:, 1].astype(int), uv[:, 0].astype(int)), r)
    np.add.at(n, (uv[:, 1].astype(int), uv[:, 0].astype(int)), 1)
    out = np.full((h, w), np.nan); out[n > 0] = s[n > 0] / n[n > 0]
    return out


def board_segments(rimg, sph, cfg):
    """Planar, board-sized range segments bounded by depth edges (single-shot board ROI)."""
    r = rimg.copy()
    valid = np.isfinite(r)
    # Fill 1-2 px scan gaps in the range image so the board is not split by them.
    filled = cv2.inpaint(np.nan_to_num(r).astype(np.float32), (~valid).astype(np.uint8), 2, cv2.INPAINT_NS)
    near_valid = cv2.dilate(valid.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    r = np.where(near_valid, filled, np.nan)
    thr = np.maximum(float(cfg["depth_jump_m"]), float(cfg["depth_jump_fraction"]) * np.nan_to_num(r, nan=0.0))
    edge = ~np.isfinite(r)
    for dy, dx in ((0, 1), (1, 0), (0, -1), (-1, 0)):
        nb = np.roll(r, (dy, dx), axis=(0, 1))
        edge |= np.nan_to_num(np.abs(r - nb), nan=0.0) > thr
    # Erode before labelling so thin bridges (the hanging pole, mixed edge pixels) do
    # not join the board to the ceiling or walls; regions are grown back afterwards.
    k = np.ones((int(cfg["bridge_cut_px"]),) * 2, np.uint8)
    core = cv2.erode((~edge).astype(np.uint8), k)
    labels, n = ndimage.label(core)
    labels = np.where(~edge, ndimage.grey_dilation(labels, footprint=k.astype(bool)), 0)
    pix = np.radians(float(sph["horizontal_fov_deg"]) / (int(sph["width"]) - 1))
    out = []
    for i, sl in enumerate(ndimage.find_objects(labels), start=1):
        if sl is None:
            continue
        mask = labels[sl] == i
        cnt = int(mask.sum())
        if cnt < int(cfg["minimum_region_px"]):
            continue
        rng = float(np.nanmedian(r[sl][mask]))
        area = cnt * (rng * pix) ** 2
        if cfg["board_area_m2"][0] <= area <= cfg["board_area_m2"][1]:
            full = np.zeros_like(edge); full[sl] = mask
            out.append(dict(mask=full, slice=sl, area_m2=area, range_m=rng, pixels=cnt))
    out.sort(key=lambda d: -d["pixels"])
    return out[: int(cfg["maximum_regions"])]


# --------------------------------------------------------------- plane fit
def fit_plane(points, tol, iterations=400, seed=0):
    """RANSAC + least squares; returns (n, d, inliers) with n.x + d = 0."""
    rng = np.random.default_rng(seed); best = None
    for _ in range(iterations):
        p = points[rng.choice(len(points), 3, replace=False)]
        n = np.cross(p[1] - p[0], p[2] - p[0])
        if np.linalg.norm(n) < 1e-9:
            continue
        n /= np.linalg.norm(n)
        inl = np.abs((points - p[0]) @ n) < tol
        if best is None or inl.sum() > best.sum():
            best = inl
    for _ in range(2):
        q = points[best]; c = q.mean(0)
        n = np.linalg.svd(q - c, full_matrices=False)[2][-1]
        best = np.abs((points - c) @ n) < tol
    return n, float(-n @ c), best


def largest_cluster(points, ids, gap):
    vox = np.floor(points / gap).astype(np.int64); vox -= vox.min(0)
    grid = np.zeros(vox.max(0) + 1, bool); grid[tuple(vox.T)] = True
    labels, n = ndimage.label(grid, structure=np.ones((3, 3, 3)))
    if n <= 1:
        return ids
    lab = labels[tuple(vox.T)]
    return ids[lab == np.bincount(lab)[1:].argmax() + 1]


# ------------------------------------------------------------- pattern fit
def _plane_basis(points, normal):
    c = points.mean(0); n = normal / np.linalg.norm(normal)
    if n @ c > 0:
        n = -n  # towards the sensor
    q = points - c; q -= np.outer(q @ n, n)
    e1 = np.linalg.svd(q, full_matrices=False)[2][0]; e1 -= (e1 @ n) * n; e1 /= np.linalg.norm(e1)
    return c, n, e1, np.cross(n, e1)


def _template(squares, s_px, theta, sign):
    nx, ny = squares
    r = int(np.ceil(0.5 * np.hypot(nx, ny) * s_px + 2))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1].astype(np.float64)
    c, sn = np.cos(theta), np.sin(theta)
    a = (c * xx + sn * yy) / s_px + nx / 2; b = (-sn * xx + c * yy) / s_px + ny / 2
    inside = (a > 0) & (a < nx) & (b > 0) & (b < ny)
    return (np.sign(np.sin(np.pi * a) * np.sin(np.pi * b)) * sign * inside).astype(np.float32), r


def _soft_score(p, uv, z, squares, sign, edge=0.08, k=4.0):
    theta, tx, ty, s = p
    if s <= 0:
        return 1e9
    nx, ny = squares; c, sn = np.cos(theta), np.sin(theta)
    x, y = uv[:, 0] - tx, uv[:, 1] - ty
    a = (c * x + sn * y) / s + nx / 2; b = (-sn * x + c * y) / s + ny / 2
    sig = lambda v: 1 / (1 + np.exp(-np.clip(v / edge, -50, 50)))
    w = sig(a) * sig(nx - a) * sig(b) * sig(ny - b)
    return -float(np.sum(w * np.tanh(k * np.sin(np.pi * a) * np.sin(np.pi * b)) * sign * z))


def fit_pattern(points, intensity, normal, board, cfg, square_m=None):
    """Fit the ideal checker pattern to reflectivity on the board plane.

    `points` must lie on the plane.  Returns all inner corners in 3D (row-major,
    cols = board[0]) plus contrast / coverage diagnostics."""
    cols, rows = map(int, board); squares = (cols + 1, rows + 1)
    sign = -1.0 if cfg.get("corner_square", "black") == "black" else 1.0
    c, n, e1, e2 = _plane_basis(points, normal)
    uv = np.column_stack(((points - c) @ e1, (points - c) @ e2))
    med = np.median(intensity); mad = 1.4826 * np.median(np.abs(intensity - med)) + 1e-6
    z = np.clip((intensity - med) / mad, -2.5, 2.5)
    lo = np.maximum(np.percentile(uv, 0.5, 0), np.median(uv, 0) - 0.6) - 0.05
    hi = np.minimum(np.percentile(uv, 99.5, 0), np.median(uv, 0) + 0.6) + 0.05
    keep = np.all((uv >= lo) & (uv <= hi), axis=1); uv, z = uv[keep], z[keep]
    res = float(cfg["raster_m"]); ij = np.floor((uv - lo) / res).astype(int)
    acc = np.zeros((ij[:, 1].max() + 1, ij[:, 0].max() + 1)); cnt = np.zeros_like(acc)
    np.add.at(acc, (ij[:, 1], ij[:, 0]), z); np.add.at(cnt, (ij[:, 1], ij[:, 0]), 1)
    img = np.divide(acc, cnt, out=np.zeros_like(acc), where=cnt > 0).astype(np.float32)
    sizes = [square_m] if square_m else np.arange(cfg["square_search_m"][0], cfg["square_search_m"][1] + 1e-9, cfg["square_step_m"])
    best = None
    for s in sizes:
        for theta in np.radians(np.arange(0.0, 180.0, float(cfg["angle_step_deg"]))):
            tpl, r = _template(squares, s / res, theta, sign)
            pad = max(tpl.shape[0] - img.shape[0], tpl.shape[1] - img.shape[1], 0) + 1
            src = cv2.copyMakeBorder(img, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=0)
            _, val, _, loc = cv2.minMaxLoc(cv2.matchTemplate(src, tpl, cv2.TM_CCORR))
            if best is None or val > best[0]:
                best = (val, theta, lo + (np.array([loc[0] + r - pad, loc[1] + r - pad]) + 0.5) * res, s)
    _, th0, t0, s0 = best
    near = np.linalg.norm(uv - t0, axis=1) < 0.6 * np.hypot(*squares) * s0 + 0.05
    opt = dict(xatol=1e-5, fatol=1e-4, maxiter=4000)
    if square_m:
        f = minimize(lambda p: _soft_score(np.r_[p, square_m], uv[near], z[near], squares, sign), np.r_[th0, t0],
                     method="Nelder-Mead", options=opt)
        theta, tx, ty = f.x; s = square_m
    else:
        f = minimize(lambda p: _soft_score(p, uv[near], z[near], squares, sign), np.r_[th0, t0, s0],
                     method="Nelder-Mead", options=opt)
        theta, tx, ty, s = f.x
    cs, sn = np.cos(theta), np.sin(theta)
    x, y = uv[:, 0] - tx, uv[:, 1] - ty
    a = (cs * x + sn * y) / s + squares[0] / 2; b = (-sn * x + cs * y) / s + squares[1] / 2
    fa, fb = a - np.floor(a), b - np.floor(b)
    core = (a > 0.15) & (a < squares[0] - 0.15) & (b > 0.15) & (b < squares[1] - 0.15) & \
           (np.abs(fa - 0.5) < 0.3) & (np.abs(fb - 0.5) < 0.3)
    parity = (np.floor(a) + np.floor(b)) % 2 == 0
    cc, oc = z[core & parity], z[core & ~parity]
    contrast = float(np.mean(oc) - np.mean(cc)) * (-sign) if len(cc) and len(oc) else 0.0
    coverage = float(np.mean([np.any(core & (np.floor(a) == i) & (np.floor(b) == j))
                              for i in range(squares[0]) for j in range(squares[1])]))
    ia, jb = np.meshgrid(np.arange(1, cols + 1), np.arange(1, rows + 1))
    ac, bc = ia.ravel() - squares[0] / 2, jb.ravel() - squares[1] / 2
    qx = tx + s * (cs * ac - sn * bc); qy = ty + s * (sn * ac + cs * bc)
    corners = c + qx[:, None] * e1 + qy[:, None] * e2
    ob = np.array([[0, 0], [squares[0], 0], [squares[0], squares[1]], [0, squares[1]]], float) - np.array(squares) / 2
    ox = tx + s * (cs * ob[:, 0] - sn * ob[:, 1]); oy = ty + s * (sn * ob[:, 0] + cs * ob[:, 1])
    return dict(corners_xyz=corners, square_size_m=float(s), contrast=contrast, cell_coverage=coverage,
                normal=n, centroid=c, e1=e1, e2=e2, outline_xyz=c + ox[:, None] * e1 + oy[:, None] * e2,
                raster=img, raster_origin=lo, raster_res=res)


# -------------------------------------------------------- board detection
def detect_board(xyz, intensity, board, cfg, square_m=None, frame=None):
    """Paper lattice + robust refinement.  Returns a dict; raises RuntimeError on failure."""
    sph, det, rb = cfg["spherical_projection"], cfg["detection"], cfg["robust"]
    image, uv, ids, valid = project(xyz, intensity, sph)
    filled = fill_holes(image, valid) if rb["fill_holes"] else image
    h, w = image.shape
    rimg = range_image(xyz, uv, ids, (h, w))
    segments = board_segments(rimg, sph, rb) if rb["board_roi"] else []
    pix = np.full(len(xyz), -1); pix[ids] = uv[:, 1].astype(int) * w + uv[:, 0].astype(int)
    rng = np.linalg.norm(xyz, axis=1)
    m = int(rb["crop_margin_px"])
    candidates = []
    for sgm in segments:
        sl = sgm["slice"]
        candidates.append(dict(mask=sgm["mask"], window=(max(sl[0].start - m, 0), min(sl[0].stop + m, h),
                                                         max(sl[1].start - m, 0), min(sl[1].stop + m, w))))
    # The paper's whole-image lattice search is always a candidate as well.
    try:
        grid, matched, cand = lidar_lattice(filled, board, det)
        cols, rows = map(int, board)
        H, _ = cv2.findHomography(np.array([[i, j] for j in range(rows) for i in range(cols)], float), grid.astype(float))
        out = cv2.perspectiveTransform(np.array([[[-1.5, -1.5]], [[cols + .5, -1.5]], [[cols + .5, rows + .5]], [[-1.5, rows + .5]]]), H)
        mask = np.zeros((h, w), np.uint8); cv2.fillPoly(mask, [np.int32(out.reshape(-1, 2))], 1)
        ys, xs = np.nonzero(mask)
        if len(xs):
            candidates.append(dict(mask=mask.astype(bool), window=(max(ys.min() - m, 0), min(ys.max() + m, h),
                                                                  max(xs.min() - m, 0), min(xs.max() + m, w)),
                                   lattice=(grid, matched, cand)))
    except RuntimeError:
        pass
    if not candidates:
        raise RuntimeError("No board candidate (range segment or lattice) found in the LiDAR data")

    tried = []
    for cd in candidates:
        y0, y1, x0, x1 = cd["window"]
        if "lattice" not in cd:
            try:
                g, mt, cn = lidar_lattice(np.ascontiguousarray(filled[y0:y1, x0:x1]), board, det)
                cd["lattice"] = (g + [x0, y0], mt, cn + [x0, y0])
            except RuntimeError:
                cd["lattice"] = None
        grow = np.ones((int(rb["roi_grow_px"]),) * 2, np.uint8)  # range gate + plane remove the background
        mask = cv2.dilate(cd["mask"].astype(np.uint8), grow).ravel().astype(bool)
        sel = np.flatnonzero((pix >= 0) & mask[np.clip(pix, 0, None)])
        if len(sel) < int(rb["minimum_points"]):
            continue
        hist, edges = np.histogram(rng[sel], bins=np.arange(rng[sel].min(), rng[sel].max() + 0.04, 0.02))
        peak = edges[np.argmax(np.convolve(hist, np.ones(5), "same"))] + 0.01
        sel = sel[np.abs(rng[sel] - peak) < float(rb["range_gate_m"])]
        if len(sel) < int(rb["minimum_points"]):
            continue
        bid = largest_cluster(xyz[sel], sel, float(rb["cluster_gap_m"]))
        if len(bid) < int(rb["minimum_points"]):
            continue
        # Plane from the darker half (bright cells saturate and range-walk short).
        dark = bid[intensity[bid] <= np.median(intensity[bid])]
        n, d, _ = fit_plane(xyz[dark], float(rb["plane_tolerance_m"]))
        bid = bid[np.abs(xyz[bid] @ n + d) < float(rb["board_thickness_m"])]
        P = xyz[bid]; rays = P / np.linalg.norm(P, axis=1, keepdims=True)
        Pp = rays * (-d / (rays @ n))[:, None]  # along each beam onto the plane: angles only
        ok = np.linalg.norm(Pp - P, axis=1) < 2 * float(rb["board_thickness_m"])
        bid, P, Pp = bid[ok], P[ok], Pp[ok]
        if len(bid) < int(rb["minimum_points"]):
            continue
        try:
            fit = fit_pattern(Pp, intensity[bid], n, board, rb, square_m)
        except Exception:  # noqa: BLE001 - try the next candidate
            continue
        tried.append(dict(cd, fit=fit, board_ids=bid, on_plane=Pp, normal=n, score=fit["contrast"] * fit["cell_coverage"]))
    if not tried:
        raise RuntimeError("No LiDAR board candidate could be fitted")
    best = max(tried, key=lambda t: t["score"])
    fit = best["fit"]
    # Free square-size fit as a consistency check of the configured size.
    free = fit_pattern(best["on_plane"], intensity[best["board_ids"]], best["normal"], board, rb) if square_m else fit
    lat = best["lattice"]
    paper_xyz = paper_idx = None
    if lat is not None:
        paper_xyz, paper_idx = recover_xyz(lat[0], lat[1], uv, ids, xyz, float(det["point_recovery_radius_px"]))
    drift = _stationarity(xyz, frame, best["board_ids"], fit["normal"]) if frame is not None else None
    return dict(corners_xyz=fit["corners_xyz"], lattice_indices=np.arange(len(fit["corners_xyz"])), fit=fit,
                square_size_m=fit["square_size_m"], free_square_size_m=free["square_size_m"],
                contrast=fit["contrast"], cell_coverage=fit["cell_coverage"], board_points=int(len(best["board_ids"])),
                board_range_m=float(np.linalg.norm(fit["centroid"])), board_drift_mm=drift,
                window=best["window"], candidates=len(candidates), lattice=lat,
                paper_xyz=paper_xyz, paper_indices=paper_idx, intensity_image=filled, segments=segments)


def _stationarity(xyz, frame, board_ids, normal, window=10):
    P = xyz[board_ids]; f = frame[board_ids]; c = P.mean(0); offs = []
    for s in range(0, int(f.max()) + 1, window):
        m = (f >= s) & (f < s + window)
        if m.sum() >= 200:
            offs.append(float((P[m].mean(0) - c) @ normal) * 1000)
    return float(np.ptp(offs)) if len(offs) > 1 else None


def gates(board_result, square_m, cfg):
    """Reasons why the LiDAR board should be reviewed (empty list = accepted)."""
    rb, out = cfg["robust"], []
    if board_result["contrast"] < rb["minimum_contrast"]:
        out.append(f"low LiDAR cell contrast {board_result['contrast']:.2f}")
    if board_result["cell_coverage"] < rb["minimum_cell_coverage"]:
        out.append(f"incomplete LiDAR cell coverage {board_result['cell_coverage']:.2f}")
    if square_m and abs(board_result["free_square_size_m"] - square_m) > rb["square_size_tolerance"] * square_m:
        out.append(f"free-fit square {board_result['free_square_size_m'] * 1000:.1f} mm vs configured {square_m * 1000:.1f} mm")
    if board_result["board_drift_mm"] is not None and board_result["board_drift_mm"] > rb["maximum_board_drift_mm"]:
        out.append(f"board moved {board_result['board_drift_mm']:.1f} mm during the capture")
    return out
