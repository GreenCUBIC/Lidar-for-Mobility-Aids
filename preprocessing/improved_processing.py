#!/usr/bin/env python3
import argparse, os, re, json, glob
from pathlib import Path
from typing import Dict, Tuple, List
import numpy as np
import random
import math
from pathlib import Path

# Optional deps
try:
    import pandas as pd
except Exception:
    pd = None

try:
    import open3d as o3d
    _O3D = True
except Exception:
    _O3D = False

from sklearn.cluster import DBSCAN
DEBUG = False

# ---------- IO ----------
def read_csv_points(csv_path: Path, xcol, ycol, zcol,
                    icol=None,  # accepted but ignored
                    delimiter=",", has_header=True):
    """
    Read ONLY X,Y,Z from CSV and return (N,3) float32.
    - If has_header=True: xcol/ycol/zcol are column NAMES.
    - If has_header=False: xcol/ycol/zcol are INTEGER indices.
    'icol' is accepted for backward compatibility but ignored.
    """
    # --- Fast path with pandas ---
    if pd is not None:
        if has_header:
            # Read with explicit column check
            df = pd.read_csv(csv_path, delimiter=delimiter or ",", nrows=0)
            available_cols = df.columns.tolist()
            
            # Verify columns exist
            for col in [xcol, ycol, zcol]:
                if col not in available_cols:
                    raise ValueError(
                        f"Column '{col}' not found in {csv_path.name}\n"
                        f"Available columns: {available_cols}"
                    )
            
            # Now read the actual data
            df = pd.read_csv(csv_path, delimiter=delimiter or ",",
                           usecols=[xcol, ycol, zcol])
            
            # CRITICAL: Maintain explicit column order
            result = df[[xcol, ycol, zcol]].to_numpy(dtype=np.float32)
            
            # Sanity check
            if np.any(~np.isfinite(result)):
                n_bad = np.sum(~np.isfinite(result))
                print(f"[WARN] {csv_path.name} has {n_bad} non-finite values")
            
            return result
        else:
            usecols = [int(xcol), int(ycol), int(zcol)]
            df = pd.read_csv(csv_path, delimiter=delimiter or ",",
                             header=None, usecols=usecols)
            df.columns = [0, 1, 2]
            return df[[0, 1, 2]].to_numpy(dtype=np.float32)

    # --- Fallback without pandas ---
    rows = []
    if has_header:
        import csv as _csv
        with open(csv_path, "r", newline="") as f:
            r = _csv.DictReader(f, delimiter=delimiter or ",")
            for row in r:
                rows.append([float(row[xcol]),
                             float(row[ycol]),
                             float(row[zcol])])
    else:
        import csv as _csv
        xi, yi, zi = int(xcol), int(ycol), int(zcol)
        with open(csv_path, "r", newline="") as f:
            r = _csv.reader(f, delimiter=delimiter or ",")
            for row in r:
                rows.append([float(row[xi]),
                             float(row[yi]),
                             float(row[zi])])
    return np.asarray(rows, dtype=np.float32)

_warned_intensity = False  # optional: warn once if intensity is requested

def write_bin(path: Path, pts: np.ndarray, include_intensity=None):
    """
    Always write XYZ only (float32). Extra columns are ignored.
    'include_intensity' is accepted for backward compatibility but ignored.
    """
    global _warned_intensity
    if include_intensity and not _warned_intensity:
        # One-time heads-up so it's obvious in logs that intensity is dropped
        print("[INFO] include_intensity was requested but is ignored; exporting XYZ only.", flush=True)
        _warned_intensity = True

    path.parent.mkdir(parents=True, exist_ok=True)
    if pts.size == 0:
        np.array([], dtype=np.float32).tofile(path)
        return
    pts[:, :3].astype(np.float32, copy=False).tofile(path)

# ---------- Geometry ----------

def clip_distance(pts: np.ndarray, max_dist: float):
    if pts.size == 0: return pts, pts
    xyz = pts[:, :3]
    dist = np.linalg.norm(xyz, axis=1)
    good = np.isfinite(xyz).all(axis=1) & (dist <= max_dist)
    return pts[good], pts[~good]

def _o3d_pcd(xyz: np.ndarray):
    p = o3d.geometry.PointCloud()
    p.points = o3d.utility.Vector3dVector(xyz.astype(np.float64))
    return p

def remove_ground(pts: np.ndarray, ransac_dist=0.03, max_iter=400):
    if pts.size == 0: return pts, np.empty((0, pts.shape[1]), np.float32)
    if _O3D and pts.shape[0] >= 1000:
        pcd = _o3d_pcd(pts[:, :3])
        try:
            model, inliers = pcd.segment_plane(distance_threshold=ransac_dist, ransac_n=3, num_iterations=max_iter)
            a,b,c,d = model
            # horizontal-ish if |c| dominates
            if abs(c) >= max(abs(a), abs(b)):
                inliers = np.asarray(inliers, dtype=int)
                keep_mask = np.ones(len(pts), dtype=bool); keep_mask[inliers] = False
                return pts[keep_mask], pts[~keep_mask]
        except Exception:
            pass
    # fallback: remove low z-band
    z = pts[:,2]; z0 = np.percentile(z, 5.0)
    keep = z > (z0 + 0.02)
    return pts[keep], pts[~keep]

def remove_walls_iter(pts: np.ndarray, ransac_dist=0.03, max_iter=500, repeats=3):
    if pts.size == 0 or not _O3D:
        return pts, np.empty((0, pts.shape[1]), np.float32), []
    
    if repeats == 0:  # Handle --max-walls 0
        return pts, np.empty((0, pts.shape[1]), np.float32), []
    
    if DEBUG:
        print(f"[WALL REMOVAL] Starting with {pts.shape[0]} points, max_walls={repeats}")
    rem = pts.copy(); removed = []; models = []
    
    for iteration in range(repeats):
        if rem.shape[0] < 800: 
            if DEBUG:
                print(f"  Iteration {iteration}: Too few points ({rem.shape[0]}), stopping")
            break
        
        pcd = _o3d_pcd(rem[:, :3])
        try:
            model, inliers = pcd.segment_plane(distance_threshold=ransac_dist, ransac_n=3, num_iterations=max_iter)
            a, b, c, d = model
            if DEBUG:
                print(f"  Iteration {iteration}: Found plane a={a:.3f}, b={b:.3f}, c={c:.3f}, d={d:.3f}")
                print(f"    Inliers: {len(inliers)} ({len(inliers)/rem.shape[0]*100:.1f}%)")
                print(f"    Verticality check: |c|={abs(c):.3f} (need < 0.15 for vertical)")
            
            # Stricter verticality check for walls
            if abs(c) < 0.15 and len(inliers) >= 1500:  # Tighter than before
                inliers = np.asarray(inliers, dtype=int)
                removed.append(rem[inliers])
                models.append((float(a), float(b), float(c), float(d)))
                keep_mask = np.ones(len(rem), dtype=bool)
                keep_mask[inliers] = False
                rem = rem[keep_mask]
                if DEBUG:
                    print(f"    ✓ Removed wall plane, {rem.shape[0]} points remaining")
            else:
                if abs(c) >= 0.15:
                    if DEBUG:
                        print(f"    ✗ Plane too horizontal (|c|={abs(c):.3f}), stopping")
                else:
                    if DEBUG:
                        print(f"    ✗ Too few inliers ({len(inliers)}), stopping")
                break
        except Exception as e:
            if DEBUG:
                print(f"  Iteration {iteration}: RANSAC failed - {e}")
            break
    
    removed_total = np.concatenate(removed, axis=0) if removed else np.empty((0, pts.shape[1]), np.float32)
    if DEBUG:
        print(f"[WALL REMOVAL] Complete: Removed {removed_total.shape[0]} points in {len(models)} planes")
    return rem, removed_total, models

def clear_corner_intersections(pts: np.ndarray, models, band_single=0.06, band_corner=0.10):
    """
    Remove points that are simultaneously close to >=2 detected planes (the wall–wall 'gutter').
    band_single: how close counts as 'near a plane'
    band_corner: distance band to BOTH planes to be removed (slightly larger than band_single)
    """
    if pts.size == 0 or not models or len(models) < 2:
        return pts

    xyz = pts[:, :3]
    P = xyz.shape[0]
    # distances to each plane -> shape (P, n_planes)
    dists = []
    for (a,b,c,d) in models:
        n = np.array([a,b,c], dtype=np.float64)
        dn = np.linalg.norm(n) + 1e-12
        d = np.abs(xyz @ n + d) / dn
        dists.append(d)
    D = np.stack(dists, axis=1)

    # near how many planes?
    near_counts = (D <= band_single).sum(axis=1)

    # remove if near >=2 planes (corner gutter)
    # also slightly expand the margin by using band_corner on the top-2 closest planes
    top2 = np.sort(D, axis=1)[:, :2] if D.shape[1] >= 2 else D
    near_both_corner = (top2[:, 0] <= band_corner) & (top2[:, 1] <= band_corner)

    kill = (near_counts >= 2) | near_both_corner
    keep = ~kill
    return pts[keep]

def dbscan_clusters(pts: np.ndarray, eps=0.25, min_samples=60):
    if pts.size == 0: return []
    labels = DBSCAN(eps=eps, min_samples=min_samples).fit(pts[:, :3]).labels_
    clusters = []
    for lbl in sorted(set(labels)):
        if lbl == -1: continue
        idx = np.where(labels == lbl)[0]
        clusters.append(pts[idx])
    return clusters

def bbox_dims(pts: np.ndarray):
    m = pts[:, :3].min(0); M = pts[:, :3].max(0)
    return (M - m)

def is_person_like(pts: np.ndarray, min_points=200,
                   h=(1.2, 2.2), w=(0.2, 1.2), d=(0.2, 1.2)):
    if pts.shape[0] < min_points: return False
    dx, dy, dz = bbox_dims(pts)
    H = dz
    W = dx
    D = dy
    '''
    H = max(dx, dy, dz)             # tallest axis ~ height
    dims = sorted([dx, dy, dz])     # remaining two as width/depth
    W, D, _ = dims[0], dims[1], dims[2]
    '''
    return (h[0] <= H <= h[1]) and (w[0] <= W <= w[1]) and (d[0] <= D <= d[1])

def detect_corner_pattern_advanced(pts: np.ndarray) -> tuple:
    """
    Advanced corner detection using multiple geometric features.
    Returns (is_corner, confidence, reason)
    """
    if pts.shape[0] < 20:
        return False, 0.0, "too_few_points"
    
    xyz = pts[:, :3]
    
    # 1. Get PCA metrics
    lin, pla, sph = pca_metrics(pts)
    
    # 2. Get bounding box dimensions
    dx, dy, dz = bbox_dims(pts)
    sorted_dims = sorted([dx, dy, dz])
    
    # 3. Calculate aspect ratios
    if sorted_dims[2] > 1e-6:
        thin_ratio = sorted_dims[0] / sorted_dims[2]  # Thinnest vs longest
        medium_ratio = sorted_dims[1] / sorted_dims[2]  # Medium vs longest
        aspect_ratio = sorted_dims[2] / (sorted_dims[0] + 1e-6)
    else:
        return False, 0.0, "zero_size"
    
    # 4. Analyze point distribution along principal axes
    mean = xyz.mean(axis=0)
    centered = xyz - mean
    cov = np.cov(centered.T)
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Project points onto principal axes
    projected = centered @ eigenvectors
    
    # Check for L-shape pattern: bimodal distribution on main axes
    hist_main, _ = np.histogram(projected[:, 2], bins=10)  # Main axis
    hist_second, _ = np.histogram(projected[:, 1], bins=10)  # Second axis
    
    # Normalize histograms
    hist_main = hist_main / (hist_main.sum() + 1e-6)
    hist_second = hist_second / (hist_second.sum() + 1e-6)
    
    # Check for concentration at edges (corner pattern)
    edge_concentration = (hist_main[0] + hist_main[-1] + hist_second[0] + hist_second[-1]) / 4
    
    # 5. Calculate "corner score" based on multiple criteria
    corner_score = 0.0
    reasons = []
    
    # Very high linearity is a strong corner indicator
    if lin > 0.8:
        corner_score += 0.4
        reasons.append(f"very_linear({lin:.2f})")
    elif lin > 0.75:
        corner_score += 0.2
        reasons.append(f"linear({lin:.2f})")
    
    # High planarity with low sphericity = wall/corner
    if pla > 0.7 and sph < 0.05:
        corner_score += 0.3
        reasons.append(f"planar({pla:.2f})")
    
    # Extreme aspect ratio
    if aspect_ratio > 8:
        corner_score += 0.3
        reasons.append(f"extreme_aspect({aspect_ratio:.1f})")
    elif aspect_ratio > 5:
        corner_score += 0.15
        reasons.append(f"high_aspect({aspect_ratio:.1f})")
    
    # Very thin in one dimension
    if sorted_dims[0] < 0.15 and sorted_dims[2] > 1.0:
        corner_score += 0.25
        reasons.append(f"thin_edge({sorted_dims[0]:.2f}m)")
    
    # Edge concentration pattern
    if edge_concentration > 0.4:
        corner_score += 0.2
        reasons.append(f"edge_pattern({edge_concentration:.2f})")
    
    # L-shape detection: high planarity but distributed in two directions
    if pla > 0.6 and medium_ratio > 0.3 and thin_ratio < 0.2:
        corner_score += 0.25
        reasons.append("L_shape")
    
    # 6. Human anti-patterns (reduce score if human-like)
    # Humans have moderate sphericity
    if 0.15 <= sph <= 0.4:
        corner_score -= 0.15
        reasons.append(f"human_sphericity({sph:.2f})")
    
    # Human-like aspect ratios
    if 2 <= aspect_ratio <= 4:
        corner_score -= 0.1
        reasons.append(f"human_aspect({aspect_ratio:.1f})")
    
    # Human-like dimensions
    h = dz
    if 1.2 <= h <= 2.0 and sorted_dims[0] > 0.3:
        corner_score -= 0.2
        reasons.append("human_dims")
    
    # Determine if it's a corner
    is_corner = corner_score > 0.35  # Threshold for corner detection
    
    return is_corner, corner_score, "_".join(reasons) if reasons else "none"

def clear_margin_from_planes(pts: np.ndarray, models, margin=0.04):
    """Remove points within 'margin' meters of any plane (a,b,c,d)."""
    if pts.size == 0 or not models:
        return pts
    xyz = pts[:, :3]
    keep = np.ones(xyz.shape[0], dtype=bool)
    for (a,b,c,d) in models:
        n = np.array([a,b,c], dtype=np.float64)
        dist = np.abs(xyz @ n + d) / (np.linalg.norm(n) + 1e-12)
        keep &= (dist > margin)
    return pts[keep]

def clean_person_cluster(pts: np.ndarray, models, margin, nb, std):
    pts = clear_margin_from_planes(pts, models, margin=margin)
    # optional statistical outlier removal (only if Open3D available)
    try:
        if '_O3D' in globals() and _O3D and nb > 0 and pts.shape[0] >= nb:
            pcd = _o3d_pcd(pts[:, :3])
            _, idx = pcd.remove_statistical_outlier(nb_neighbors=nb, std_ratio=std)
            pts = pts[np.asarray(idx)]
    except Exception:
        pass
    return pts

def pca_metrics(pts: np.ndarray):
    xyz = pts[:, :3].astype(np.float64)
    if xyz.shape[0] < 10:
        return 0.0, 0.0, 1.0
    mu = xyz.mean(axis=0, keepdims=True)
    cov = np.cov((xyz - mu).T)
    w = np.linalg.eigvalsh(cov)  # ascending
    l1, l2, l3 = w[2], w[1], w[0] + 1e-12
    linearity  = (l1 - l2) / l1
    planarity  = (l2 - l3) / l1
    sphericity =  l3 / l1
    return float(linearity), float(planarity), float(sphericity)

def xy_circularity(pts):
    xy = pts[:, :2].astype(np.float64)
    if xy.shape[0] < 10: return 0.0
    mu = xy.mean(0, keepdims=True)
    cov = np.cov((xy - mu).T)
    w = np.linalg.eigvalsh(cov)  # [lam_min, lam_max]
    lam_min, lam_max = w[0] + 1e-12, w[1] + 1e-12
    return float(lam_min / lam_max)  # 0=line, 1=circle

def vertical_occupancy(pts, bin=0.10, min_pts=3):
    z = pts[:,2]
    h = float(z.max() - z.min())
    if h < 0.6: return 0.0
    nb = max(1, int(np.ceil(h/bin)))
    hist, _ = np.histogram(z, bins=nb)
    filled = (hist >= min_pts).sum()
    return filled / nb

def is_not_wall_like(pts: np.ndarray, planarity_max=0.8, sphericity_min=0.02):
    _, pla, sph = pca_metrics(pts)
    # A wall has very high planarity and very low sphericity
    return not (pla > planarity_max and sph < sphericity_min)

def near_wall_fraction(pts: np.ndarray, models, band=0.06) -> float:
    if pts.size == 0 or not models:
        return 0.0
    xyz = pts[:, :3]
    n_close = 0
    for (a,b,c,d) in models:
        n = np.array([a,b,c], dtype=np.float64)
        dist = np.abs(xyz @ n + d) / (np.linalg.norm(n) + 1e-12)
        n_close += np.count_nonzero(dist <= band)
    # multiple planes double-count; clamp at total N
    n_close = min(n_close, xyz.shape[0])
    return float(n_close) / float(xyz.shape[0] + 1e-12)

def planes_overlap_count(pts: np.ndarray, models, band=0.08, frac=0.20):
    if pts.size == 0 or not models: return 0
    xyz = pts[:, :3]
    overlaps = 0
    for (a,b,c,d) in models:
        n = np.array([a,b,c], dtype=np.float64); dn = np.linalg.norm(n) + 1e-12
        dist = np.abs(xyz @ n + d) / dn
        if (dist <= band).mean() >= frac:  # >=20% of points near the plane
            overlaps += 1
    return overlaps

def within_range_with_slack(x, lo, hi, slack=0.10):
    lo2 = lo * (1.0 - slack)
    hi2 = hi * (1.0 + slack)
    return (lo2 <= x <= hi2)

def violates_xy_radius(pts: np.ndarray, max_radius: float, pct: float = 100.0) -> tuple[bool, str]:
    """
    Returns (True, reason) if the cluster's XY *spread* exceeds max_radius.
    Spread = max (or percentile) distance of points from the cluster's *centroid* in XY.
    - max_radius <= 0 disables the check.
    - pct in [50..100]: 100 uses max; e.g., 95 uses 95th percentile (robust to a few outliers).
    """
    if not max_radius or max_radius <= 0.0 or pts.size == 0:
        return False, ""
    xy = pts[:, :2].astype(np.float64)
    cx, cy = xy.mean(axis=0)
    r = np.sqrt((xy[:, 0] - cx) ** 2 + (xy[:, 1] - cy) ** 2)
    if pct >= 100.0:
        r_stat = float(r.max())
        tag = "max"
    else:
        pct = float(np.clip(pct, 50.0, 100.0))
        r_stat = float(np.percentile(r, pct))
        tag = f"P{int(pct)}"
    if r_stat > max_radius:
        return True, f"{tag}_spread_r={r_stat:.2f} > {max_radius:.2f}"
    return False, ""


def cluster_score_aggressive(pts: np.ndarray,
                  models,
                  h_rng=(0.8, 2.2), w_rng=(0.25, 1.25), d_rng=(0.25, 1.35),
                  planarity_max=0.8, sphericity_min=0.02,
                  near_wall_band=0.06, near_wall_frac_max=0.4,
                  max_person_points: int = 0,
                  max_person_penalty: float = 2.0,
                  max_person_penalty_shape: str = "log"):

    """
    SOFT scoring: no hard rejects; risky patterns add penalties.
    Returns (ok=True, score, features_dict).
    """
    import math

    # ---- features ----
    n = pts.shape[0]
    dx, dy, dz = bbox_dims(pts)
    H, W, D = dz, dx, dy

    lin, pla, sph = pca_metrics(pts)
    fwall = near_wall_fraction(pts, models, band=near_wall_band)
    circ  = xy_circularity(pts)
    occ   = vertical_occupancy(pts)
    nplanes = planes_overlap_count(pts, models, band=0.08, frac=0.20)

    aspect_ratio      = H / (W + 1e-6)
    width_depth_ratio = W / (D + 1e-6)

    is_corner, corner_confidence, corner_reason = detect_corner_pattern_advanced(pts)

    # ---- base score ----
    score = math.log(max(n, 1))  # more points = better baseline

    # ---- size penalty if exceeding soft cap ----
    max_over = 0
    size_pen = 0.0
    if max_person_points and n > max_person_points:
        max_over = n - max_person_points
        frac = max_over / float(max_person_points)

        if max_person_penalty_shape == "linear":
            grow = frac                      # linear in overflow fraction
        elif max_person_penalty_shape == "sqrt":
            grow = math.sqrt(frac + 1e-12)   # gentler growth
        else:  # "log"
            grow = math.log1p(max_over)      # log in absolute overflow

        size_pen = max_person_penalty * grow
        score -= size_pen

    # Heavy penalty for corner confidence (kept from your original)
    score -= 5.0 * corner_confidence

    # ---- dimensions: bonus if inside, graded penalty if outside ----
    inside = (h_rng[0] <= H <= h_rng[1]) and (w_rng[0] <= W <= w_rng[1]) and (d_rng[0] <= D <= d_rng[1])
    if inside:
        score += 3.0
    else:
        h_penalty = max(0, h_rng[0] - H, H - h_rng[1]) * 1.0
        w_penalty = max(0, w_rng[0] - W, W - w_rng[1]) * 1.0
        d_penalty = max(0, d_rng[0] - D, D - d_rng[1]) * 1.0
        score -= (h_penalty + w_penalty + d_penalty)

    # ---- sphericity (favor moderate volumetric) ----
    if 0.02 <= sph <= 0.10:
        score += 1.5
    elif sph < 0.02:
        score -= 0.5
    elif sph <= 0.25:
        score += 0.5
    else:
        score -= 0.5

    # ---- linearity (penalize very line-like) ----
    if lin < 0.40:
        score += 1.0
    elif lin <= 0.70:
        score -= 1.5 * (lin - 0.40)
    else:
        score -= 2.0 - 1.2 * (0.70 / max(lin, 1e-6))

    # ---- XY circularity (prefer roundish cross-section over razor edges) ----
    if circ >= 0.15:
        score += 1.0 * min(circ / 0.5, 1.0)
    else:
        score -= (0.15 - circ) * 3.0

    # ---- vertical occupancy (prefer continuous Z fill) ----
    if occ >= 0.60:
        score += 1.5
    elif occ >= 0.40:
        score += 0.5
    else:
        score -= 1.0

    # ---- planarity penalty ----
    if pla > 0.6:
        score -= 2.0 * (pla - 0.6)

    # ---- wall proximity penalty ----
    score -= 2.0 * fwall

    # ---- aspect ratios ----
    if 2.0 <= aspect_ratio <= 5.0:
        score += 1.0
    elif aspect_ratio > 6.0:
        score -= 3.0

    if 0.3 <= width_depth_ratio <= 3.0:
        score += 0.5

    # ---- planes overlap (corner gutter heuristic) ----
    if nplanes >= 2:
        score -= 4.0
    elif nplanes == 1:
        score -= 1.0

    # ---- previously HARD rejects -> now SOFT penalties ----
    score_pen = 0.0
    if is_corner or corner_confidence > 0.3:
        score_pen += 3.5
    if (pla > planarity_max and sph < sphericity_min):
        score_pen += 3.0
    if fwall > near_wall_frac_max:
        score_pen += 2.5
    if aspect_ratio > 7.0:
        score_pen += 3.0

    score -= score_pen

    # Always OK; ranking decides
    ok = True

    return ok, score, {
        "n": n, "H": float(H), "W": float(W), "D": float(D),
        "lin": float(lin), "pla": float(pla), "sph": float(sph),
        "fwall": float(fwall),
        "aspect_ratio": float(aspect_ratio),
        "xy_circ": float(circ),
        "z_occ": float(occ),
        "is_corner": bool(is_corner),
        "corner_conf": float(corner_confidence),
        "corner_reason": corner_reason,
        "nplanes": int(nplanes),
        "score_pen": float(score_pen),
        "maxpts_cap": int(max_person_points),
        "maxpts_over": int(max_over),
        "maxpts_pen": float(size_pen),
        "maxpts_shape": max_person_penalty_shape,
    }



# ---------- Naming & parsing ----------

def lreid_name(pid: int, cam: int, tid: int, fid: int) -> str:
    # 0001C1T0000F000.bin
    return f"{pid:04d}C{cam:d}T{tid:04d}F{fid:03d}.bin"

_digit_re = re.compile(r"(\d+)")
_frame_re = re.compile(r"[Ff]rame[:\s-]*(\d+)", re.IGNORECASE)  # Add this

def _seq_key_from_parts(parts: Tuple[str, ...]) -> str:
    # parts = (split, pid, class, [optional deeper folders], file.csv)
    return str(Path(*parts[3:-1])) if len(parts) > 4 else "_root_"

def natural_frame_sort_key(p: Path, csv_root: Path) -> Tuple[str, str, str, str, int, str]:
    """
    Sort key that keeps frames in true chronological/numeric order
    within each (split, pid, class, subseq) group.
    """
    rel = p.relative_to(csv_root)
    parts = rel.parts
    # Guard: some odd files could slip in
    split = parts[0] if len(parts) > 0 else ""
    pid_folder = parts[1] if len(parts) > 1 else ""
    cls = parts[2] if len(parts) > 2 else ""
    seq_key = _seq_key_from_parts(parts)
    stem = Path(parts[-1]).stem if len(parts) else ""
    frame = parse_frame_from_name(stem, default_idx=0)
    return (split, pid_folder, cls, seq_key, frame, stem)

def parse_pid_from_folder(name: str, fallback_id: int) -> int:
    m = _digit_re.search(name)
    if m: return int(m.group(1)) % 10000 or 1
    return fallback_id

def parse_frame_from_name(stem: str, default_idx: int) -> int:
    """Extract frame number from filename, looking for 'Frame N' pattern first."""
    # Try to find "Frame N" pattern first
    m = _frame_re.search(stem)
    if m:
        return int(m.group(1))
    
    # Fallback: find last digit sequence (more likely to be frame number than first)
    matches = _digit_re.findall(stem)
    if matches:
        return int(matches[-1])
    
    return int(default_idx)

def build_track_ranges(paths: List[str], info_rows: List[List[int]]) -> np.ndarray:
    """
    Convert per-frame rows [pid, cam, tid, fid] aligned to 'paths' into
    LReID-style per-tracklet ranges: [start_idx, end_idx, pid, cam].
    Ranges are contiguous segments where (pid, cam, tid) are constant.
    """
    ranges = []
    if not paths:
        return np.zeros((0, 4), dtype=np.int32)

    cur_key = None  # (pid, cam, tid)
    start = None
    for i, (pid_i, cam_i, tid_i, fid_i) in enumerate(info_rows):
        key = (int(pid_i), int(cam_i), int(tid_i))
        if cur_key is None:
            cur_key = key
            start = i
        elif key != cur_key:
            # close previous track
            ranges.append([start, i - 1, cur_key[0], cur_key[1]])
            cur_key = key
            start = i

    # close last one
    if cur_key is not None:
        ranges.append([start, len(paths) - 1, cur_key[0], cur_key[1]])

    return np.asarray(ranges, dtype=np.int32)

# ---------- Main ----------

def main():
    ap = argparse.ArgumentParser(description="Overfit structure → stepwise BINs + ReID3D-ready export")
    ap.add_argument("--csv-root", required=True, help="Root: overfit_data/")
    ap.add_argument("--out-root", required=True, help="Output root directory")
    # CSV schema (defaults match your Velodyne CSVs)
    ap.add_argument("--x-col", default="Points_m_XYZ:0")
    ap.add_argument("--y-col", default="Points_m_XYZ:1")
    ap.add_argument("--z-col", default="Points_m_XYZ:2")
    ap.add_argument("--intensity-col", default=None, help="Set and pass --include-intensity to write XYZI")
    ap.add_argument("--include-intensity", action="store_true", help="(ignored) pipeline exports XYZ only")
    ap.add_argument("--delimiter", default=",")
    ap.add_argument("--csv-has-header", action="store_true", default=True)
    # Distance clamp
    ap.add_argument("--max-dist", type=float, default=30.0)
    # Planes
    ap.add_argument("--ground-dist", type=float, default=0.03)
    ap.add_argument("--wall-dist", type=float, default=0.03)
    ap.add_argument("--max-walls", type=int, default=3)
    # Clustering / person filter
    ap.add_argument("--dbscan-eps", type=float, default=0.25)
    ap.add_argument("--dbscan-min-samples", type=int, default=60)
    ap.add_argument("--min-person-points", type=int, default=200)
    ap.add_argument("--person-h", type=float, nargs=2, default=[1.2, 2.2])
    ap.add_argument("--person-w", type=float, nargs=2, default=[0.2, 1.2])
    ap.add_argument("--person-d", type=float, nargs=2, default=[0.2, 1.2])
    ap.add_argument("--largest-if-none", action="store_true",
                help="Fallback: export largest cluster if no person-like cluster found")
    # Split mapping (your tree has train/val/test; ReID3D uses train/query/gallery)
    ap.add_argument("--map-val-to", choices=["val","query","gallery"], default="query")
    ap.add_argument("--map-test-to", choices=["test","query","gallery"], default="gallery")
    # Constants
    ap.add_argument("--camera-id", type=int, default=1,
                help="Use 1 unless you truly have multi-LiDAR setup")

    # Additional flags
    ap.add_argument("--keep-top1", action="store_true",
                help="Always include the largest cluster per frame as a candidate")
    ap.add_argument("--max-person-per-frame", type=int, default=1,
                help="Export at most N clusters per frame (after gating)")
    ap.add_argument("--intra-clearance", type=float, default=0.0,
                help="Trim points within this distance (m) of detected wall planes (0=off)")
    ap.add_argument("--intra-outlier-nb", type=int, default=16,
                help="Statistical outlier removal: neighbors")
    ap.add_argument("--intra-outlier-std", type=float, default=2.0,
                help="Statistical outlier removal: std ratio")
    ap.add_argument("--near-wall-band", type=float, default=0.06,
                help="Band (m) around detected wall planes for near-wall fraction")
    ap.add_argument("--near-wall-frac-max", type=float, default=0.4,
                help="Reject clusters if fraction of points within band exceeds this")
    ap.add_argument("--planarity-max", type=float, default=0.8,
                help="Reject clusters with planarity above this AND sphericity below sphericity-min")
    ap.add_argument("--sphericity-min", type=float, default=0.02,
                help="Sphericity threshold paired with planarity-max")
    ap.add_argument("--use-scoring", action="store_true",
                help="Score clusters and pick the top scoring ones instead of size/first-pass gates")
    ap.add_argument("--export-topk", type=int, default=0,
                help="If >0, export the top-K largest clusters per frame regardless of gates/scoring")
    ap.add_argument("--include-classes", nargs="+", default=None,
                help="Only process these class labels (e.g., 0 1 3). If omitted, all classes are processed.")
    ap.add_argument("--exclude-classes", nargs="+", default=None,
                help="Skip these class labels (e.g., 2 4 6). Applied after include-classes if both are given.")
    ap.add_argument("--force-split", choices=["none", "train_test"], default="none",
                help="Override the source split folders and reassign persons by identity.")
    ap.add_argument("--train-persons", type=int, default=21)
    ap.add_argument("--test-persons",  type=int, default=9)
    ap.add_argument("--split-seed",    type=int, default=0)
    ap.add_argument("--preclear-margin", type=float, default=0.0,
                help="Global plane clearance before clustering. Set 0 to disable.")
    ap.add_argument("--verbose", action="store_true", 
                help="Print detailed debug information during processing")
    ap.add_argument("--corner-threshold", type=float, default=0.35,
                help="Threshold for corner detection confidence (lower = more aggressive)")
    ap.add_argument("--dims-strong-gate", action="store_true",
                help="Hard-reject clusters that are outside H/W/D by a margin")
    ap.add_argument("--dims-gate-slack", type=float, default=0.10,
                help="Allowed fractional slack beyond each bound (e.g., 0.10 = 10%)")
    ap.add_argument("--max-xy-radius", type=float, default=0.0,
                help="Reject clusters whose XY *spread* about their own centroid exceeds this radius (m). 0=off")
    ap.add_argument("--max-person-points", type=int, default=0,
                help="Soft cap for scoring: if a cluster has more points than this, it gets penalized. 0 = no penalty.")
    ap.add_argument("--max-person-penalty", type=float, default=2.0,
                help="Penalty strength when cluster size exceeds --max-person-points.")
    ap.add_argument("--max-person-penalty-shape", choices=["log", "linear", "sqrt"], default="log",
                help="How the penalty grows with the amount of overflow.")

    #ADDED FOR DEBUGGING
    ap.add_argument("--xy-radius-percentile", type=float, default=100.0,
                help="Percentile [50..100] of XY radius spread to use (100 = max).")

    # --- NEW: frame range selection (by frame number extracted from filename) ---
    ap.add_argument("--frame-start", type=int, default=None,
                help="Inclusive starting frame number to process (e.g., 100 means skip frames < 100).")
    ap.add_argument("--frame-end", type=int, default=None,
                help="Inclusive ending frame number to process (e.g., 300 means skip frames > 300).")


    args = ap.parse_args()

    # Sanity check for frame range
    if args.frame_start is not None and args.frame_end is not None:
        if args.frame_start > args.frame_end:
            raise SystemExit(f"--frame-start ({args.frame_start}) must be <= --frame-end ({args.frame_end})")


    csv_root = Path(args.csv_root)
    out_root = Path(args.out_root)
    dbg_root = out_root / "debug_steps"
    reid_root = out_root / "reid3d"

    # === NEW: ReID3D dataset builders ===
    # Lists of relative paths (relative to reid_root) and aligned info rows [pid, cam, tid, fid]
    train_paths: List[str] = []
    test_paths:  List[str] = []
    train_info_rows: List[List[int]] = []
    test_info_rows:  List[List[int]] = []

    # We'll mark which test paths are queries; later we convert them to indices into test_paths
    query_relpaths: List[str] = []

    # Normalize class filters to string sets like {"0","1","5"}
    inc_classes = set(str(c).strip() for c in (args.include_classes or []))
    exc_classes = set(str(c).strip() for c in (args.exclude_classes or []))

    # Collect all CSVs
    csv_files = [Path(p) for p in glob.glob(str(csv_root / "**/*.csv"), recursive=True)]
    if not csv_files:
        raise SystemExit("No CSVs found under --csv-root")

    def class_allowed(cls_name: str) -> bool:
        cs = str(cls_name).strip()
        if inc_classes and (cs not in inc_classes):
            return False
        if exc_classes and (cs in exc_classes):
            return False
        return True

    # Pre-filter by class for any logic that depends on person coverage (forced split)
    csv_files_for_split = []
    for p in csv_files:
        try:
            relp = p.relative_to(csv_root)
        except ValueError:
            continue
        parts = relp.parts
        if len(parts) < 4:
            continue
        src_split, pid_folder, cls_folder = parts[0], parts[1], parts[2]
        if src_split not in ("train","val","test"):
            continue
        if not class_allowed(cls_folder):
            continue
        csv_files_for_split.append(p)

    # ---- Optional identity-based split override (built on class-filtered files) ----
    forced_split_by_pid = None
    if args.force_split == "train_test":
        pid_folders = []
        for p in csv_files_for_split:
            relp = p.relative_to(csv_root)
            pid_folders.append(relp.parts[1])
        unique_pids = sorted(set(pid_folders))

        need = args.train_persons + args.test_persons
        if len(unique_pids) < need:
            raise SystemExit(f"Not enough identities after class filtering: have {len(unique_pids)}, need {need}")


        rng = random.Random(args.split_seed)
        pids_shuffled = unique_pids[:]
        rng.shuffle(pids_shuffled)

        train_pids = set(pids_shuffled[:args.train_persons])
        test_pids  = set(pids_shuffled[args.train_persons:args.train_persons+args.test_persons])

        forced_split_by_pid = {pid: ("train" if pid in train_pids else "test" if pid in test_pids else None)
                            for pid in unique_pids}
        print(f"[SPLIT] Forced split: train={len(train_pids)} IDs, test={len(test_pids)} IDs (seed={args.split_seed})")

    # State
    pid_map: Dict[str,int] = {}                                     # folder pid -> int
    next_pid = 1
    #tid_map: Dict[Tuple[str,int,str,str], int] = {}                  # (split,pid,class,seq_key) -> tid
    #next_tid_for_key: Dict[Tuple[str,int], int] = {}                 # (split,pid) -> next tid
    #frame_counters: Dict[Tuple[str,int,int], int] = {}               # (split,pid,tid) -> next fid
    # NEW: per-(split,pid) next free TID, and per-sequence (split,pid,cls,seq_key) state
    next_tid_for_pid: Dict[Tuple[str,int], int] = {}  # (split,pid) -> next free tid
    seq_state: Dict[Tuple[str,int,str,str], Dict[str,int]] = {}  # -> {"base_tid": int, "frame_idx": int}
    manifest: List[dict] = []

    # Split mapping
    def map_split(split: str) -> str:
        if split == "train": return "train"
        if split == "val":   return args.map_val_to
        if split == "test":  return args.map_test_to
        return split  # fallback

    ordered_files = sorted(csv_files, key=lambda p: natural_frame_sort_key(p, csv_root))

    for idx, csv_path in enumerate(ordered_files):
        try:
            rel = csv_path.relative_to(csv_root)
        except ValueError:
            # skip odd paths
            continue

        parts = rel.parts
        if len(parts) < 4:
            # need at least split/pid/class/file.csv (allow deeper after class)
            continue

        split, pid_folder, cls = parts[0], parts[1], parts[2]
        if split not in ("train","val","test"):
            continue
        
        # ---- Class filtering ----
        if not class_allowed(cls):
            continue

        # ---- Forced identity split (if enabled) ----
        if forced_split_by_pid is not None:
            forced = forced_split_by_pid.get(pid_folder, None)
            if forced is None:
                # PID not selected into the 21/9 quota -> skip entirely
                continue
            split = forced  # override source split based on PID


        sub_after_class = Path(*parts[3:-1]) if len(parts) > 4 else Path()  # e.g., seqA
        stem = Path(parts[-1]).stem

        # PID
        if pid_folder not in pid_map:
            pid_map[pid_folder] = parse_pid_from_folder(pid_folder, next_pid)
            next_pid += 1
        pid = pid_map[pid_folder]

        # Tracklet allocation per (split,pid,cls,seq_key)
        seq_key = str(sub_after_class) if str(sub_after_class) else "_root_"
        tid_key = (split, pid, cls, seq_key)
        base_key = (split, pid)

        if base_key not in next_tid_for_pid:
            next_tid_for_pid[base_key] = 0

        if tid_key not in seq_state:
            base_tid = next_tid_for_pid[base_key]
            seq_state[tid_key] = {"base_tid": base_tid, "frame_idx": 0}

        base_tid = seq_state[tid_key]["base_tid"]
        frame_idx = seq_state[tid_key]["frame_idx"]

        # --- NEW: frame-range filter (per sequence) ---
        # Use the "Frame N" pattern if present; otherwise fall back to the local frame_idx.
        frame_num = parse_frame_from_name(stem, default_idx=frame_idx)

        if (args.frame_start is not None and frame_num < args.frame_start) or \
           (args.frame_end   is not None and frame_num > args.frame_end):
            if args.verbose:
                print(f"[SKIP FRAME] {rel} -> frame_num={frame_num} "
                      f"outside [{args.frame_start if args.frame_start is not None else '-inf'}, "
                      f"{args.frame_end if args.frame_end is not None else '+inf'}]")

            # Important: still advance per-sequence frame counter so chunking/track IDs stay consistent
            seq_state[tid_key]["frame_idx"] = frame_idx + 1
            continue


        # Load CSV
        try:
            pts = read_csv_points(csv_path,
                                xcol=args.x_col, ycol=args.y_col, zcol=args.z_col,
                                icol=args.intensity_col if args.include_intensity else None,
                                delimiter=args.delimiter, has_header=args.csv_has_header)
        except Exception as e:
            print(f"[ERROR] Failed to read {csv_path.name}: {e}")
            continue  # Skip to next file
        
        # Sanity check
        if pts.shape[0] == 0:
            print(f"[WARN] {csv_path.name} has zero points, skipping")
            continue
        
        #debug prints optional
        if args.verbose:
            print(f"\n[PROCESSING] {rel}")
            print(f"  Shape: {pts.shape}")
            print(f"  X range: [{pts[:,0].min():.3f}, {pts[:,0].max():.3f}]")
            print(f"  Y range: [{pts[:,1].min():.3f}, {pts[:,1].max():.3f}]")
            print(f"  Z range: [{pts[:,2].min():.3f}, {pts[:,2].max():.3f}]")

        # Step A: raw
        A = pts
        if DEBUG:
            A_dir = dbg_root / "A_raw_bin" / split / pid_folder / cls / sub_after_class
            write_bin(A_dir / f"{stem}.bin", A, include_intensity=args.include_intensity)

        # Step B: distance-clipped
        B_keep, B_drop = clip_distance(A, args.max_dist)
        if DEBUG:
            B_dir = dbg_root / "B_clipped_bin" / split / pid_folder / cls / sub_after_class
            write_bin(B_dir / f"{stem}.bin", B_keep, include_intensity=args.include_intensity)
            write_bin(dbg_root / "B_clipped_removed_bin" / split / pid_folder / cls / sub_after_class / f"{stem}.bin",
                    B_drop, include_intensity=args.include_intensity)
                  
        # Step C: ground removal
        C_keep, C_ground = remove_ground(B_keep, ransac_dist=args.ground_dist)
        if DEBUG:
            C_dir = dbg_root / "C_no_ground_bin" / split / pid_folder / cls / sub_after_class
            write_bin(C_dir / f"{stem}.bin", C_keep, include_intensity=args.include_intensity)
            write_bin(dbg_root / "C_ground_removed_bin" / split / pid_folder / cls / sub_after_class / f"{stem}.bin",
                    C_ground, include_intensity=args.include_intensity)

        # Step D: walls removal
        wall_ret = remove_walls_iter(C_keep, ransac_dist=args.wall_dist, repeats=args.max_walls)
        if isinstance(wall_ret, tuple) and len(wall_ret) == 3:
            D_keep, D_walls, D_models = wall_ret
        else:
            D_keep, D_walls = wall_ret
            D_models = []   # no plane models available

        if DEBUG:
            D_dir = dbg_root / "D_nowalls_bin" / split / pid_folder / cls / sub_after_class
            write_bin(D_dir / f"{stem}.bin", D_keep, include_intensity=args.include_intensity)
            write_bin(dbg_root / "D_walls_removed_bin" / split / pid_folder / cls / sub_after_class / f"{stem}.bin",
                    D_walls, include_intensity=args.include_intensity)

        # ---- D2: optional global plane clearance BEFORE clustering ----
        if args.preclear_margin > 0.0 and D_models:
            D_keep = clear_margin_from_planes(D_keep, D_models, margin=args.preclear_margin)
            if DEBUG:
                write_bin(dbg_root / "D2_nowalls_cleared_bin" / split / pid_folder / cls / sub_after_class / f"{stem}.bin",
                        D_keep, include_intensity=args.include_intensity)

        # NEW: explicit corner-gutter clearance (back-left corner killer)
        if D_models and len(D_models) >= 2:
            D_keep = clear_corner_intersections(D_keep, D_models, band_single=0.06, band_corner=0.10)
            if DEBUG:
                write_bin(dbg_root / "D3_clear_corner_bin" / split / pid_folder / cls / sub_after_class / f"{stem}.bin",
                        D_keep, include_intensity=args.include_intensity)

        # Step E: clustering
        clusters = dbscan_clusters(D_keep, eps=args.dbscan_eps, min_samples=args.dbscan_min_samples)

        # Aggressive filtering with advanced corner detection
        # Aggressive filtering with advanced corner detection
        filtered_clusters = []
        for ci, c in enumerate(clusters):
            # Check if wall-like
            if not is_not_wall_like(c, planarity_max=args.planarity_max, sphericity_min=args.sphericity_min):
                if args.verbose:
                    _, pla, sph = pca_metrics(c)
                    print(f"  Rejected wall: N={c.shape[0]} pla={pla:.3f} sph={sph:.3f}")
                # dump rejected cluster
                try:
                    if DEBUG:
                        rej_dir = dbg_root / "E_reject_walllike_bin" / split / pid_folder / cls / sub_after_class / stem
                        write_bin(rej_dir / f"reject_{ci:04d}.bin", c, include_intensity=args.include_intensity)
                except Exception:
                    pass
                continue
            
            # Advanced corner detection
            is_corner, corner_conf, corner_reason = detect_corner_pattern_advanced(c)
            if is_corner or corner_conf > args.corner_threshold:
                if args.verbose:
                    lin, pla, sph = pca_metrics(c)
                    print(f"  Rejected corner: N={c.shape[0]} conf={corner_conf:.3f} reason={corner_reason}")
                # dump rejected cluster
                try:
                    if DEBUG:
                        rej_dir = dbg_root / "E_reject_corner_bin" / split / pid_folder / cls / sub_after_class / stem
                        write_bin(rej_dir / f"reject_{ci:04d}.bin", c, include_intensity=args.include_intensity)
                except Exception:
                    pass
                continue
            
            filtered_clusters.append(c)

        clusters = filtered_clusters

        #Write all clusters to E_clusters_bin
        '''
        E_all_dir = dbg_root / "E_clusters_bin" / split / pid_folder / cls / sub_after_class / stem
        for ci, c in enumerate(clusters):
            write_bin(E_all_dir / f"cluster_{ci:04d}.bin", c, include_intensity=args.include_intensity)
        '''
        # Write all clusters to E_clusters_bin + summary.json
        if DEBUG:
            E_all_dir = dbg_root / "E_clusters_bin" / split / pid_folder / cls / sub_after_class / stem
            E_all_dir.mkdir(parents=True, exist_ok=True)

            e_report = []
            for ci, c in enumerate(clusters):
                # metrics
                dx, dy, dz = bbox_dims(c)
                H = dz  # Height is always Z
                W = dx
                D = dy 

                lin, pla, sph = pca_metrics(c)
                fwall = near_wall_fraction(c, D_models, band=args.near_wall_band)
                circ = xy_circularity(c)
                occ  = vertical_occupancy(c)
                is_corner, corner_conf, corner_reason = detect_corner_pattern_advanced(c)
                # XY spread (percentile-aware)
                violates_r, why_r = violates_xy_radius(
                    c, max_radius=args.max_xy_radius,
                    pct=getattr(args, "xy_radius_percentile", 100.0)
                )
                # save bin
                write_bin(E_all_dir / f"cluster_{ci:04d}.bin", c, include_intensity=args.include_intensity)
                # append summary row
                e_report.append({
                    "idx": ci, "N": int(c.shape[0]),
                    "dx": float(dx), "dy": float(dy), "dz": float(dz),
                    "H": float(H), "W": float(W), "D": float(D),
                    "lin": float(lin), "pla": float(pla), "sph": float(sph),
                    "near_wall_frac": float(fwall),
                    "xy_circularity": float(circ),
                    "z_occupancy": float(occ),
                    "corner_conf": float(corner_conf),
                    "corner_reason": corner_reason,
                    "xy_spread_violation": bool(violates_r),
                    "xy_spread_why": why_r
                })

        # write per-frame report
        if DEBUG:
            try:
                import json as _json
                (_E := E_all_dir / "summary.json").write_text(_json.dumps(e_report, indent=2))
            except Exception:
                pass

        if args.verbose and clusters:
            print(f"[CLUSTERS] Frame {stem}: {len(clusters)} clusters after filtering")
            for i in range(min(5, len(clusters))):
                c = clusters[i]
                lin, pla, sph = pca_metrics(c)
                dx, dy, dz = bbox_dims(c)
                is_corner, corner_conf, _ = detect_corner_pattern_advanced(c)
                print(f"  Cluster {i}: N={c.shape[0]:4d} dims=({dx:.2f},{dy:.2f},{dz:.2f}) "
                    f"lin={lin:.3f} pla={pla:.3f} sph={sph:.3f} corner_conf={corner_conf:.3f}")

        #check
        sizes = [c.shape[0] for c in clusters]
        #print(f"[DBG] {rel} clusters={len(sizes)} largest={max(sizes) if sizes else 0}")

        # Sort by size to have a stable order (helps debugging)
        order = sorted(range(len(clusters)), key=lambda i: clusters[i].shape[0], reverse=True)

        # --- Fast path: export top-K largest clusters regardless of gates/scoring ---
        scores_dbg = []
        cand = []
        if getattr(args, "export_topk", 0) > 0:
            k = min(args.export_topk, len(order))
            if args.max_person_per_frame > 0:
                k = min(k, args.max_person_per_frame)
            cand = [clusters[i] for i in order[:k]]

        if not cand:
            cand = []
            scores_dbg = []

            # Class-aware gates for wheelchair etc.
            h_rng = tuple(args.person_h); w_rng = tuple(args.person_w); d_rng = tuple(args.person_d)
            cls_name = str(cls).lower()
            if cls_name in ("wheelchair", "wheel chair", "chair", "5"):
                h_rng = (0.8, 1.8); w_rng = (0.35, 1.20); d_rng = (0.35, 1.35)

            if args.use_scoring:
                # Use aggressive scoring function
                scored = []
                for i in order:
                    c = clusters[i]
                    # --- Compute dims once
                    dx, dy, dz = bbox_dims(c)
                    H = dz
                    W = dx
                    D = dy

                    # --- Optional hard dimension gate
                    if args.dims_strong_gate:
                        h_ok = within_range_with_slack(H, h_rng[0], h_rng[1], slack=args.dims_gate_slack)
                        w_ok = within_range_with_slack(W, w_rng[0], w_rng[1], slack=args.dims_gate_slack)
                        d_ok = within_range_with_slack(D, d_rng[0], d_rng[1], slack=args.dims_gate_slack)
                        if not (h_ok and w_ok and d_ok):
                            if args.verbose:
                                print(f"  Rejected by DIM gate: N={c.shape[0]} "
                                    f"H={H:.2f} W={W:.2f} D={D:.2f} "
                                    f"allowed H{h_rng} W{w_rng} D{d_rng} slack={args.dims_gate_slack:.2f}")
                            # Optional: dump rejects for inspection
                            try:
                                if DEBUG:
                                    rej_dir = dbg_root / "E_dims_reject_bin" / split / pid_folder / cls / sub_after_class / stem
                                    write_bin(rej_dir / f"reject_{i:04d}.bin", c, include_intensity=args.include_intensity)
                            except Exception:
                                pass
                            # Skip this cluster entirely
                            continue
                        
                    # --- Optional XY radius gate (keep clusters near LiDAR center)
                    violates_r, why_r = violates_xy_radius(c, max_radius=args.max_xy_radius, pct=getattr(args, "xy_radius_percentile", 100.0))
                    if violates_r:
                        if args.verbose:
                            print(f"  Rejected by XY-RADIUS: N={c.shape[0]}  {why_r}")
                        try:
                            if DEBUG:
                                rej_dir = dbg_root / "E_xy_radius_reject_bin" / split / pid_folder / cls / sub_after_class / stem
                                write_bin(rej_dir / f"reject_{i:04d}.bin", c, include_intensity=args.include_intensity)
                        except Exception:
                            pass
                        continue

                    ok, sc, feat = cluster_score_aggressive(
                        clusters[i], D_models,
                        h_rng=h_rng, w_rng=w_rng, d_rng=d_rng,
                        planarity_max=args.planarity_max,
                        sphericity_min=args.sphericity_min,
                        near_wall_band=args.near_wall_band,
                        near_wall_frac_max=args.near_wall_frac_max,
                        max_person_points=args.max_person_points,
                        max_person_penalty=args.max_person_penalty,
                        max_person_penalty_shape=args.max_person_penalty_shape,
                    )
                    scores_dbg.append({"idx": i, "ok": ok, "score": sc, **feat})
                    if ok:
                        scored.append((sc, clusters[i]))
                
                scored.sort(key=lambda x: x[0], reverse=True)
                
                if scored:
                    # Take top scoring clusters
                    cand = [c for _, c in scored[:max(1, args.max_person_per_frame)]]
                elif order and args.keep_top1:
                    # if everything was rejected but keep_top1, check corner confidence
                    top_cluster = clusters[order[0]]
                    is_corner, corner_conf, _ = detect_corner_pattern_advanced(top_cluster)
                    violates_r, why_r = violates_xy_radius(
                        top_cluster, max_radius=args.max_xy_radius,
                        pct=getattr(args, "xy_radius_percentile", 100.0)
                    )
                    if corner_conf < 0.5 and not violates_r:
                        cand = [top_cluster]
                    elif args.verbose:
                        print(f"  Skip top1 fallback: corner_conf={corner_conf:.3f} violates_xy={violates_r} {why_r}")
                
                if args.verbose and scores_dbg:
                    print(f"  Scoring results (top 5):")
                    for s in sorted(scores_dbg, key=lambda x: x['score'], reverse=True)[:5]:
                        print(f"    Score={s['score']:.2f} ok={s['ok']} N={s['n']} "
                              f"H={s['H']:.2f} lin={s['lin']:.3f} pla={s['pla']:.3f} "
                              f"sph={s['sph']:.3f} corner_conf={s.get('corner_conf', 0):.3f} "
                              f"reason={s.get('corner_reason', 'none')}")
            else:
                # Fallback: your original gate-based path
                for i in order:
                    c = clusters[i]

                    # --- Compute dims once
                    dx, dy, dz = bbox_dims(c)
                    H = dz
                    W = dx
                    D = dy

                    # --- Optional hard dimension gate
                    if args.dims_strong_gate:
                        h_ok = within_range_with_slack(H, h_rng[0], h_rng[1], slack=args.dims_gate_slack)
                        w_ok = within_range_with_slack(W, w_rng[0], w_rng[1], slack=args.dims_gate_slack)
                        d_ok = within_range_with_slack(D, d_rng[0], d_rng[1], slack=args.dims_gate_slack)
                        if not (h_ok and w_ok and d_ok):
                            if args.verbose:
                                print(f"  Rejected by DIM gate: N={c.shape[0]} "
                                    f"H={H:.2f} W={W:.2f} D={D:.2f} "
                                    f"allowed H{h_rng} W{w_rng} D{d_rng} slack={args.dims_gate_slack:.2f}")
                            try:
                                if DEBUG:
                                    rej_dir = dbg_root / "E_dims_reject_bin" / split / pid_folder / cls / sub_after_class / stem
                                    write_bin(rej_dir / f"reject_{i:04d}.bin", c, include_intensity=args.include_intensity)
                            except Exception:
                                pass
                            continue
                    # --- Optional XY radius gate (keep clusters near LiDAR center)
                    violates_r, why_r = violates_xy_radius(c, max_radius=args.max_xy_radius, pct=getattr(args, "xy_radius_percentile", 100.0))
                    if violates_r:
                        if args.verbose:
                            print(f"  Rejected by XY-RADIUS: N={c.shape[0]}  {why_r}")
                        try:
                            if DEBUG:
                                rej_dir = dbg_root / "E_xy_radius_reject_bin" / split / pid_folder / cls / sub_after_class / stem
                                write_bin(rej_dir / f"reject_{i:04d}.bin", c, include_intensity=args.include_intensity)
                        except Exception:
                            pass
                        continue

                    if is_person_like(c, min_points=args.min_person_points, h=h_rng, w=w_rng, d=d_rng):
                        cand.append(c)
                if args.keep_top1 and order and all(clusters[order[0]] is not x for x in cand):
                    # Check corner confidence before adding
                    top_cluster = clusters[order[0]]
                    is_corner, corner_conf, _ = detect_corner_pattern_advanced(top_cluster)
                    violates_r, why_r = violates_xy_radius(
                        top_cluster, max_radius=args.max_xy_radius,
                        pct=getattr(args, "xy_radius_percentile", 100.0)
                    )
                    if corner_conf < 0.5 and not violates_r:
                        cand.append(top_cluster)
                    elif args.verbose:
                        print(f"  Skip top1 candidate: corner_conf={corner_conf:.3f} violates_xy={violates_r} {why_r}")
                if args.max_person_per_frame > 0 and len(cand) > args.max_person_per_frame:
                    cand = sorted(cand, key=lambda a: a.shape[0], reverse=True)[:args.max_person_per_frame]
                if not cand and order and args.keep_top1:
                    top_cluster = clusters[order[0]]
                    is_corner, corner_conf, _ = detect_corner_pattern_advanced(top_cluster)
                    violates_r, why_r = violates_xy_radius(
                        top_cluster, max_radius=args.max_xy_radius,
                        pct=getattr(args, "xy_radius_percentile", 100.0)
                    )
                    if corner_conf < 0.5 and not violates_r:
                        cand = [top_cluster]
                    elif args.verbose:
                        print(f"  Skip last-resort top1: corner_conf={corner_conf:.3f} violates_xy={violates_r} {why_r}")

        # Enforce XY-spread gate on all selected candidates (applies even when top1 + topk=0)
        if args.max_xy_radius > 0.0:
            kept = []
            for i, c in enumerate(cand):
                violates_r, why_r = violates_xy_radius(
                    c, max_radius=args.max_xy_radius,
                    pct=getattr(args, "xy_radius_percentile", 100.0)
                )
                if violates_r:
                    if args.verbose:
                        print(f"  Cand[{i}] dropped by XY-SPREAD: {why_r}")
                    try:
                        if DEBUG:
                            rej_dir = dbg_root / "F_xy_radius_reject_bin" / split / pid_folder / cls / sub_after_class / stem
                            write_bin(rej_dir / f"reject_{i:04d}.bin", c, include_intensity=args.include_intensity)
                    except Exception:
                        pass
                    continue
                kept.append(c)
            cand = kept

        # ---- Universal post-candidate XY-SPREAD filter (applies even for keep_top1/export_topk=0)
        if args.max_xy_radius > 0.0 and cand:
            if DEBUG:
                F_rej_dir = dbg_root / "F_xy_radius_reject_bin" / split / pid_folder / cls / sub_after_class / stem
                F_rej_dir.mkdir(parents=True, exist_ok=True)
            kept = []
            for j, c in enumerate(cand):
                vr, why = violates_xy_radius(c, args.max_xy_radius, pct=getattr(args, "xy_radius_percentile", 100.0))
                if vr:
                    if args.verbose:
                        print(f"  [UNIVERSAL FILTER] Drop candidate #{j}: {why}")
                    if DEBUG:
                        write_bin(F_rej_dir / f"reject_{j:04d}.bin", c, include_intensity=args.include_intensity)
                else:
                    kept.append(c)
            cand = kept

        # ---- Decision file linking selected candidates back to E indices
        selected_indices = []
        for j, c in enumerate(cand):
            hit = next((i for i, ec in enumerate(clusters) if ec is c), None)
            selected_indices.append({"cand_slot": j, "E_idx": hit, "N": int(c.shape[0])})

        try:
            import json as _json
            if DEBUG:
                dec_path = dbg_root / "F_person_cands_bin" / split / pid_folder / cls / sub_after_class / stem / "decision.json"
                dec_path.parent.mkdir(parents=True, exist_ok=True)
                _json.dump({
                    "selected": selected_indices,
                    "max_person_per_frame": int(args.max_person_per_frame),
                    "keep_top1": bool(args.keep_top1),
                    "use_scoring": bool(args.use_scoring),
                    "export_topk": int(args.export_topk),
                    "max_xy_radius": float(args.max_xy_radius),
                    "xy_radius_percentile": float(args.xy_radius_percentile)
                }, open(dec_path, "w"), indent=2)
        except Exception:
            pass

        # (Optional) dump per-cluster scores for this frame
        try:
            if DEBUG:
                dbg_json = dbg_root / "F_person_cands_bin" / split / pid_folder / cls / sub_after_class / f"{stem}_scores.json"
                dbg_json.parent.mkdir(parents=True, exist_ok=True)
                import json as _json
                _json.dump(scores_dbg, open(dbg_json, "w"), indent=2)
        except Exception:
            pass


        # --- G) CLEAN SELECTED CANDIDATES (post-clustering clean) ---
        if args.intra_clearance > 0.0 or (_O3D and args.intra_outlier_nb > 0):
            cand = [
                clean_person_cluster(
                    c, D_models,
                    margin=args.intra_clearance,
                    nb=args.intra_outlier_nb,
                    std=args.intra_outlier_std
                ) for c in cand
            ]

        #print(f"[DBG] kept={len(cand)} sizes={[c.shape[0] for c in cand]} split_out={map_split(split)} pid={pid} tid={tid} fid={fid}")

        # Save cleaned candidates for inspection
        if DEBUG:
            F_dir = dbg_root / "F_person_cands_bin" / split / pid_folder / cls / sub_after_class / stem
            for ci, c in enumerate(cand):
                write_bin(F_dir / f"person_{ci:04d}.bin", c, include_intensity=args.include_intensity)

        '''
        # ReID3D export (XYZ by default for maximum compatibility)
        out_split = map_split(split)
        for ci, c in enumerate(cand):
            name = lreid_name(pid=pid, cam=args.camera_id, tid=tid, fid=fid+ci)
            write_bin(reid_root / out_split / name, c, include_intensity=False)

            manifest.append({
                "src_csv": str(csv_path),
                "split_src": split,
                "split_out": out_split,
                "person_folder": pid_folder,
                "class": cls,
                "pid": pid,
                "cam": args.camera_id,
                "tid": tid,
                "fid": int(fid+ci),
                "outfile": str(reid_root / out_split / name),
                "n_points": int(c.shape[0])
            })
        '''

        # === NEW: ReID3D export (LReID-style tree + lists/info) ===
        # Map logical split to ReID3D dirs: "train" -> bbox_train, ("query","gallery") -> bbox_test
        out_split = map_split(split)  # "train", "query", or "gallery"
        for ci, c in enumerate(cand):
            # ci = 0,1,... per candidate in this frame
            frame_idx_ci = frame_idx + ci
            chunk = frame_idx_ci // 30
            fid_within = frame_idx_ci % 30
            export_tid = base_tid + chunk
            name = lreid_name(pid=pid, cam=args.camera_id, tid=export_tid, fid=fid_within)


            if out_split == "train":
                subdir = Path("bbox_train") / f"{pid:04d}"
                rel_path = (subdir / name).as_posix()  # e.g., bbox_train/0007/0007C1T0003F012.bin
                abs_path = reid_root / rel_path
                write_bin(abs_path, c, include_intensity=False)

                train_paths.append(rel_path)
                train_info_rows.append([int(pid), 1, int(export_tid), int(fid_within)])

            else:
                # Both "query" and "gallery" live under bbox_test; query membership is recorded via query_IDX
                subdir = Path("bbox_test") / f"{pid:04d}"
                rel_path = (subdir / name).as_posix()
                abs_path = reid_root / rel_path
                write_bin(abs_path, c, include_intensity=False)

                test_paths.append(rel_path)
                test_info_rows.append([int(pid), 1, int(export_tid), int(fid_within)])
                if out_split == "query":
                    query_relpaths.append(rel_path)

            manifest.append({
                "src_csv": str(csv_path),
                "split_src": split,
                "split_out": out_split,
                "person_folder": pid_folder,
                "class": cls,
                "pid": pid,
                "cam": args.camera_id,
                "tid": int(export_tid),
                "fid": int(fid_within),
                "outfile": str(abs_path),
                "n_points": int(c.shape[0])
            })

        # --- PATCH 4: advance per-sequence frame counter and bump next free TID if needed
        if len(cand) > 0:
            # if this frame spilled into one or more 30-frame chunks, reserve those TIDs
            max_chunk_used = (frame_idx + (len(cand) - 1)) // 30
            next_tid_for_pid[base_key] = max(next_tid_for_pid[base_key], base_tid + max_chunk_used + 1)

        # move to the next frame for this (split,pid,cls,seq_key)
        seq_state[tid_key]["frame_idx"] = frame_idx + 1


    # Save manifest
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "manifest.json").write_text(json.dumps({
        "args": vars(args),
        "count_exported": len(manifest),
        "note": "BINs under reid3d/ are float32 XYZ with ReID3D (LReID) filename convention.",
        "items": manifest[:1000]  # cap to keep file reasonable; edit if needed
    }, indent=2))
    print(f"[OK] Wrote step BINs to: {dbg_root}")
    print(f"[OK] Wrote ReID3D-ready BINs to: {reid_root} (mapping val→{args.map_val_to}, test→{args.map_test_to})")
    print(f"[OK] Manifest: {out_root/'manifest.json'}")

    # === NEW: write ReID3D info/ lists ===
    info_dir = reid_root / "info"
    info_dir.mkdir(parents=True, exist_ok=True)

    # Deterministic ordering (important: keep alignment between txt and info)
    # We'll simply keep the order we appended during export; no re-sorting here.

    # Write train/test path lists (relative to reid_root)
    #(reid_root / "train.txt").write_text("\n".join(train_paths) + ("\n" if train_paths else ""), encoding="utf-8")
    #(reid_root / "test.txt").write_text("\n".join(test_paths) + ("\n" if test_paths else ""), encoding="utf-8")

    # --- Build LReID tracklet-range files (start end pid cam) ---
    train_ranges = build_track_ranges(train_paths, train_info_rows)
    test_ranges  = build_track_ranges(test_paths,  test_info_rows)

    # Assert camid is always 1 (single LiDAR)
    if train_ranges.size:
        assert (train_ranges[:, 3] == 1).all(), "Non-1 camid found in train ranges"
    if test_ranges.size:
        assert (test_ranges[:, 3] == 1).all(), "Non-1 camid found in test ranges"

    # Save LReID-style tracklet ranges
    np.savetxt(info_dir / "tracks_train_info.txt", train_ranges, fmt="%d")
    np.savetxt(info_dir / "tracks_test_info.txt",  test_ranges,  fmt="%d")
    #np.save(info_dir / "tracks_train_info.npy", train_ranges)
    #np.save(info_dir / "tracks_test_info.npy",  test_ranges)

    # --- Build query indices: ONE QUERY PER PERSON (PID), LReID-style ---
    # Reminder:
    #   test_ranges: [n_tracks, 4] with rows [start_idx, end_idx, pid, cam]
    #   tracks_test_info.txt is exactly this array.
    #
    # query_IDX.txt must contain indices into *rows of tracks_test_info.txt*,
    # i.e., 0-based indices into test_ranges.

    pid_to_track_idxs = {}
    for track_idx, (start_idx, end_idx, pid, cam) in enumerate(test_ranges):
        pid = int(pid)
        pid_to_track_idxs.setdefault(pid, []).append(track_idx)

    q_idx = []
    missing = 0  # keep for summary print

    # Policy: choose ONE representative tracklet per PID.
    # Here we pick the tracklet whose end index is largest (i.e., "last" track)
    # You can change to min(...) if you want "first" instead.
    for pid, t_idxs in pid_to_track_idxs.items():
        chosen = max(
            t_idxs,
            key=lambda ti: (
                test_ranges[ti][1],  # end_idx: prefer later tracklets
                test_ranges[ti][0],  # start_idx as tiebreak
                ti                   # stable index tiebreak
            )
        )
        q_idx.append(chosen)

    # Create ndarray even if empty
    query_idx = np.asarray(sorted(set(q_idx)), dtype=np.int32) if q_idx else np.zeros((0,), dtype=np.int32)

    # Save queries (0-based indices into tracks_test_info.txt)
    info_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(info_dir / "query_IDX.txt", query_idx.reshape(-1, 1), fmt="%d")
    
    #np.save   (info_dir / "query_IDX.npy", query_idx)
    #np.save   (info_dir / "query_info.npy", query_idx)  # convenience alias


    print(f"[OK] ReID3D dataset summary @ {reid_root}")
    print(f"  train: {len(train_paths)}  test: {len(test_paths)}  query: {len(query_idx)}  (missing_from_test: {missing})")
    #print(f"  wrote: train.txt, test.txt, info/tracks_*_info.(txt|npy), info/query_IDX.(txt|npy)")
    print(
    f"  train: {len(train_paths)}"
    f"  test: {len(test_paths)}"
    f"  query: {int(getattr(query_idx, 'size', len(query_idx)))}"
    f"  (missing_from_test: {int(missing)})"
    )


if __name__ == "__main__":
    main()