#!/usr/bin/env python3
# align_two_angles.py — register angle B to angle A (3D skeletons)
# - Accepts two NPZs produced by your pipeline (each may have 580 joints)
# - Normalizes to (T,P,J,3), filters 580 -> BML-MoVi-87, applies axis order / flips / scale
# - Optionally estimates temporal offset, then estimates similarity transform (s,R,t) using Umeyama
# - Applies transform to B, saves aligned outputs and (optionally) a fused average

import sys, argparse
import numpy as np

# ---------- knobs ----------
AXIS_ORDER = (0, 2, 1)     # reorder axes: e.g., (0,2,1) swaps y<->z
FLIPS      = (+1, +1, -1)  # flip signs: e.g., flip Z
SCALE_MM   = True          # mm -> meters
CENTER_AT  = "none"        # IMPORTANT: keep "none" for registration (we estimate translation)
                           # Later you can center for visualization.

# Stable joints to drive registration (subset of BML87 indices after 580->87 filtering).
# We'll pick torso + hips + shoulders + knees + ankles by index in the 87-vector.
# (These indices correspond to your 'names' list below. Adjust if needed.)
STABLE_BML87_NAMES = [
    'sternum','mhip','pelv','thor',
    'lsho','rsho','lhip','rhip','lkne','rkne','lank','rank'
]
# If your exact names differ, tweak the list or their mapping below.

# ---------- BML-MoVi-87 selection (indices inside the 580 candidate set) ----------
BML87_IDX = np.array([
    264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,
    284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,
    304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,
    324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,
    344,345,346,347,348,349,350
], dtype=int)

# BML-MoVi-87 names (length 87). (These match your earlier list; corrected a couple typos.)
BML87_NAMES = [
    'backneck','upperback','clavicle','sternum','umbilicus',
    'lfronthead','lbackhead','lback','lshom','lupperarm',
    'lelbm','lforearm','lwrithumbside','lwripinkieside',
    'lfin','lasis','lpsis','lfrontthigh','lthigh','lknem',
    'lankm','lhee','lfifthmetatarsal','ltoe','lcheek',
    'lbreast','lelbinner','lwaist','lthumb','lfrontinnerthigh',
    'linnerknee','lshin','lfirstmetatarsal','lfourthtoe','lscapula',
    'lbum','rfronthead','rbackhead','rback','rshom','rupperarm',
    'relbm','rforearm','rwrithumbside','rwripinkieside','rfin',
    'rasis','rpsis','rfrontthigh','rthigh','rknem','rankm','rhee',
    'rfifthmetatarsal','rtoe','rcheek','rbreast','relbinner',
    'rwaist','rthumb','rfrontinnerthigh','rinnerknee','rshin',
    'rfirstmetatarsal','rfourthtoe','rscapula','rbum','head',
    'mhip','pelv','thor','lank','lelb','lhip','lhan',
    'lkne','lsho','lwri','lfoo','rank','rel','rhip',
    'rhan','rkne','rsho','rwri','rfoo'
]

# Map friendly names to 87 indices
BML87_NAME_TO_IDX = {name: i for i, name in enumerate(BML87_NAMES)}

# ---------- helpers ----------
def load_kp_TPJ3(npz_path):
    data = np.load(npz_path, allow_pickle=True)
    key = "keypoints3d" if "keypoints3d" in data.files else ("keypoints" if "keypoints" in data.files else None)
    if key is None:
        raise KeyError(f"{npz_path}: no 'keypoints3d' or 'keypoints'. Keys={list(data.files)}")
    kp = data[key]
    # unwrap: many pipeline dumps are list/object arrays per frame
    if isinstance(kp, np.ndarray) and kp.dtype == object:
        kp = list(kp)
    if isinstance(kp, (list, tuple)):
        # Expect list[ (P,J,3) ] over frames; make it (T,P,J,3)
        T = len(kp)
        if T == 0:
            return np.zeros((0,1,0,3))
        # pad to max P and consistent J
        J = next((a.shape[1] for a in kp if isinstance(a, np.ndarray) and a.size), 0)
        Pmax = max(((a.shape[0] if a.size else 0) for a in kp), default=1)
        out = np.full((T, max(Pmax,1), J, 3), np.nan, dtype=float)
        for t, a in enumerate(kp):
            if isinstance(a, np.ndarray) and a.size:
                p = a.shape[0]
                out[t, :p, :, :] = a
        return out
    # numeric array
    arr = np.asarray(kp)
    # move coord axis to last if needed
    coord_axes = [i for i,s in enumerate(arr.shape) if s == 3]
    if coord_axes and coord_axes[-1] != arr.ndim-1:
        arr = np.moveaxis(arr, coord_axes[-1], -1)
    # squeeze singletons beyond 4D
    while arr.ndim > 4 and 1 in arr.shape[:-1]:
        arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[-1] == 3:  # (T,J,3) or (P,J,3)
        T = arr.shape[0]
        return arr.reshape(T, 1, arr.shape[1], 3)
    if arr.ndim == 4 and arr.shape[-1] == 3:  # (T,P,J,3)
        return arr
    raise ValueError(f"{npz_path}: unsupported kp3d shape {arr.shape}")

def preprocess(kp_TPJ3):
    """Filter 580->87 if needed, then apply axis order, flips, scaling."""
    T, P, J, C = kp_TPJ3.shape
    if J == 580:
        kp_TPJ3 = kp_TPJ3[:, :, BML87_IDX, :]  # (T,P,87,3)
        J = 87
    # axis order + flips
    kp_TPJ3 = kp_TPJ3[..., AXIS_ORDER] * np.array(FLIPS)[None, None, None, :]
    if SCALE_MM:
        kp_TPJ3 = kp_TPJ3 * 0.001
    return kp_TPJ3  # (T,P,J,3) where J is 87 or whatever it was

def pick_person(kp_TPJ3):
    """Ensure single-person tracks: pick first person if multiple."""
    if kp_TPJ3.shape[1] == 1:
        return kp_TPJ3[:, 0]  # (T,J,3)
    # fallback: choose person 0; you can add bbox-based selection here
    return kp_TPJ3[:, 0]

def nanmask_points(A, B):
    """Return boolean mask where both A and B are finite (no NaN)"""
    return np.isfinite(A).all(axis=-1) & np.isfinite(B).all(axis=-1)

def umeyama_similarity(X, Y, with_scale=True):
    """
    Find s, R, t : Y ~ s*R*X + t  (X->Y)
    X, Y: (N,3) matched points (no NaNs). Returns (s, R(3x3), t(3,))
    """
    X = np.asarray(X, float); Y = np.asarray(Y, float)
    assert X.shape == Y.shape and X.ndim == 2 and X.shape[1] == 3
    muX = X.mean(axis=0); muY = Y.mean(axis=0)
    X0 = X - muX; Y0 = Y - muY
    C = (Y0.T @ X0) / X.shape[0]
    U, S, Vt = np.linalg.svd(C)
    R = U @ Vt
    if np.linalg.det(R) < 0:  # reflection fix
        Vt[-1, :] *= -1
        R = U @ Vt
    if with_scale:
        varX = (X0**2).sum() / X.shape[0]
        s = (S.sum()) / varX if varX > 0 else 1.0
    else:
        s = 1.0
    t = muY - s * (R @ muX)
    return s, R, t

def apply_similarity(pts, s, R, t):
    """pts: (...,3); returns transformed"""
    shp = pts.shape
    P = pts.reshape(-1, 3)
    out = (s * (P @ R.T)) + t[None, :]
    return out.reshape(shp)

def time_offset_by_pelvis(A_TJ3, B_TJ3, lag_range=30):
    """
    Estimate time offset (B -> A) by minimizing distance between pelvis tracks.
    Returns best_lag where B shifted by lag aligns to A (positive lag = B starts later).
    """
    # Try to locate pelvis/mhip/pelv index
    pelvis_idx = None
    for name in ("pelv", "mhip", "sternum"):
        if name in BML87_NAME_TO_IDX:
            pelvis_idx = BML87_NAME_TO_IDX[name]
            break
    if pelvis_idx is None: return 0
    A = A_TJ3[:, pelvis_idx, :]
    B = B_TJ3[:, pelvis_idx, :]
    best_lag, best_err = 0, np.inf
    for lag in range(-lag_range, lag_range+1):
        if lag >= 0:
            Aseg = A[lag:]
            Bseg = B[:len(Aseg)]
        else:
            Bseg = B[-lag:]
            Aseg = A[:len(Bseg)]
        if len(Aseg) < 5:  # not enough overlap
            continue
        m = np.isfinite(Aseg).all(-1) & np.isfinite(Bseg).all(-1)
        if not np.any(m): continue
        err = np.mean(np.linalg.norm(Aseg[m]-Bseg[m], axis=1))
        if err < best_err:
            best_err = err; best_lag = lag
    return best_lag

def build_registration_set(A_TJ3, B_TJ3, names=BML87_NAMES, stable_names=STABLE_BML87_NAMES):
    """
    Stack frames & joints from a stable subset for registration.
    Returns X, Y as (N,3) matched points (A vs B).
    """
    use_idx = [BML87_NAME_TO_IDX[n] for n in stable_names if n in BML87_NAME_TO_IDX]
    A_sel = A_TJ3[:, use_idx, :]  # (T,K,3)
    B_sel = B_TJ3[:, use_idx, :]  # (T,K,3)
    mask = np.isfinite(A_sel).all(-1) & np.isfinite(B_sel).all(-1)
    X = A_sel[mask]  # (N,3)
    Y = B_sel[mask]
    return X, Y

def save_npz(path, **arrays):
    np.savez_compressed(path, **arrays)

# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("A_npz", help="Angle A npz (reference)")
    ap.add_argument("B_npz", help="Angle B npz (to be aligned to A)")
    ap.add_argument("--out_aligned", default="B_aligned_to_A.npz", help="Output npz for aligned B")
    ap.add_argument("--out_fused", default=None, help="Optional fused average npz (A+B)/2")
    ap.add_argument("--no_scale", action="store_true", help="Estimate rigid (no scale) instead of similarity")
    ap.add_argument("--max_lag", type=int, default=30, help="Max frames for temporal offset search")
    args = ap.parse_args()

    # Load & preprocess
    A_TPJ3 = preprocess(load_kp_TPJ3(args.A_npz))  # (T,P,J,3)
    B_TPJ3 = preprocess(load_kp_TPJ3(args.B_npz))
    A_TJ3  = pick_person(A_TPJ3)  # (T,J,3)
    B_TJ3  = pick_person(B_TPJ3)

    # Temporal alignment (optional but helpful if starts are offset)
    lag = time_offset_by_pelvis(A_TJ3, B_TJ3, lag_range=args.max_lag)
    if lag >= 0:
        A_sync = A_TJ3[lag:]
        B_sync = B_TJ3[:len(A_sync)]
    else:
        B_sync = B_TJ3[-lag:]
        A_sync = A_TJ3[:len(B_sync)]

    # Build registration correspondences from stable joints across overlapping frames
    X, Y = build_registration_set(A_sync, B_sync)
    if len(X) < 6:
        print("Warning: not enough matched points for a robust fit; proceeding anyway.")
    s, R, t = umeyama_similarity(Y, X, with_scale=(not args.no_scale))  # map B->A

    # Apply to the FULL B sequence (not just synced part), so lengths are preserved
    B_aligned = apply_similarity(B_TJ3, s, R, t)  # (T,J,3)

    # Save aligned B (and reference A) so you can visualize/animate together later
    save_npz(args.out_aligned, A_ref=A_TJ3, B_aligned=B_aligned, lag=lag, s=s, R=R, t=t)

    print(f"Saved aligned B → {args.out_aligned}")
    print(f"Transform: scale={s:.6f}, R=\n{R}\n t={t}")

    # Optional fused average (only where both are finite in overlap)
    if args.out_fused:
        # Bring A and aligned B into same temporal range (no time warp beyond shift)
        if lag >= 0:
            A_f = A_TJ3.copy()
            B_f = np.full_like(A_TJ3, np.nan)
            B_f[lag:lag+len(B_TJ3)] = B_aligned[:len(A_TJ3)-lag]
        else:
            B_f = B_aligned.copy()
            A_f = np.full_like(B_TJ3, np.nan)
            A_f[-lag:-lag+len(A_TJ3)] = A_TJ3[:len(B_TJ3)+lag]
        M = np.isfinite(A_f) & np.isfinite(B_f)
        fused = np.where(M, 0.5*(A_f+B_f), np.where(np.isfinite(A_f), A_f, B_f))
        save_npz(args.out_fused, fused=fused, A_ref=A_TJ3, B_aligned=B_aligned, lag=lag)
        print(f"Saved fused average → {args.out_fused}")

if __name__ == "__main__":
    main()
