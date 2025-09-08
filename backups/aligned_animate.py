#!/usr/bin/env python3
# animate_kp3d_coords.py — animate 3D keypoints on axes (no video)
# - Works with standard NPZs (keypoints3d/keypoints) OR alignment NPZs (A_ref, B_aligned)
# - Normalizes to (T,P,J,3); filters 580→BML-MoVi-87
# - Axis reorder, flips, mm→m, optional centering
# - Overlays multiple streams (A, B) with different colors

import sys, argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa

# -------- knobs you can edit --------
AXIS_ORDER = (0, 2, 1)      # e.g. (0,2,1) swaps y<->z
FLIPS      = (+1, +1, -1)   # flip Z
SCALE_MM   = True           # mm -> m
CENTER_AT  = "pelv"         # 'pelv', 'mhip', 'sternum', or 'none'
FPS        = 30

# BML-MoVi-87 subset indices inside the 580-candidate vector
BML87_IDX = np.array([
    264,265,266,267,268,269,270,271,272,273,274,275,276,277,278,279,280,281,282,283,
    284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,
    304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,
    324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,
    344,345,346,347,348,349,350
], dtype=int)

# A minimal name list so we can find pelvis if present
BML87_NAMES = [
    'backneck','upperback','clavicle','sternum','umbilicus','lfronthead','lbackhead','lback','lshom','lupperarm',
    'lelbm','lforearm','lwrithumbside','lwripinkieside','lfin','lasis','lpsis','lfrontthigh','lthigh','lknem',
    'lankm','lhee','lfifthmetatarsal','ltoe','lcheek','lbreast','lelbinner','lwaist','lthumb','lfrontinnerthigh',
    'linnerknee','lshin','lfirstmetatarsal','lfourthtoe','lscapula','lbum','rfronthead','rbackhead','rback','rshom',
    'rupperarm','relbm','rforearm','rwrithumbside','rwripinkieside','rfin','rasis','rpsis','rfrontthigh','rthigh',
    'rknem','rankm','rhee','rfifthmetatarsal','rtoe','rcheek','rbreast','relbinner','rwaist','rthumb',
    'rfrontinnerthigh','rinnerknee','rshin','rfirstmetatarsal','rfourthtoe','rscapula','rbum','head',
    'mhip','pelv','thor','lank','lelb','lhip','lhan','lkne','lsho','lwri','lfoo','rank','rel','rhip','rhan',
    'rkne','rsho','rwri','rfoo'
]
BML87_NAME_TO_IDX = {n:i for i,n in enumerate(BML87_NAMES)}

# ------------- helpers -------------
def unwrap_first(x):
    if isinstance(x, np.ndarray) and x.dtype == object:
        if x.shape == (): return x.item()
        if x.shape[0] >= 1: return x[0]
    if isinstance(x, (list, tuple)): return x[0]
    return x

def to_T_P_J_3(kp):
    """Normalize any kp array to (T,P,J,3)."""
    if isinstance(kp, (list, tuple)):
        T = len(kp)
        if T == 0: return np.zeros((0,1,0,3))
        J = next((a.shape[1] for a in kp if isinstance(a, np.ndarray) and a.size), 0)
        Pmax = max(((a.shape[0] if isinstance(a, np.ndarray) and a.size else 0) for a in kp), default=1)
        out = np.full((T, max(Pmax,1), J, 3), np.nan, dtype=float)
        for t,a in enumerate(kp):
            if isinstance(a, np.ndarray) and a.size:
                p = a.shape[0]
                out[t,:p,:,:] = a
        return out
    arr = np.asarray(kp)
    coord_axes = [i for i,s in enumerate(arr.shape) if s == 3]
    if coord_axes and coord_axes[-1] != arr.ndim-1:
        arr = np.moveaxis(arr, coord_axes[-1], -1)
    while arr.ndim > 4 and 1 in arr.shape[:-1]:
        arr = np.squeeze(arr)
    if arr.ndim == 3 and arr.shape[-1] == 3:
        return arr.reshape(arr.shape[0], 1, arr.shape[1], 3)
    if arr.ndim == 4 and arr.shape[-1] == 3:
        return arr
    raise ValueError(f"Unsupported kp3d shape {arr.shape}")

def preprocess_TPJ3(kp_TPJ3):
    """Filter 580→87 if needed; reorder axes; flips; mm→m."""
    T,P,J,C = kp_TPJ3.shape
    if J == 580:
        kp_TPJ3 = kp_TPJ3[:,:,BML87_IDX,:]
        J = 87
    kp_TPJ3 = kp_TPJ3[..., AXIS_ORDER] * np.array(FLIPS)[None,None,None,:]
    if SCALE_MM:
        kp_TPJ3 *= 0.001
    return kp_TPJ3  # (T,P,J,3)

def to_T_J_3_one_person(kp_TPJ3):
    """Pick first person for display."""
    return kp_TPJ3[:,0,:,:]  # (T,J,3)

def center_sequence(T_J_3, joint_name):
    if joint_name.lower() == "none": return T_J_3
    idx = BML87_NAME_TO_IDX.get(joint_name, None)
    if idx is None:  # try alternates
        for cand in ("pelv","mhip","sternum","thor"):
            idx = BML87_NAME_TO_IDX.get(cand)
            if idx is not None: break
    if idx is None: return T_J_3
    ctr = T_J_3[:, idx:idx+1, :]  # (T,1,3)
    return T_J_3 - ctr

def set_axes_equal(ax, pts):
    mn = np.nanmin(pts, axis=0); mx = np.nanmax(pts, axis=0)
    ctr = (mn+mx)/2.0; r = np.max(mx-mn)/2.0
    ax.set_xlim(ctr[0]-r, ctr[0]+r)
    ax.set_ylim(ctr[1]-r, ctr[1]+r)
    ax.set_zlim(ctr[2]-r, ctr[2]+r)

def load_streams(path, streams):
    """
    Returns list of (label, T,J,3) for requested streams.
    streams: "auto" -> keypoints3d/keypoints or both A,B if present
             comma list e.g. "A,B" or "B" or "A"
    """
    data = np.load(path, allow_pickle=True)
    keys = set(data.files)

    def load_key(k):
        arr = unwrap_first(data[k])
        tpj3 = to_T_P_J_3(arr)
        return preprocess_TPJ3(tpj3)

    seqs = {}
    if "keypoints3d" in keys or "keypoints" in keys:
        k = "keypoints3d" if "keypoints3d" in keys else "keypoints"
        seqs["X"] = to_T_J_3_one_person(load_key(k))
    if "A_ref" in keys:
        seqs["A"] = to_T_J_3_one_person(load_key("A_ref"))
    if "B_aligned" in keys:
        seqs["B"] = to_T_J_3_one_person(load_key("B_aligned"))

    if streams == "auto":
        if "A" in seqs or "B" in seqs:
            wanted = [k for k in ("A","B") if k in seqs]
        elif "X" in seqs:
            wanted = ["X"]
        else:
            raise KeyError(f"No usable arrays in {path}. Keys: {list(keys)}")
    else:
        wanted = [s.strip().upper() for s in streams.split(",")]
        for s in wanted:
            if s not in seqs:
                raise KeyError(f"Requested stream '{s}' not found in {path}. Have {list(seqs.keys())}")

    out = [(w, seqs[w]) for w in wanted]
    # Optional centering (applied per stream)
    if CENTER_AT.lower() != "none":
        out = [(lab, center_sequence(TJ3, CENTER_AT)) for (lab, TJ3) in out]
    return out  # list of (label, T,J,3)

# ------------- main -------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("NPZ", help="input npz (standard or aligned)")
    ap.add_argument("--streams", default="auto", help="Which to draw: auto | X | A | B | A,B")
    ap.add_argument("--out", default=None, help="Optional MP4 output path")
    args = ap.parse_args()

    streams = load_streams(args.NPZ, args.streams)  # list[(label, T,J,3)]
    # Ensure equal T across streams by truncating to min length
    T_min = min(TJ3.shape[0] for _,TJ3 in streams)
    streams = [(lab, TJ3[:T_min]) for lab, TJ3 in streams]
    T = T_min

    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection="3d")
    colors = {"X":"C0","A":"C0","B":"C3"}  # blue for ref, red for aligned
    scatters, lines = [], []

    # global bounds
    all_pts = np.concatenate([TJ3.reshape(-1,3) for _,TJ3 in streams], axis=0)
    mn = np.nanmin(all_pts, axis=0); mx = np.nanmax(all_pts, axis=0)
    ctr = (mn+mx)/2.0; r = np.max(mx-mn)/2.0
    ax.set_xlim(ctr[0]-r, ctr[0]+r); ax.set_ylim(ctr[1]-r, ctr[1]+r); ax.set_zlim(ctr[2]-r, ctr[2]+r)
    try: ax.set_box_aspect([1,1,1])
    except: pass
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    title = ax.set_title(f"3D overlay — frame 0/{T-1}")

    # init artists
    for lab, TJ3 in streams:
        pts0 = TJ3[0]
        s = ax.scatter(pts0[:,0], pts0[:,1], pts0[:,2], s=25, color=colors.get(lab,"C7"), alpha=0.9, label=lab)
        scatters.append(s)
    ax.legend(loc="upper right")
    set_axes_equal(ax, all_pts.reshape(-1,3))

    def update(t):
        title.set_text(f"3D overlay — frame {t}/{T-1}")
        for s,(lab,TJ3) in zip(scatters, streams):
            pts = TJ3[t]
            s._offsets3d = (pts[:,0], pts[:,1], pts[:,2])
        return [title] + scatters

    anim = FuncAnimation(fig, update, frames=T, interval=1000/FPS, blit=False)

    if args.out:
        try:
            writer = FFMpegWriter(fps=FPS, bitrate=6000)
            anim.save(args.out, writer=writer)
            print("Saved", args.out)
        except Exception as e:
            print("FFmpeg writer failed; showing interactively instead. Error:", e)
            plt.show()
    else:
        plt.show()

if __name__ == "__main__":
    main()