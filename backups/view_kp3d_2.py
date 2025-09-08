#!/usr/bin/env python3
# plot_kp3d_frame.py — visualize ONE 3D skeleton frame on axes (no video)
# Mirrors the 2D script’s behavior: robust unwrap, (T,P,J,3) normalize,
# 56→COCO_25→OpenPose_25 mapping via aliases, and clean edges.

import sys
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa

# -------- CLI --------
NPZ   = sys.argv[1] if len(sys.argv) > 1 else "4_1.npz"
FRAME = int(sys.argv[2]) if len(sys.argv) > 2 else 0  # frame index

# -------- Names (same as 2D script) --------
OPENPOSE_25 = [
    "Nose","Sternum","Right Shoulder","Right Elbow","Right Wrist","Left Shoulder",
    "Left Elbow","Left Wrist","Pelvis","Right Hip","Right Knee","Right Ankle",
    "Left Hip","Left Knee","Left Ankle","Right Eye","Left Eye","Right Ear","Left Ear",
    "Left Big Toe","Left Little Toe","Left Heel","Right Big Toe","Right Little Toe","Right Heel",
]
COCO_25 = [
    "Sternum","Nose","Pelvis","Left Shoulder","Left Elbow","Left Wrist","Left Hip","Left Knee",
    "Left Ankle","Right Shoulder","Right Elbow","Right Wrist","Right Hip","Right Knee","Right Ankle",
    "Left Eye","Left Ear","Right Eye","Right Ear","Left Big Toe","Left Little Toe","Left Heel",
    "Right Big Toe","Right Little Toe","Right Heel",
]
# Your full list; most dumps put the semantic 56 at the tail:
BML_MOVI_87 = [
    "backneck","upperback","clavicle","sternum","umbilicus","lfronthead","lbackhead","lback","lshom",
    "lupperarm","lelbm","lforearm","lwrithumbside","lwripinkieside","lfin","lasis","lpsis","lfrontthigh",
    "lthigh","lknem","lankm","Left Heel","lfifthmetatarsal","Left Big Toe","lcheek","lbreast","lelbinner",
    "lwaist","lthumb","lfrontinnerthigh","linnerknee","lshin","lfirstmetatarsal","lfourthtoe","lscapula",
    "lbum","rfronthead","rbackhead","rback","rshom","rupperarm","relbm","rforearm","rwrithumbside",
    "rwripinkieside","rfin","rasis","rpsis","rfrontthigh","rthigh","rknem","rankm","Right Heel",
    "rfifthmetatarsal","Right Big Toe","rcheek","rbreast","relbinner","rwaist","rthumb","rfrontinnerthigh",
    "rinnerknee","rshin","rfirstmetatarsal","rfourthtoe","rscapula","rbum","Head","mhip","Pelvis","Sternum",
    "Left Ankle","Left Elbow","Left Hip","Left Hand","Left Knee","Left Shoulder","Left Wrist","Left Foot",
    "Right Ankle","Right Elbow","Right Hip","Right Hand","Right Knee","Right Shoulder","Right Wrist","Right Foot",
]
BML56 = BML_MOVI_87[-56:]

# COCO_25 name → closest bml_movi_87 name (head/toes aliases)
COCO_TO_BML56 = {
    "Nose": "Head",
    "Left Eye": "lfronthead",
    "Right Eye": "rfronthead",
    "Left Ear": "lbackhead",
    "Right Ear": "rbackhead",
    "Left Little Toe": "lfourthtoe",
    "Right Little Toe": "rfourthtoe",
}

# OpenPose edges (indices correspond to OPENPOSE_25 order)
EDGES_25 = [
    (0,1),(0,15),(0,16),(15,17),(16,18),
    (1,2),(1,5),(1,8),(8,12),(8,9),
    (2,3),(3,4),
    (5,6),(6,7),
    (9,10),(10,11),(11,24),(11,22),(11,23),
    (12,13),(13,14),(14,21),(14,19),(14,20),
]

# -------- helpers (mirror 2D script structure) --------
def unwrap_first(x):
    """Return a concrete ndarray for the first track/person if stored as list/object-array."""
    if isinstance(x, np.ndarray) and x.dtype == object:
        if x.shape == ():
            return x.item()
        if x.shape[0] >= 1:
            return x[0]
    if isinstance(x, (list, tuple)):
        return x[0]
    return x

def to_T_P_J_3(kp):
    """
    Normalize any 3D kp array to (T, P, J, 3).
    Accepts (T,J,3), (T,P,J,3), (P,T,J,3), (1,T,J,3), etc.
    Picks time as the larger of the first two axes (like the 2D viewer).
    """
    arr = np.asarray(kp)

    # Move coord axis (size==3) to last if needed
    coord_axes = [i for i, s in enumerate(arr.shape) if s == 3]
    if coord_axes and coord_axes[-1] != arr.ndim - 1:
        arr = np.moveaxis(arr, coord_axes[-1], -1)

    # Squeeze extra singleton dims (but keep last=coords)
    while arr.ndim > 4 and 1 in arr.shape[:-1]:
        arr = np.squeeze(arr)

    if arr.ndim == 3:
        A,B,C = arr.shape
        if C != 3:
            raise ValueError(f"Last dim must be 3 (x,y,z), got {C}")
        # Heuristic: frames >> joints
        out = arr if A >= B else np.swapaxes(arr, 0, 1)
        T,J,_ = out.shape
        return out.reshape(T, 1, J, 3)

    if arr.ndim == 4:
        A,B,J,C = arr.shape
        if C != 3:
            raise ValueError(f"Last dim must be 3 (x,y,z), got {C}")
        # Choose time axis as the larger of A,B (frames >> persons)
        return arr if A >= B else np.swapaxes(arr, 0, 1)

    # Handle (1,T,J,3)-like
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
        return to_T_P_J_3(arr)

    raise ValueError(f"Unsupported kp3d shape: {arr.shape}")

def norm_name(s: str) -> str:
    return " ".join(s.lower().strip().split())

def build_index(source_names, wanted_names, alias_map=None):
    src_map = {norm_name(n): i for i, n in enumerate(source_names)}
    idx, missing = [], []
    for w in wanted_names:
        w_src = alias_map.get(w, w) if alias_map else w
        k = norm_name(w_src)
        if k not in src_map:
            missing.append((w, w_src))
            idx.append(None)
        else:
            idx.append(src_map[k])
    if missing:
        lines = [f"  wanted '{w}' → tried '{ws}' (not found)" for (w, ws) in missing]
        raise KeyError("Missing joints in source:\n" + "\n".join(lines))
    if len(set(idx)) != len(idx):
        raise ValueError("Non-unique indices in mapping")
    return idx

def to_openpose25_TPJ(kp_TPJ3):
    """
    Input: (T, P, J, 3) where J in {25,56}.
    Output: (T, P, 25, 3) in OpenPose-25 order.
    """
    T, P, J, C = kp_TPJ3.shape
    assert C == 3, f"Expect 3D coords; got C={C}"
    if J == 25:
        idx_op = [COCO_25.index(n) for n in OPENPOSE_25]
        return kp_TPJ3[:, :, idx_op, :]
    if J == 56:
        idx_coco = build_index(BML56, COCO_25, alias_map=COCO_TO_BML56)  # 56 -> 25
        kp25c = kp_TPJ3[:, :, idx_coco, :]
        idx_op = [COCO_25.index(n) for n in OPENPOSE_25]
        return kp25c[:, :, idx_op, :]
    raise ValueError(f"Unsupported joint count J={J}; expected 25 or 56 (got {J}).")

def set_axes_equal(ax, pts):
    """
    Make X/Y/Z have the same scale so shapes aren't visually stretched.
    Doesn't modify the data, only the axis limits.
    """
    mn = pts.min(axis=0).astype(float)
    mx = pts.max(axis=0).astype(float)
    ctr = (mn + mx) / 2.0
    radius = np.max(mx - mn) / 2.0  # largest half-range across axes
    ax.set_xlim(ctr[0] - radius, ctr[0] + radius)
    ax.set_ylim(ctr[1] - radius, ctr[1] + radius)
    ax.set_zlim(ctr[2] - radius, ctr[2] + radius)
    
# -------- main --------
def main():
    data = np.load(NPZ, allow_pickle=True)
    key = "keypoints3d" if "keypoints3d" in data.files else ("keypoints" if "keypoints" in data.files else None)
    if key is None:
        raise KeyError(f"No 'keypoints3d' or 'keypoints' in {NPZ}. Keys: {list(data.files)}")

    kp3d_raw = unwrap_first(data[key])      # pick first person/track if multiple
    kp_TPJ3  = to_T_P_J_3(kp3d_raw)         # (T,P,J,3)
    T, P, J, C = kp_TPJ3.shape
    f = max(0, min(FRAME, T-1))

    # Try mapping to OpenPose-25 for edges; else scatter only
    if J in (25, 56):
        kp25 = to_openpose25_TPJ(kp_TPJ3)   # (T,P,25,3)
        pts = kp25[f, 0]                    # (25,3)
        edges = EDGES_25
        title = f"3D OpenPose-25 — frame {f}/{T-1}"
    else:
        pts = kp_TPJ3[f, 0]                 # (J,3)
        edges = []
        title = f"3D (J={J}) — frame {f}/{T-1} (no edges)"

    # Plot as-is (no axis flips/scaling to match the 2D viewer’s philosophy)
    fig = plt.figure(figsize=(7,7))
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(pts[:,0], pts[:,1], pts[:,2], s=25)

    for i, j in edges:
        if i < len(pts) and j < len(pts):
            xi, yi, zi = pts[i]; xj, yj, zj = pts[j]
            ax.plot([xi,xj],[yi,yj],[zi,zj])

    # Nice cube-ish bounds
    mn = pts.min(axis=0); mx = pts.max(axis=0)
    span = mx - mn
    pad = span * 0.2 + 1e-6
    ax.set_xlim(mn[0]-pad[0], mx[0]+pad[0])
    ax.set_ylim(mn[1]-pad[1], mx[1]+pad[1])
    ax.set_zlim(mn[2]-pad[2], mx[2]+pad[2])
    try:
        ax.set_box_aspect([1,1,1])
    except Exception:
        pass
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    ax.set_title(title + f" | persons={P} joints={pts.shape[0]}")
    set_axes_equal(ax, pts)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()