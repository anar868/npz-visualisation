#!/usr/bin/env python3
# plot_kp2d_frame.py — plot one 2D skeleton frame on coordinates (no video background)

import sys
import numpy as np
import matplotlib.pyplot as plt

# -------- CLI --------
NPZ   = sys.argv[1] if len(sys.argv) > 1 else "4_1.npz"
FRAME = int(sys.argv[2]) if len(sys.argv) > 2 else 0

# If your y-axis grows downward (image coords), set this True to flip visually:
INVERT_Y = True

# -------- Names you shared --------
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
# full list from your snippet; we’ll use the last 56 names as source order
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

# Map COCO_25 names -> closest names in bml_movi_87 (56-joint tail)
COCO_TO_BML56 = {
    "Nose": "Head",
    "Left Eye": "lfronthead",
    "Right Eye": "rfronthead",
    "Left Ear": "lbackhead",
    "Right Ear": "rbackhead",
    "Left Little Toe": "lfourthtoe",
    "Right Little Toe": "rfourthtoe",
}

# OpenPose BODY_25-ish edges (indices correspond to OPENPOSE_25 order)
EDGES_25 = [
    (0,1),(0,15),(0,16),(15,17),(16,18),
    (1,2),(1,5),(1,8),(8,12),(8,9),
    (2,3),(3,4),
    (5,6),(6,7),
    (9,10),(10,11),(11,24),(11,22),(11,23),
    (12,13),(13,14),(14,21),(14,19),(14,20),
]

# -------- helpers --------
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

def to_T_P_J_C(kp):
    """
    Normalize any 2D kp array to (T, P, J, C).
    Accepts (T,J,C), (T,P,J,C), (P,T,J,C), (1,T,J,C), object arrays/lists, etc.
    """
    arr = np.asarray(kp)

    # squeeze extra singletons (but keep last=channels)
    while arr.ndim > 4 and 1 in arr.shape:
        arr = np.squeeze(arr)

    if arr.ndim == 3:
        T, J, C = arr.shape
        return arr.reshape(T, 1, J, C)

    if arr.ndim == 4:
        A, B, J, C = arr.shape
        # choose time axis as the larger of A/B (frames >> persons)
        return arr if A >= B else np.swapaxes(arr, 0, 1)

    # handle (1,T,J,C)-style
    if arr.ndim == 5 and arr.shape[0] == 1:
        arr = np.squeeze(arr, axis=0)
        return to_T_P_J_C(arr)

    raise ValueError(f"Unsupported kp2d shape: {arr.shape}")

def norm_name(s: str) -> str:
    return " ".join(s.lower().strip().split())

def build_index(source_names, wanted_names, alias_map=None):
    """
    Map wanted_names -> indices in source_names (case/space-normalized).
    If alias_map is provided, translate wanted_names via alias_map first.
    """
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

def to_openpose25_TPJ(kp_TPJ2or3):
    """
    Input: (T, P, J, 2/3) where J in {25,56}.
    Output: (T, P, 25, 2/3) in OpenPose-25 order.
    """
    T, P, J, C = kp_TPJ2or3.shape
    if J == 25:
        # assume COCO_25 order; reorder to OpenPose order by index positions
        idx_op = [COCO_25.index(n) for n in OPENPOSE_25]
        return kp_TPJ2or3[:, :, idx_op, :]
    if J == 56:
        idx_coco = build_index(BML56, COCO_25, alias_map=COCO_TO_BML56)  # 56 -> 25
        kp25c = kp_TPJ2or3[:, :, idx_coco, :]
        idx_op = [COCO_25.index(n) for n in OPENPOSE_25]
        return kp25c[:, :, idx_op, :]
    raise ValueError(f"Unsupported joint count J={J}; expected 25 or 56 (got {J}).")

# -------- main --------
def main():
    data = np.load(NPZ, allow_pickle=True)
    # Accept either 'keypoints2d' or 'keypoints'
    key_name = "keypoints2d" if "keypoints2d" in data.files else ("keypoints" if "keypoints" in data.files else None)
    if key_name is None:
        raise KeyError(f"No 'keypoints2d' or 'keypoints' in {NPZ}. Keys: {list(data.files)}")

    kp2d_raw = unwrap_first(data[key_name])     # pick first person/track if multiple
    kp_TPJ = to_T_P_J_C(kp2d_raw)               # (T,P,J,C)
    T, P, J, C = kp_TPJ.shape
    frame = max(0, min(FRAME, T-1))

    # Normalize to OpenPose-25 if possible
    if J in (25, 56):
        kp25 = to_openpose25_TPJ(kp_TPJ)        # (T,P,25,C)
        pts = kp25[frame, 0]                    # first person, chosen frame => (25,C)
        edges = EDGES_25
        title = f"2D OpenPose-25 — frame {frame}/{T-1}"
    else:
        pts = kp_TPJ[frame, 0]                  # (J,C) unknown skeleton, scatter only
        edges = []
        title = f"2D (J={J}) — frame {frame}/{T-1} (no edges)"

    xy = pts[:, :2]
    conf = pts[:, 2] if C >= 3 else None

    # ---- plot on coordinates (no background) ----
    fig, ax = plt.subplots(figsize=(7,7))
    if conf is not None:
        c = np.clip(conf, 0, 1)
        ax.scatter(xy[:,0], xy[:,1], s=20 + 60*c, alpha=0.3 + 0.7*c)
    else:
        ax.scatter(xy[:,0], xy[:,1], s=30)

    for i, j in edges:
        if i < xy.shape[0] and j < xy.shape[0]:
            ax.plot([xy[i,0], xy[j,0]], [xy[i,1], xy[j,1]], linewidth=2)

    ax.set_aspect('equal', adjustable='box')
    if INVERT_Y:
        ax.invert_yaxis()
    ax.set_xlabel("x"); ax.set_ylabel("y")
    ax.set_title(title + f" | persons={P} joints={xy.shape[0]}")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()